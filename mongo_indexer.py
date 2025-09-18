# SPDX-License-Identifier: MIT
# mongo_indexer.py
from __future__ import annotations

import base64, io, os, re, json, time, math, shutil, tempfile, hashlib, zipfile, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime, timezone

# third-party
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import tiktoken
import lxml.etree as ET
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string

from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch 
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.errors import CollectionInvalid
from pymongo import ASCENDING, DESCENDING
try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _NoStreamlit:
        def cache_resource(self, *a, **k):
            def _wrap(fn): return fn
            return _wrap
        def cache_data(self, *a, **k):
            def _wrap(fn): return fn
            return _wrap
        def write(self, *a, **k): pass
    st = _NoStreamlit()  # minimal shim

# ????????????????????????? CONFIG (mirrors your project) ?????????????????????????
BASE_DIR            = "/root/Downloads/Jupyterlab/Ankur_Notebooks/RAG/DEV_ragflow/"
BASE_DATA_DIR       = "/root/Downloads/Jupyterlab/Ankur_Notebooks/RAG/DEV_ragflow/collections_mongo"
DB_NAME             = "RAG"
ATLAS_VECTOR_INDEX  = "vector_index_demo"

DEFAULT_AZURE_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-cognitive-eoahf.openai.azure.com/")
DEFAULT_GPT_DEPLOYMENT     = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4.1")
DEFAULT_VISION_DEPLOYMENT  = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4.1-mini")
DEFAULT_EMBED_DEPLOYMENT   = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
AZURE_API_VERSION          = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("ADV_API_KEY", "")
ATLAS_CONNECTION_STR       = os.getenv("ATLAS_CONNECTION_STR", "")

MAX_PDF_PAGES       = 1000
TABLE_CHUNK_FORMAT  = "markdown"  # "markdown" | "json" | "tsv"

_CHART_NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
{
  "fields": [
    {
      "numDimensions": 3072,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "filepath",
      "type": "filter"
    },
    {
      "path": "page",
      "type": "filter"
    },
    {
      "path": "type",
      "type": "filter"
    }
  ]
}
def _embedding_dims_from_env(default: int = 3072) -> int:
    # Try to infer dimensions from deployment name; fall back to 3072
    dep = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "") or os.getenv("EMBEDDING", "")
    dep = dep.split(":")[-1] if ":" in dep else dep
    name = dep.lower()
    if "text-embedding-3-large" in name:  # OpenAI
        return 3072
    if "text-embedding-3-small" in name or "ada-002" in name:
        return 1536
    return default
def _ensure_collection_exists(client: MongoClient, db_name: str, coll_name: str):
    try:
        client[db_name].create_collection(coll_name)
    except CollectionInvalid:
        pass  # already exists
    except Exception:
        # Most servers auto-create on first insert; swallow other errors here
        pass

def _ensure_vector_index(
    client: MongoClient,
    db_name: str,
    coll_name: str,
    *,
    index_name: str = ATLAS_VECTOR_INDEX,
    dims: int | None = None,
    filter_paths: list[str] | None = None,
):
    """
    Create a vectorSearch index if it doesn't exist.
    Uses LangChain's default document shape:
      - text  -> "text"
      - embedding -> "embedding"
      - metadata -> "metadata" (so prefilter paths must be "metadata.*")
    """
    _ensure_collection_exists(client, db_name, coll_name)

    coll = client[db_name][coll_name]
    dims = dims or _embedding_dims_from_env()
    # match your requested filters, but under LangChain's default "metadata." prefix
    filter_paths = filter_paths or [
        "metadata.filepath",
        "metadata.filename",
        "metadata.file_ext",
        "metadata.type",
        "metadata.page_number",  # for chunks (main collection)
    ]


    # If the index already exists, do nothing
    try:
        existing = list(coll.list_search_indexes())
        if any(ix.get("name") == index_name for ix in existing):
            return
    except Exception:
        # If list fails (older server/driver), just try to create.
        pass

    # Build index model (vector + filters)
    definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": int(dims),
                "similarity": "cosine",
            },
            *({"type": "filter", "path": p} for p in filter_paths),
        ]
    }

    model = SearchIndexModel(
        name=index_name,
        definition=definition,
        type="vectorSearch",
    )

    # This call is async server-side; it returns immediately.
    coll.create_search_index(model=model)

def get_mongo_vector_store_cached(collection: str) -> MongoDBAtlasVectorSearch:
    if not ATLAS_CONNECTION_STR:
        raise RuntimeError("ATLAS_CONNECTION_STR is not set in environment.")
    client = MongoClient(ATLAS_CONNECTION_STR)

    # pick the right dimension for your embed model
    dims = 3072  # text-embedding-3-large ? 3072; use 1536 for -3-small

    return MongoDBAtlasVectorSearch(
        collection         = client[DB_NAME][collection],
        embedding          = get_embedding_cached(),
        index_name         = ATLAS_VECTOR_INDEX,
        relevance_score_fn = "cosine",
        dimensions         = dims,            # <-- enables auto-create if missing
        auto_create_index  = True,
        auto_index_timeout = 600,             # wait up to 10 min for index ready
    )
    
@st.cache_resource(show_spinner=False)
def get_embedding_cached():
    return AzureOpenAIEmbeddings(
        azure_deployment   = DEFAULT_EMBED_DEPLOYMENT,
        azure_endpoint     = DEFAULT_AZURE_ENDPOINT,
        openai_api_key     = key,
        openai_api_version = AZURE_API_VERSION,
    )
# ????????????????????????? utilities ?????????????????????????
def _mkdir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _iso(ts: float | int | None) -> str | None:
    if ts is None: return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()

def _sha256_of_file(p: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
# --- duplicate helpers ---
def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _norm_cell(s: Any) -> str:
    # robust string normalization for semantic hashing
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return re.sub(r"\s+", " ", s.strip())

def _excel_semantic_sha256(path: Path) -> str:
    """
    Hash the *table content* (not workbook metadata or internal XML order).
    - Includes header row and values for each sheet
    - Drops fully empty rows/cols
    - Normalizes whitespace and NaNs
    """
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        h = hashlib.sha256()
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet, header=0, dtype=object)
            except Exception:
                continue
            if df is None or df.shape[1] == 0:
                continue
            df = df.replace({np.nan: ""})
            df = df.applymap(_norm_cell)
            # drop fully-empty rows/cols
            df = df.loc[~(df == "").all(axis=1)]
            df = df.loc[:, ~(df == "").all(axis=0)]
            # nothing left?
            if df.shape[0] == 0 and df.shape[1] == 0:
                continue

            # Sheet identity + shape to prevent accidental collisions
            h.update(b"\n--SHEET--\n"); h.update(sheet.encode("utf-8"))
            h.update(f"|shape={df.shape[0]}x{df.shape[1]}".encode("utf-8"))

            # header
            h.update(b"\nH:")
            h.update(("\t".join([_norm_cell(c) for c in df.columns.tolist()])).encode("utf-8"))

            # rows (row-wise, tab-separated)
            for row in df.itertuples(index=False, name=None):
                h.update(b"\nR:")
                h.update(("\t".join(row)).encode("utf-8"))
        return h.hexdigest()
    except Exception:
        return ""

def _file_fs_meta(p: Path) -> Dict[str, Any]:
    stt = p.stat()
    return {
        "file_name": p.name,
        "file_ext": p.suffix.lower(),
        "file_stem": p.stem,
        "file_path": str(p),
        "file_size_bytes": stt.st_size,
        "file_ctime_iso": _iso(stt.st_ctime),
        "file_mtime_iso": _iso(stt.st_mtime),
        "file_atime_iso": _iso(stt.st_atime),
        "file_sha256": _sha256_of_file(p),
    }

def _safe_json_block(txt: str, fallback: Any = None) -> Any:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return fallback
    try:
        return json.loads(m.group(0))
    except Exception:
        return fallback

def _df_to_text(df: pd.DataFrame, *, fmt: str = TABLE_CHUNK_FORMAT) -> str:
    if fmt == "markdown":
        return df.to_markdown(index=False)
    if fmt == "json":
        return df.to_json(orient="records")
    return df.to_csv(sep="\t", index=False)

def _tokenizer(model_name: str = DEFAULT_GPT_DEPLOYMENT):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

# ????????????????????????? splitter policy (your exact spec) ?????????????????????????
def _split_params(page_lengths: List[int]) -> tuple[int, int]:
    if not page_lengths:
        return 1000, 100
    tgt = int(np.percentile(page_lengths, 60)) // 2
    chunk_size = int(np.clip(tgt, 400, 2000))
    chunk_overlap = max(int(chunk_size * 0.1), 40)
    return chunk_size, chunk_overlap

# ????????????????????????? chart helpers ?????????????????????????
def _iter_chart_series(xlsx_path: str):
    with zipfile.ZipFile(xlsx_path) as z:
        chart_files = [n for n in z.namelist() if n.startswith("xl/charts/") and n.endswith(".xml")]
        if not chart_files:
            return
        excel = pd.ExcelFile(xlsx_path, engine="openpyxl")
        for cid, cf in enumerate(chart_files, 1):
            root = ET.fromstring(z.read(cf))
            for sid, ser in enumerate(root.findall(".//c:ser", _CHART_NS), 1):
                f_node = ser.find(".//c:val//c:f", _CHART_NS)
                if f_node is None:
                    continue
                rng = f_node.text.replace("$", "")
                sheet, addr = rng.split("!")
                col_letters = re.findall(r"[A-Z]+", addr)[0]
                rows = [int(n) for n in re.findall(r"\d+", addr)]
                start_row, end_row = rows[0] - 1, rows[-1] - 1
                col_idx = column_index_from_string(col_letters) - 1
                df = excel.parse(sheet.replace("'", ""), header=None, engine="openpyxl")
                raw = df.iloc[start_row:end_row + 1, col_idx]
                ser = pd.to_numeric(raw, errors="coerce").dropna().reset_index(drop=True)
                if ser.empty:
                    continue
                yield cid, sid, sheet, ser


# ????????????????????????? LLM helpers (OCR + summary) ?????????????????????????
def _get_cached_vision_llm() -> AzureChatOpenAI:
    key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("ADV_API_KEY", "")
    if hasattr(st, "cache_resource"):
        @st.cache_resource(show_spinner=False)
        def _get():
            return AzureChatOpenAI(
                azure_deployment   = DEFAULT_VISION_DEPLOYMENT,
                azure_endpoint     = DEFAULT_AZURE_ENDPOINT,
                openai_api_key     = key,
                openai_api_version = AZURE_API_VERSION,
                temperature        = 0.0,
                request_timeout    = 20,
                max_retries        = 0,
            )
        return _get()
    return AzureChatOpenAI(
        azure_deployment   = DEFAULT_VISION_DEPLOYMENT,
        azure_endpoint     = DEFAULT_AZURE_ENDPOINT,
        openai_api_key     = key,
        openai_api_version = AZURE_API_VERSION,
        temperature        = 0.0,
        request_timeout    = 20,
        max_retries        = 0,
    )

def _ocr_page_with_gpt(page: fitz.Page, zoom: float = 2.0) -> str:
    llm = _get_cached_vision_llm()
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    messages = [
        {"role": "system", "content":
            "You are a world-class OCR engine. Transcribe ALL legible text. "
            "Preserve headings, lists and tables; return only markdown. "
            "If blank, return '----'."},
        {"role": "user", "content": [{"type": "image_url",
                                      "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]},
    ]
    delay = 2.0
    for attempt in range(4):
        try:
            return llm.invoke(messages).content.strip()
        except Exception:
            if attempt == 3:
                return "----"
            time.sleep(delay)
            delay = min(delay * 1.7, 12.0)

def _preview_blocks_for_llm(path: Path, *, max_rows: int = 20) -> tuple[List[str], List[dict]]:
    txt_blocks, images = [], []
    ext = path.suffix.lower()
    if ext == ".pdf":
        pdf = fitz.open(path)
        PAGE_LIMIT = min(10, len(pdf))
        for pg in pdf[:PAGE_LIMIT]:
            pix = pg.get_pixmap(dpi=110, alpha=False)
            images.append({"type": "image_url",
                           "image_url": {"url": "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()}})
        pdf.close()
    elif ext in (".ppt", ".pptx"):
        pdf = _convert_office_to_pdf(path)
        if pdf:
            return _preview_blocks_for_llm(pdf)
    elif ext in (".xlsx", ".xls"):
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            for sheet in xls.sheet_names:
                try:
                    df = xls.parse(sheet, header=0, engine="openpyxl")
                except Exception:
                    continue
                if df.empty or df.shape[1] == 0:
                    continue
                head = df.head(max_rows)
                tail = df.tail(max_rows)
                md = pd.concat([head, tail]).to_markdown(index=False)
                txt_blocks.append(f"## Sheet: {sheet}\n{md}")
        except Exception:
            pass
    return txt_blocks, images

def _summarise_file_llm(path: Path) -> tuple[str, List[str]]:
    txts, imgs = _preview_blocks_for_llm(path)
    system = (
        "You are an expert documentation assistant and summarizer. "
        "Generate ONE concise paragraph (~120 words) and list 15-20 comma-separated keywords. "
        "Respond strictly in JSON with keys: {summary:str, keywords:list[str]}"
    )
    user = [{"type": "text", "text": "File preview follows."}]
    for t in txts:
        user.append({"type": "text", "text": t})
    user.extend(imgs)
    raw = _get_cached_vision_llm().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ).content
    data = _safe_json_block(raw, fallback={"summary": "", "keywords": []})
    return data.get("summary", "").strip(), [k.strip() for k in data.get("keywords", [])]

# ????????????????????????? Mongo Vector wrapper ?????????????????????????
def _embedding_client() -> AzureOpenAIEmbeddings:
    if hasattr(st, "cache_resource"):
        @st.cache_resource(show_spinner=False)
        def _get():
            return AzureOpenAIEmbeddings(
                azure_deployment   = DEFAULT_EMBED_DEPLOYMENT,
                azure_endpoint     = DEFAULT_AZURE_ENDPOINT,
                openai_api_key     = key,
                openai_api_version = AZURE_API_VERSION,
            )
        return _get()
    return AzureOpenAIEmbeddings(
        azure_deployment   = DEFAULT_EMBED_DEPLOYMENT,
        azure_endpoint     = DEFAULT_AZURE_ENDPOINT,
        openai_api_key     = key,
        openai_api_version = AZURE_API_VERSION,
    )

def _atlas_client() -> MongoClient:
    ATLAS_CONNECTION_STR =  os.getenv("ATLAS_CONNECTION_STR", "")
    if not ATLAS_CONNECTION_STR:
        raise RuntimeError("ATLAS_CONNECTION_STR is not set.")
    return MongoClient(ATLAS_CONNECTION_STR)

def _vector_store(collection: str) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection          = _atlas_client()[DB_NAME][collection],
        embedding           = _embedding_client(),
        index_name          = ATLAS_VECTOR_INDEX,
        relevance_score_fn  = "cosine",
    )

# ????????????????????????? Office ? PDF ?????????????????????????
def _convert_office_to_pdf(src: Path) -> Optional[Path]:
    outdir = src.parent
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", str(src), "--outdir", str(outdir)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )
        pdf = outdir / f"{src.stem}.pdf"
        return pdf if pdf.exists() else None
    except Exception:
        return None

# ????????????????????????? PDF metadata helpers ?????????????????????????
def _pdf_doc_meta(doc: fitz.Document) -> Dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "pdf_title": meta.get("title"),
        "pdf_author": meta.get("author"),
        "pdf_subject": meta.get("subject"),
        "pdf_keywords": meta.get("keywords"),
        "pdf_creator": meta.get("creator"),
        "pdf_producer": meta.get("producer"),
        "pdf_creationDate": meta.get("creationDate"),
        "pdf_modDate": meta.get("modDate"),
        "pdf_encrypted": doc.is_encrypted,
        "pdf_page_count": doc.page_count,
    }

def _page_geom(page: fitz.Page) -> Dict[str, Any]:
    rect = page.rect
    return {
        "page_width_pt": rect.width,
        "page_height_pt": rect.height,
        "page_rotation": page.rotation,
        "page_mediabox": [rect.x0, rect.y0, rect.x1, rect.y1],
    }

def _page_label(doc: fitz.Document, i: int, page: fitz.Page) -> Optional[str]:
    try:
        return page.get_label()
    except Exception:
        try:
            labels = doc.get_page_labels()
            if labels and i < len(labels): return labels[i]
        except Exception:
            return None
    return None

def _page_hash(page: fitz.Page) -> str:
    try:
        txt = page.get_text("text") or ""
        if txt.strip():
            return hashlib.sha1(txt.encode("utf-8")).hexdigest()
    except Exception:
        pass
    try:
        pix = page.get_pixmap(dpi=50, alpha=False)
        return hashlib.sha1(pix.tobytes()).hexdigest()
    except Exception:
        return ""

# ????????????????????????? Ingestion Manager ?????????????????????????
ProgressCb = Optional[Callable[[Dict[str, Any]], None]]

@dataclass
class IngestConfig:
    force_ocr_ppt: bool = True
    table_format: str = TABLE_CHUNK_FORMAT
    max_pdf_pages: int = MAX_PDF_PAGES

class MongoIndexer:
    """
    Reusable indexing + collection manager for MongoDB Atlas Vector Search.
    - Raw files + metadata.json live under BASE_DATA_DIR/<collection>
    - Vectors stored in Atlas collection <collection>
    - File summaries/keywords in Atlas collection <collection>_filemeta
    """

    def __init__(self,
                 base_data_dir: str = BASE_DATA_DIR,
                 db_name: str = DB_NAME,
                 atlas_index: str = ATLAS_VECTOR_INDEX):
        self.base = _mkdir(base_data_dir)
        self.db_name = db_name
        self.index_name = atlas_index

    # filesystem / metadata
    def list_collections(self) -> List[str]:
        return sorted([d.name for d in self.base.iterdir() if d.is_dir()])

    def collection_dir(self, name: str) -> Path:
        return _mkdir(self.base / name)

    def metadata_path(self, name: str) -> Path:
        return self.collection_dir(name) / "metadata.json"

    def load_metadata(self, name: str) -> Dict[str, Any]:
        p = self.metadata_path(name)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def save_metadata(self, name: str, meta: Dict[str, Any]) -> None:
        self.metadata_path(name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_collection(self, name: str) -> None:
        self.collection_dir(name)
        client = _atlas_client()
        dims = _embedding_dims_from_env()
    
        _ensure_vector_index(client, self.db_name, name, index_name=self.index_name, dims=dims)
        _ensure_vector_index(client, self.db_name, f"{name}_filemeta", index_name=self.index_name, dims=dims)
    
        # NEW: normal Mongo index for fast lookups / uniqueness on filepath
        try:
            client[self.db_name][f"{name}_filemeta"].create_index(
                [("metadata.filepath", ASCENDING)], unique=True, sparse=True
            )
        except Exception:
            pass

    def drop_collection(self, name: str, *, preserve_files: bool = True) -> None:
        client = _atlas_client()
        client[self.db_name][name].delete_many({})
        client[self.db_name][f"{name}_filemeta"].delete_many({})
        mp = self.metadata_path(name)
        if mp.exists(): mp.unlink()
        if not preserve_files:
            coll_dir = self.collection_dir(name)
            for child in coll_dir.iterdir():
                try:
                    if child.is_file():
                        child.unlink(missing_ok=True)
                except Exception:
                    pass

    # streamlit-friendly uploads
    def ingest_uploaded_files(
    self, uploaded_files: Iterable[Any], collection: str,
    *, on_status: ProgressCb = None, on_file: ProgressCb = None, on_page: ProgressCb = None,
    cfg: IngestConfig = IngestConfig(),
    ) -> None:
        saved_paths: List[Path] = []
        coll_dir = self.collection_dir(collection)
        _, indexed_hashes, _ = self.get_indexed_signatures(collection)

        seen_batch_hashes: set[str] = set()

        for uf in uploaded_files:
            raw = uf.read()
            sha = _sha256_bytes(raw).lower()
            if sha in indexed_hashes or sha in seen_batch_hashes:
                if on_status: on_status({"status": f"skip duplicate (SHA) : {uf.name}"})
                continue
            dest = coll_dir / uf.name
            dest.write_bytes(raw)
            saved_paths.append(dest)
            seen_batch_hashes.add(sha)

        if saved_paths:
            self.ingest_paths([str(p) for p in saved_paths], collection,
                            on_status=on_status, on_file=on_file, on_page=on_page, cfg=cfg)

    # batch ingest
    def ingest_paths(
    self, file_paths: Iterable[str], collection: str,
    *, on_status: ProgressCb = None, on_file: ProgressCb = None, on_page: ProgressCb = None,
    cfg: IngestConfig = IngestConfig(),
    ) -> None:
        paths: List[Path] = []
        coll_dir = self.collection_dir(collection)

        # Copy materialization (as before)
        for p in file_paths:
            P = Path(p)
            if P.suffix.lower() == ".zip":
                with zipfile.ZipFile(P) as zf:
                    zf.extractall(coll_dir)
                    for n in zf.namelist():
                        fp = coll_dir / n
                        if fp.is_file():
                            paths.append(fp)
            else:
                dest = coll_dir / P.name
                if P.resolve() != dest.resolve():
                    shutil.copy2(P, dest)
                paths.append(dest)

        # Dedupe against DB and within batch
        _, indexed_hashes, _ = self.get_indexed_signatures(collection)
        seen_batch_hashes: set[str] = set()

        total = len(paths)
        for i, p in enumerate(paths, 1):
            if on_file: on_file({"file": p.name, "idx": i, "total": total})

            try:
                sha = _sha256_of_file(p).lower()
            except Exception:
                sha = ""

            if sha and (sha in indexed_hashes or sha in seen_batch_hashes):
                if on_status: on_status({"status": f"skip duplicate (SHA) : {p.name}"})
                continue
            seen_batch_hashes.add(sha)

            # Only now remove the previous records for this filepath (true reindex)
            client = _atlas_client()
            client[self.db_name][collection].delete_many({"metadata.filepath": str(p)})
            client[self.db_name][f"{collection}_filemeta"].delete_many({"metadata.filepath": str(p)})

            try:
                self._process_and_add_file(p, collection, on_status=on_status, on_page=on_page, cfg=cfg)
            except Exception as e:
                if on_status: on_status({"status": f"error: {p.name}: {e}"})


    def list_filemeta(self, collection: str, limit: int = 50) -> list[dict]:
        # Exclude only the big fields; allow any shape (flattened or nested).
        proj = {"text": 0, "embedding": 0}
        return list(
            self._filemeta_coll(collection)
            .find({}, proj)
            .sort([("_id", DESCENDING)])
            .limit(int(limit))
        )



    def _filemeta_coll(self, collection: str):
        return _atlas_client()[self.db_name][f"{collection}_filemeta"]

    def get_indexed_signatures(self, collection: str) -> tuple[set[str], set[str], set[str]]:
        """
        Returns (indexed_paths, indexed_file_hashes, indexed_semantic_hashes).
        """
        paths, hashes, sems = set(), set(), set()
        cur = self._filemeta_coll(collection).find(
            {}, {"metadata.filepath": 1, "metadata.file_sha256": 1, "metadata.semantic_sha256": 1}
        )
        for d in cur:
            md = d.get("metadata", {})
            p = md.get("filepath")
            h = md.get("file_sha256")
            s = md.get("semantic_sha256")
            if p: paths.add(os.path.abspath(p))
            if isinstance(h, str) and h: hashes.add(h.lower())
            if isinstance(s, str) and s: sems.add(s.lower())
        return paths, hashes, sems


    # core per-file pipeline
    def _process_and_add_file(
        self,
        path: Path,
        collection: str,
        *,
        on_status: ProgressCb,
        on_page:   ProgressCb,
        cfg: IngestConfig,
    ) -> None:
        def log(m: str):
            if on_status: on_status({"status": m})

        stem, ext = path.stem, path.suffix.lower()
        file_meta = _file_fs_meta(path)  # size, times, sha256, etc.
        entry: Dict[str, Any] = {
            # filesystem
            "filepath": str(path),
            "filename": path.name,
            "file_ext": file_meta["file_ext"],
            "file_stem": file_meta["file_stem"],
            "file_dir": str(path.parent),
            "file_size_bytes": file_meta["file_size_bytes"],
            "file_sha256": file_meta["file_sha256"],
            "file_ctime_iso": file_meta["file_ctime_iso"],
            "file_mtime_iso": file_meta["file_mtime_iso"],
            "file_atime_iso": file_meta["file_atime_iso"],
            # provenance
            "collection": collection,
            "indexer_version": "v2",         # bump when schema changes
            "ingest_source": "path_scan",    # set "upload" in ingest_uploaded_files -> ingest_paths if you like
            "type": None,                    # will be set per branch
        }

        errors: List[str] = []

        # Office ? PDF ? recurse
        if ext in (".doc", ".docx", ".ppt", ".pptx", ".rtf"):
            converted = _convert_office_to_pdf(path)
            if not converted:
                entry.update(skipped=True, reason="conversion_failed")
                self._write_filemeta_doc(collection, entry, summary="conversion failed", keywords=["conversion_failed"])
                return
            self._process_and_add_file(
                converted, collection,
                on_status=on_status, on_page=on_page,
                cfg=IngestConfig(force_ocr_ppt=True, table_format=cfg.table_format, max_pdf_pages=cfg.max_pdf_pages)
            )
            return
        # PDF
        if ext == ".pdf":
            try:
                doc = fitz.open(path)
            except Exception as e:
                entry.update(skipped=True, reason=f"pdf-open-failed: {e}")
                self._write_filemeta_doc(collection, entry, summary="pdf open failed", keywords=["pdf","open_failed"])
                return
    
            if doc.page_count > cfg.max_pdf_pages:
                entry.update(
                    skipped=True, reason="too_many_pages",
                    num_pages=doc.page_count, **_pdf_doc_meta(doc)
                )
                self._write_filemeta_doc(collection, entry, summary="skipped: too many pages", keywords=["pdf","too_many_pages"])
                doc.close()
                return
    
            pdf_meta = _pdf_doc_meta(doc)
            ocr_flags: List[bool] = []
            page_texts: List[str] = []
            for page in doc:
                raw = page.get_text("text").strip()
                did_ocr = False
                if cfg.force_ocr_ppt or len(raw) < 50:
                    try:
                        raw = _ocr_page_with_gpt(page)
                        did_ocr = True
                    except Exception as e:
                        errors.append(f"OCR p.{page.number+1}: {e}")
                        raw = "----"
                page_texts.append(raw)
                ocr_flags.append(did_ocr)
    
            chunk_size, chunk_overlap = _split_params([len(t) for t in page_texts])
            step = max(1, chunk_size - chunk_overlap)
            docs_to_add: List[Document] = []
    
            for i, page in enumerate(doc, 1):
                text = page_texts[i - 1]
                geom = _page_geom(page)
                label = _page_label(doc, i - 1, page)
                p_hash = _page_hash(page)
                page_char_len = len(text)
                starts = list(range(0, page_char_len, step)) or [0]
                total_chunks_page = len(starts)
    
                for j, start in enumerate(starts):
                    end = min(start + chunk_size, page_char_len)
                    chunk_txt = text[start:end]
                    meta = {
                        **file_meta, **pdf_meta, **geom,
                        "type": "pdf_page_chunk",
                        "filepath": str(path),
                        "filename": path.name,
                        "page_number": i,
                        "page_label": label,
                        "page_hash": p_hash,
                        "page_char_len": page_char_len,
                        "chunk_idx_in_page": j,
                        "total_chunks_on_page": total_chunks_page,
                        "chunk_char_start": start,
                        "chunk_char_end": end,
                        "chunk_size_chars": chunk_size,
                        "chunk_overlap_chars": chunk_overlap,
                        "split_policy": "dynamic_60p_to_half_avg",
                    }
                    raw_id = f"{path}|p{i}|{start}-{end}"
                    meta["chunk_id"] = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                    docs_to_add.append(Document(page_content=chunk_txt or " ", metadata=meta))
    
                if on_page:
                    on_page({"file": path.name, "page": i, "total_pages": doc.page_count, "ocr": (cfg.force_ocr_ppt or len(text) < 50)})
    
            doc.close()
            self._add_documents_with_retry(_vector_store(collection), docs_to_add, on_status=on_status)
    
            summary, keywords = _summarise_file_llm(path)
            entry.update(
                type="pdf",
                # PDF meta
                **pdf_meta,                          # title/author/producer/... already built
                # File-level stats
                num_pages=len(page_texts),
                ocr_pages=int(sum(ocr_flags)),
                ocr_ratio=(float(sum(ocr_flags)) / max(1, len(ocr_flags))),
                total_chunks=len(docs_to_add),
                chunk_size_chars=chunk_size,
                chunk_overlap_chars=chunk_overlap,
                # Diagnostics
                errors=errors,
                error_count=len(errors),
            )

            self._write_filemeta_doc(collection, entry, summary=summary, keywords=keywords)
            return
        # Excel
        elif ext in (".xlsx", ".xls"):
            # compute semantic hash up front
            sem_sha = _excel_semantic_sha256(path)
            if sem_sha:
                entry["semantic_sha256"] = sem_sha
                # check for an existing workbook with identical sheet content
                existing = self._filemeta_coll(collection).find_one({"metadata.semantic_sha256": sem_sha})
                if existing and existing.get("metadata", {}).get("filepath") != str(path):
                    # Reuse prior summary/keywords, avoid embeddings entirely
                    prior_md = existing.get("metadata", {})
                    entry.update(skipped=True, reason="duplicate_semantic", duplicate_of=prior_md.get("filepath"))
                    self._write_filemeta_doc(
                        collection, entry,
                        summary=prior_md.get("summary", "duplicate of existing workbook"),
                        keywords=prior_md.get("keywords", ["excel", "duplicate"])
                    )
                    return

            sheet_chunks: List[Document] = []
            chart_tables: List[Document] = []
            bad_sheets: List[str] = []
            wb_props = {}
            try:
                wb = load_workbook(path, read_only=True, data_only=True)
                wp = wb.properties
                wb_props = {
                    "xlsx_title": wp.title,
                    "xlsx_subject": wp.subject,
                    "xlsx_creator": wp.creator,
                    "xlsx_last_modified_by": wp.lastModifiedBy,
                    "xlsx_created_iso": wp.created.isoformat() if wp.created else None,
                    "xlsx_modified_iso": wp.modified.isoformat() if wp.modified else None,
                    "xlsx_keywords": wp.keywords,
                }
                wb.close()
            except Exception:
                pass

            xls = pd.ExcelFile(path, engine="openpyxl")
            for s_idx, sheet_name in enumerate(xls.sheet_names):
                try:
                    full_df = xls.parse(sheet_name, header=0).fillna("")
                    rows_total, cols_total = full_df.shape
                    # header-aware rows_per
                    rows_per = _guess_rows_per(full_df) or max(1, rows_total - 1)

                    def _chunks(df):
                        for r0 in range(0, max(rows_total - 1, 1), rows_per):
                            r1 = min(r0 + rows_per, rows_total - 1)
                            body = df.iloc[r0 + 1: r1 + 1] if rows_total > 1 else pd.DataFrame()
                            sub = pd.concat([df.iloc[[0]], body]) if rows_total > 0 else body
                            yield r0, r1, 0, max(cols_total - 1, 0), sub

                    for r0, r1, c0, c1, sub in _chunks(full_df):
                        txt = _df_to_text(sub, fmt=cfg.table_format)
                        meta = {
                            **file_meta, **wb_props,
                            "type": "excel_sheet_chunk",
                            "filepath": str(path),
                            "filename": path.name,
                            "sheet": sheet_name,
                            "sheet_index": s_idx,
                            "sheet_rows_total": rows_total,
                            "sheet_cols_total": cols_total,
                            "tbl_format": cfg.table_format,
                            "row_start": int(r0),
                            "row_end": int(r1),
                            "col_start": int(c0),
                            "col_end": int(c1),
                            "nrows_in_chunk": int((r1 - r0) + 1 if rows_total > 0 else 0),
                            "ncols_in_chunk": int((c1 - c0) + 1 if cols_total > 0 else 0),
                        }
                        raw_id = f"{path}|{sheet_name}|r{r0}-{r1}|c{c0}-{c1}"
                        meta["chunk_id"] = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                        sheet_chunks.append(Document(page_content=txt or " ", metadata=meta))
                except Exception as e:
                    bad_sheets.append(sheet_name)

            try:
                for cid, sid, sheet, series in _iter_chart_series(str(path)):
                    md = series.to_frame(name="value").to_markdown(index=False)
                    meta = {
                        **file_meta, **wb_props,
                        "type": "excel_chart_table",
                        "filepath": str(path),
                        "filename": path.name,
                        "sheet": sheet,
                        "chart": f"{cid}.{sid}",
                    }
                    raw_id = f"{path}|chart|{cid}.{sid}|{sheet}"
                    meta["chunk_id"] = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                    chart_tables.append(Document(page_content=f"Chart {cid}.{sid} ({sheet})\n\n{md}", metadata=meta))
            except Exception:
                pass

            chunks = sheet_chunks + chart_tables
            if chunks:
                self._add_documents_with_retry(_vector_store(collection), chunks, on_status=on_status)
                summary, keywords = _summarise_file_llm(path)
                entry.update(
                    type="excel",
                    **wb_props,                         # workbook title/creator/created/modified/keywords
                    num_sheets=len(xls.sheet_names),
                    total_sheet_chunks=len(sheet_chunks),
                    charts=len(chart_tables),
                    bad_sheets=bad_sheets,
                    total_chunks=len(chunks),
                    chunk_size_chars=None,              # not applicable; table chunking is row-based
                    chunk_overlap_chars=None,
                )
                self._write_filemeta_doc(collection, entry, summary=summary, keywords=keywords)
            else:
                entry.update(type="excel", skipped=True, reason="no_sheets_or_charts")
                self._write_filemeta_doc(collection, entry, summary="empty excel", keywords=["excel","empty"])


        # TXT / HTML / XML
        elif ext in (".txt", ".html", ".htm", ".xml"):
            extraction = "plain_text" if ext == ".txt" else ("html_text" if ext in (".html", ".htm") else "xml_text")
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                if ext == ".txt":
                    text, title = raw, None
                elif ext in (".html", ".htm"):
                    soup = BeautifulSoup(raw, "html.parser")
                    for tag in soup(["script", "style"]): tag.extract()
                    text = " ".join(s.strip() for s in soup.stripped_strings)
                    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
                else:
                    soup = BeautifulSoup(raw, "xml")
                    text, title = " ".join(s.strip() for s in soup.stripped_strings), None
            except Exception as e:
                entry.update(skipped=True, reason=f"open-failed: {e}", **file_meta)
                return

            chunk_size, chunk_overlap = _split_params([len(text)])
            step = max(1, chunk_size - chunk_overlap)
            starts = list(range(0, len(text), step)) or [0]

            docs_to_add: List[Document] = []
            for j, start in enumerate(starts):
                end = min(start + chunk_size, len(text))
                chunk_txt = text[start:end]
                meta = {
                    **file_meta,
                    "type": f"{extraction}_chunk",
                    "filepath": str(path),
                    "filename": path.name,
                    "doc_title": title,
                    "page_number": 1,
                    "page_label": None,
                    "page_char_len": len(text),
                    "chunk_idx_in_page": j,
                    "total_chunks_on_page": len(starts),
                    "chunk_char_start": start,
                    "chunk_char_end": end,
                    "chunk_size_chars": chunk_size,
                    "chunk_overlap_chars": chunk_overlap,
                    "split_policy": "dynamic_60p_to_half_avg",
                }
                raw_id = f"{path}|p1|{start}-{end}"
                meta["chunk_id"] = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                docs_to_add.append(Document(page_content=chunk_txt or " ", metadata=meta))
            # after you build docs_to_add
            self._add_documents_with_retry(_vector_store(collection), docs_to_add, on_status=on_status)
            summary, keywords = _summarise_file_llm(path)
            entry.update(
                type=("text" if ext==".txt" else "html" if ext in (".html",".htm") else "xml"),
                doc_title=title,
                num_pages=1,
                total_chunks=len(docs_to_add),
                chunk_size_chars=chunk_size,
                chunk_overlap_chars=chunk_overlap,
            )
            self._write_filemeta_doc(collection, entry, summary=summary, keywords=keywords)

        else:
            entry.update(skipped=True, reason="conversion_failed")   # or other reason
            self._write_filemeta_doc(collection, entry, summary="skipped", keywords=[entry["reason"]])
            return

        entry["errors"] = errors
        entry["error_count"] = len(errors)
    # helpers
    def _write_filemeta_doc(
    self,
    collection: str,
    entry: Dict[str, Any],
    summary: str,
    keywords: List[str],
    ) -> None:
        """
        Store a single vectorized row in {collection}_filemeta with a rich metadata.* map.
        We upsert by metadata.filepath for idempotency.
        """
        # text used for embedding
        text = (summary or "") + (" | " if summary else "") + ", ".join(keywords or [])

        # Enforce required keys
        entry = {**entry}
        entry.setdefault("filepath", entry.get("file_path"))  # fallback
        entry.setdefault("filename", os.path.basename(entry.get("filepath", "") or ""))
        entry.setdefault("metadata_version", 2)
        entry.setdefault("summary", summary or "")
        entry.setdefault("keywords", keywords or [])
        entry.setdefault("ingest_ts_iso", datetime.now(timezone.utc).isoformat())
        entry.setdefault("semantic_sha256", entry.get("semantic_sha256"))

        # Remove any accidental top-levels (defense-in-depth)
        doc = Document(page_content=text, metadata=entry)

        coll = self._filemeta_coll(collection)
        # Hard upsert: delete and reinsert through vector store (so embedding is updated)
        coll.delete_many({"metadata.filepath": entry.get("filepath")})
        _vector_store(collection + "_filemeta").add_documents([doc])


    def _add_documents_with_retry(
        self,
        vectordb: MongoDBAtlasVectorSearch,
        docs: List[Document],
        *,
        max_retries: int = 8,
        base_delay: float = 2.5,
        max_delay: float = 60.0,
        on_status: ProgressCb = None,
    ) -> None:
        delay = base_delay
        for attempt in range(1, max_retries + 1):
            try:
                vectordb.add_documents(docs)
                if on_status: on_status({"status": f"batch ok: {len(docs)} docs"})
                return
            except Exception as e:
                s = str(e)
                transient = ("429" in s or "RateLimit" in s or "Too Many Requests" in s
                             or "timeout" in s.lower() or "ServiceUnavailable" in s)
                if not transient or attempt == max_retries:
                    raise
                sleep_s = min(delay, max_delay)
                if on_status: on_status({"status": f"transient ({attempt}/{max_retries}) ? sleep {sleep_s:.1f}s"})
                time.sleep(sleep_s)
                delay = min(delay * 1.7, max_delay)

# row sizing for Excel chunks (header-aware)
def _guess_rows_per(df: pd.DataFrame, sample_rows: int = 5, model_name: str = DEFAULT_GPT_DEPLOYMENT,
                    max_tokens: int = 2000) -> Optional[int]:
    if df.shape[0] <= 1:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    header_text = "\t".join(map(str, df.columns))
    header_tokens = len(enc.encode(header_text))
    n_samples = min(sample_rows, df.shape[0] - 1)
    token_costs = []
    for i in range(n_samples):
        row_text = "\t".join(map(str, df.iloc[i].astype(str).tolist()))
        token_costs.append(len(enc.encode(row_text)))
    if not token_costs:
        return None
    avg_row_tokens = max(int(np.mean(token_costs)), 1)
    budget = max_tokens - header_tokens
    if budget <= 0:
        return None
    rows_per = int(budget // avg_row_tokens)
    if rows_per < 1:
        return None
    return min(rows_per, df.shape[0] - 1)

# default progress callbacks (used by your app)
def default_status_cb(d: Dict[str, Any]):
    try: st.write(d.get("status", ""))
    except Exception: pass

def default_file_cb(d: Dict[str, Any]):
    try: st.write(f"Indexing {d['file']} ({d['idx']}/{d['total']})")
    except Exception: pass

def default_page_cb(d: Dict[str, Any]):
    try: st.write(f"  ? Page {d['page']}/{d['total_pages']} (OCR={d['ocr']})")
    except Exception: pass
