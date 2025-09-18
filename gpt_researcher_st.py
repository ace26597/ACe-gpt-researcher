from __future__ import annotations
import os, io, asyncio, logging, textwrap
from typing import List, Dict, Any, Tuple, Optional
from contextlib import asynccontextmanager
import streamlit as st
from copy import deepcopy

from gpt_researcher import GPTResearcher

from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch 

from langchain_openai import AzureOpenAIEmbeddings
import tempfile, markdown2, pdfkit, pypandoc, io
from streamlit.runtime.caching import cache_resource   # st.cache_resource wrapper
from docx import Document
import time
from collections import deque
import json, re, threading
from streamlit_extras.capture import stdout, stderr, logcapture
import markdown2, pdfkit, tempfile, datetime
from reasearchAI_utils import md_to_docx, md_to_pdf
import zipfile, shutil
from pathlib import Path
# NEW: direct Chroma access (no metadata.json)

# NEW imports for the News tab
import requests, datetime as dt
from urllib.parse import urlencode, quote_plus
import pandas as _pd

# Optional LLM for summaries (falls back gracefully if not configured)
try:
    from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI
except Exception:
    _AzureChatOpenAI = None
# NEW (top-level)
import requests, datetime as dt
import pandas as _pd
import re

# 🧩 our indexer
from mongo_indexer import MongoIndexer, IngestConfig, default_status_cb, default_file_cb, default_page_cb, get_embedding_cached

from news_utils import (
    run_news_from_file_graph,
    normalize_columns as news_normalize_columns,   # optional if you want to validate early
)

# ──────────────────────────── CONFIG ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
from gpt_researcher.actions import agent_creator as AG
indexer = MongoIndexer()

BASE_DIR            = "/root/Downloads/Jupyterlab/Ankur_Notebooks/RAG/DEV_ragflow/"
BASE_DATA_DIR       = "/root/Downloads/Jupyterlab/Ankur_Notebooks/RAG/DEV_ragflow/collections_mongo"
DB_NAME_RAG        = "RAG"
DB_NAME_REGGINIE   = "RegGinie"

ATLAS_CONNECTION_STR_RAG      = os.getenv("ATLAS_CONNECTION_STR")              # existing
ATLAS_CONNECTION_STR_REGGINIE = os.getenv("ATLAS_CONNECTION_STR_REG")     # <-- add this in your .env

ATLAS_VECTOR_INDEX  = "vector_index_demo"  # unchanged
ATLAS_VECTOR_INDEX_REGGINIE   = "openai_vector_index"  # unchanged

MCP_ENV_VAR = "MCP_SERVERS"

# Static MCP registry. These definitions are serialized and pushed into the
# environment before each GPT Researcher run so we do not depend on external
# configuration files. Update API keys / connection strings via the standard
# environment variables referenced below.
MCP_STATIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vector_store_rag": {
        "label": "🗂 (MongoDB) My Documents (RAG)",
        "kind": "vector_store",
        "options": {
            "database": DB_NAME_RAG,
            "connection_env": "ATLAS_CONNECTION_STR",
        },
    },
    "vector_store_reg": {
        "label": "🗂 (MongoDB) RegGinie collections",
        "kind": "vector_store",
        "options": {
            "database": DB_NAME_REGGINIE,
            "connection_env": "ATLAS_CONNECTION_STR_REG",
        },
    },
    "tavily": {
        "label": "🔎 Tavily",
        "kind": "web",
        "server": {
            "command": "python",
            "args": ["-m", "gpt_researcher.mcp_servers.tavily"],
            "env": {
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
            },
        },
    },
    "arxiv": {
        "label": "📄 arXiv",
        "kind": "web",
        "server": {
            "command": "python",
            "args": ["-m", "gpt_researcher.mcp_servers.arxiv"],
            "env": {},
        },
    },
    "pubmed_central": {
        "label": "🧪 PubMed Central",
        "kind": "web",
        "server": {
            "command": "python",
            "args": ["-m", "gpt_researcher.mcp_servers.pubmed_central"],
            "env": {
                "NCBI_API_KEY": os.getenv("NCBI_API_KEY", ""),
            },
        },
    },
}

# Normalized identifiers for MCP entries that should be treated as Mongo vector stores.
_VECTOR_MCP_KEYS = {"vector_store_rag", "mongo_rag", "vector_store_reg", "mongo_reg"}

# Normalized identifiers for MCP entries that should receive date-filter hooks.
_WEB_FILTER_AWARE_MCPS = {"tavily", "pubmed", "pubmed_central", "arxiv"}


def _normalize_mcp_key(name: str) -> str:
    """Normalize MCP keys for consistent comparisons."""
    if not name:
        return ""
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized.endswith("_mcp"):
        normalized = normalized[:-4]
    return normalized


def _load_mcp_registry() -> Dict[str, Any]:
    """Return a copy of the static MCP registry."""
    return deepcopy(MCP_STATIC_REGISTRY)


def _format_mcp_label(key: str, meta: Dict[str, Any]) -> str:
    """Return a human-friendly label for an MCP entry."""
    for candidate in ("label", "name", "title"):
        if isinstance(meta, dict) and isinstance(meta.get(candidate), str):
            return meta[candidate]
    return key

# ---------- filename helper ----------
import unicodedata
import time

def _slug_filename_part(text: str, maxlen: int = 80) -> str:
    """
    Slugify a snippet for filenames: ASCII, lowercase, hyphens/underscores only.
    """
    if not text:
        return ""
    # Normalize, strip accents
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alnum with single hyphens
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    # Collapse consecutive hyphens and trim
    text = re.sub(r"-{2,}", "-")
    return text[:maxlen].strip("-")

def _build_report_filename(question: str, report_type: str, custom_prompt: str, ext: str) -> str:
    """
    Compose a descriptive filename that includes the key args.
    Example: report__20250905__detailed-report__growth-prospects-llms__roi-500-words.pdf
    """
    date_str = time.strftime("%Y%m%d")
    parts = [
        "report",
        date_str,
        _slug_filename_part(report_type),
        _slug_filename_part(question),
    ]
    cp = _slug_filename_part(custom_prompt)
    if cp:
        parts.append(cp)
    stem = "__".join([p for p in parts if p])
    return f"{stem}.{ext.lstrip('.')}"

# ---------- logger wiring (unchanged) ----------
_LOG_NAMES = ["", "gpt_researcher","gpt_researcher.agent","gpt_researcher.actions",
              "gpt_researcher.retrievers","gpt_researcher.utils","gpt_researcher.llm_provider"]

def _hook_research_logs(handler: logging.Handler):
    for name in _LOG_NAMES:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.propagate = True

def _unhook_research_logs(handler: logging.Handler):
    for name in _LOG_NAMES:
        lg = logging.getLogger(name)
        try: lg.removeHandler(handler)
        except Exception: pass

class MongoPreFilterAdapter(MongoDBAtlasVectorSearch):
    # normalize GPT-Researcher's `filter` -> Atlas's `pre_filter`
    def similarity_search(self, query: str, k: int = 4, **kwargs):
        if "filter" in kwargs and "pre_filter" not in kwargs:
            kwargs["pre_filter"] = kwargs.pop("filter")
        return super().similarity_search(query=query, k=k, **kwargs)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs):
        if "filter" in kwargs and "pre_filter" not in kwargs:
            kwargs["pre_filter"] = kwargs.pop("filter")
        return await super().asimilarity_search(query=query, k=k, **kwargs)

def extract_json_with_regex(response: Any) -> str:
    text = _coerce_to_text(response)
    if not text: return "{}"
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    return m.group(0) if m else "{}"

async def handle_json_error(response: Any) -> Tuple[str, str]:
    raw = extract_json_with_regex(response)
    try:
        data = json.loads(raw)
        agent = str(data.get("agent", "")).strip()
        role  = str(data.get("role",  "")).strip()
        if agent and role: return agent, role
    except Exception:
        pass
    logging.warning("choose_agent: empty/invalid LLM response; using defaults.")
    return "researcher", "generalist"

# wire into gpt_researcher helper
AG.extract_json_with_regex = extract_json_with_regex
AG.handle_json_error = handle_json_error

# ─────────────────────────── gpt_researcher run ───────────────────────────────
async def _async_run(query: str, cfg: Dict[str, Any]):
    # Read config and keep the raw MCP selection (don’t mutate yet)
    selected_mcps = list(cfg.pop("selected_mcps") or [])
    urls          = cfg.pop("urls")
    mcp_servers_json = cfg.pop("mcp_servers_json", None)
    retriever_mode   = cfg.pop("retriever_mode", None)
    mcp_selection_env = cfg.pop("mcp_selection_env", None)

    env_prev: Dict[str, Optional[str]] = {}
    result: Tuple[str, Any, Any, Any, Any] = ("", None, None, None, None)

    try:
        if retriever_mode is not None:
            env_prev["RETRIEVER"] = os.environ.get("RETRIEVER")
            os.environ["RETRIEVER"] = retriever_mode
        if mcp_servers_json is not None:
            env_prev[MCP_ENV_VAR] = os.environ.get(MCP_ENV_VAR)
            os.environ[MCP_ENV_VAR] = mcp_servers_json
        if mcp_selection_env is not None:
            env_prev["MCP_SELECTED"] = os.environ.get("MCP_SELECTED")
            os.environ["MCP_SELECTED"] = mcp_selection_env

        normalized_pairs = [(mcp, _normalize_mcp_key(mcp)) for mcp in selected_mcps]
        has_rag = any(norm in {"vector_store_rag", "mongo_rag"} for _, norm in normalized_pairs)
        has_reg = any(norm in {"vector_store_reg", "mongo_reg"} for _, norm in normalized_pairs)
        web_mcps = [orig for orig, norm in normalized_pairs if norm not in _VECTOR_MCP_KEYS]
        web_mcps_norm = [norm for _, norm in normalized_pairs if norm not in _VECTOR_MCP_KEYS]
        has_vector = has_rag or has_reg
        exactly_one_vector = int(has_rag) + int(has_reg) == 1

        vector_store = None
        report_source = "web"

        # Choose vector store (only if exactly one vector backend is selected)
        if exactly_one_vector:
            coll = cfg.get("collection")
            if has_rag:
                vector_store = get_mongo_vector_store_cached(DB_NAME_RAG, ATLAS_CONNECTION_STR_RAG, coll, has_rag)
            else:
                vector_store = get_mongo_vector_store_cached(DB_NAME_REGGINIE, ATLAS_CONNECTION_STR_REGGINIE, coll, has_rag)
            logging.info("VectorStore (Mongo): %s @ %s", type(vector_store).__name__, "RAG" if has_rag else "RegGinie")

        # Decide report_source
        if exactly_one_vector and not web_mcps:
            report_source = "langchain_vectorstore"
        elif has_vector and web_mcps:
            report_source = "hybrid"
        elif web_mcps:
            report_source = "web"
        else:
            # nothing selected or both vectors selected without web → treat as web (no retrievers)
            report_source = "web"

        date_filters = extract_date_window_from_query(query)
        st.sidebar.write(date_filters)
        if date_filters and any(norm in _WEB_FILTER_AWARE_MCPS for norm in web_mcps_norm):
            install_date_filter_hooks(date_filters)

        st.sidebar.write(report_source, selected_mcps)

        if report_source == 'langchain_vectorstore':
            researcher = GPTResearcher(
                query         = query,
                report_type   = cfg["report_type"],
                tone          = cfg["tone"],
                verbose       = cfg["verbose"],
                report_source = report_source,     # "langchain_vectorstore" | "hybrid" | "web"
                vector_store  = vector_store,      # Mongo instance, or None
            )
        else:
            researcher = GPTResearcher(
                query         = query,
                report_type   = cfg["report_type"],
                tone          = cfg["tone"],
                report_format = cfg["report_format"],
                verbose       = cfg["verbose"],
                report_source = report_source,     # "langchain_vectorstore" | "hybrid" | "web"
                vector_store  = vector_store,      # Mongo instance, or None
                source_urls   = urls or None,
                complement_source_urls = cfg["complement"],
            )

        # optional extras…
        USER_EXTRAS = [
            "breadth", "depth", "max_subtopics",
            "draft_section_titles", "subtopic_name", "custom_prompt", "on_progress"
        ]
        for key in USER_EXTRAS:
            if key in cfg:
                setattr(researcher, key, cfg[key])

        ui = cfg.pop("ui", {}) if isinstance(cfg.get("ui"), dict) else {}
        trace_box = ui.get("trace_box", st.empty())
        answer_container = ui.get("answer_box", st.empty())

        # logging + streaming unchanged…
        fmt = logging.Formatter("INFO:     [%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        handler = TraceBufferHandler(max_lines=2000)
        handler.setFormatter(fmt)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        _hook_research_logs(handler)
        pumper = asyncio.create_task(_pump_trace_ui(trace_box, handler))
        st.write(researcher)
        try:
            def _emit_info(line: str):
                if not line: return
                rec = logging.LogRecord(
                    name="gpt_researcher.stdout", level=logging.INFO,
                    pathname=__file__, lineno=0, msg=line.rstrip(), args=(), exc_info=None
                )
                handler.handle(rec)

            with stdout(_emit_info, terminator=""):
                with stderr(_emit_info, terminator=""):
                    await researcher.conduct_research()
        finally:
            handler.stop()
            await pumper
            _unhook_research_logs(handler)

        md = ""
        try:
            stream_fn = getattr(researcher, "stream_report", None)
            if callable(stream_fn):
                chunks: list[str] = []
                async def _gen():
                    async for delta in stream_fn():
                        s = delta if isinstance(delta, str) else str(delta)
                        chunks.append(s)
                        yield s
                await _progressive_markdown_stream(answer_container, _gen())
                final_fn = getattr(researcher, "get_final_report", None)
                md = final_fn() if callable(final_fn) else "".join(chunks)
                if not isinstance(md, str) or not md:
                    md = "".join(chunks)
            else:
                md = await researcher.write_report()
                await _fake_stream_from_final(md, answer_container)
        except Exception as e:
            logging.exception("write/stream report failed: %s", e)
            try:
                md = await researcher.write_report()
                answer_container.markdown(md)
            except Exception as e2:
                st.error(f"Failed to generate report: {e2}")
                md = ""

        result = (
            md if md is not None else "",
            researcher.get_costs(),
            researcher.get_research_images(),
            researcher.get_source_urls(),
            researcher.get_research_sources()
        )

    finally:
        for key, previous in env_prev.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    return result

def _coerce_to_text(resp: Any) -> str:
    if resp is None: return ""
    if isinstance(resp, str): return resp
    try:
        if isinstance(resp, dict):
            ch = resp.get("choices")
            if ch and isinstance(ch, list):
                msg = ch[0].get("message") if isinstance(ch[0], dict) else None
                if msg and isinstance(msg, dict) and "content" in msg:
                    return msg["content"] or ""
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)

def run_sync(query: str, cfg: Dict[str, Any]):
    return asyncio.run(_async_run(query, cfg))

# ---------- TraceBufferHandler + streaming helpers (unchanged) ----------
class TraceBufferHandler(logging.Handler):
    def __init__(self, max_lines=10000):
        super().__init__()
        # keep a large rolling buffer so the download has (almost) everything
        self._buf = deque(maxlen=max_lines)
        self._dirty = False
        self._lock = threading.Lock()
        self.running = True

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        with self._lock:
            self._buf.append(msg)
            self._dirty = True

    def snapshot(self) -> str:
        """Return the full trace (joined)."""
        with self._lock:
            self._dirty = False
            return "\n".join(self._buf)

    def tail(self, n: int = 50) -> str:
        """Return just the last N lines for on-screen display."""
        with self._lock:
            if not self._buf:
                return ""
            # fast slice from the right
            last = list(self._buf)[-n:]
            return "\n".join(last)

    def has_updates(self) -> bool:
        with self._lock:
            return self._dirty

    def stop(self):
        self.running = False


async def _pump_trace_ui(placeholder, handler: TraceBufferHandler, interval=0.12):
    last_tail = ""
    # clear any prior run’s saved trace
    st.session_state["trace_full_md"] = ""
    while handler.running:
        if handler.has_updates():
            # full text for download
            full_text = handler.snapshot()
            st.session_state["trace_full_md"] = f"```text\n{full_text}\n```"

            # only render tail(50) to the UI
            tail_text = handler.tail(50)
            if tail_text != last_tail:
                placeholder.code(tail_text)
                last_tail = tail_text
        await asyncio.sleep(interval)

    # final flush
    full_text = handler.snapshot()
    st.session_state["trace_full_md"] = f"```text\n{full_text}\n```"
    tail_text = handler.tail(50)
    if tail_text and tail_text != last_tail:
        placeholder.code(tail_text)


async def _progressive_markdown_stream(placeholder, token_iter, chunk_chars=600):
    buf, total = [], []
    async for piece in token_iter:
        buf.append(piece); total.append(piece)
        if sum(len(x) for x in buf) >= chunk_chars:
            placeholder.markdown("".join(total)); buf.clear()
    if buf or total:
        placeholder.markdown("".join(total))

async def _fake_stream_from_final(md_text: str, placeholder, chunk_chars=800):
    acc, cur = [], []
    for para in md_text.split("\n\n"):
        cur.append(para)
        if sum(len(x) for x in cur) >= chunk_chars:
            acc.append("\n\n".join(cur) + "\n\n")
            placeholder.markdown("".join(acc))
            cur.clear()
            await asyncio.sleep(0)
    if cur: acc.append("\n\n".join(cur))
    placeholder.markdown("".join(acc))
# ──────────────────────────── DIRECTORY INGEST HELPERS ───────────────────────
# ──────────────────────────── DIRECTORY INGEST HELPERS ───────────────────────
ALLOWED_FILE_EXTS = {
    ".pdf",".doc",".docx",".ppt",".pptx",".xls",".xlsx",".rtf",
    ".zip",".html",".htm",".xml",".txt"
}


# ---------------------------- DATE FILTER HOOKS -----------------------------
import re, datetime as _dt

_DATE_RX = re.compile(
    r"""(?ix)
    (?:
        (?:after|since)\s+(?P<after>\d{4}-\d{2}-\d{2}|\b\w+\s+\d{4}\b)
      | (?:from)\s+(?P<from>\d{4}-\d{2}-\d{2}|\b\w+\s+\d{4}\b)\s+(?:to|-|until)\s+(?P<to>\d{4}-\d{2}-\d{2}|\b\w+\s+\d{4}\b)
      | (?:last)\s+(?P<num>\d+)\s+(?P<Unit>day|days|week|weeks|month|months|year|years)
      | (?:newer\s+than)\s+(?P<num2>\d+)\s+(?P<Unit2>day|days|week|weeks|month|months|year|years)
      | (?:published|updated)\s+(?:after|since)\s+(?P<after2>\d{4}-\d{2}-\d{2})
    )
    """
)

def _to_yyyy_mm_dd(s: str) -> Optional[str]:
    s = s.strip()
    # already ISO?
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # try "Month YYYY"
    try:
        dt = _dt.datetime.strptime(s, "%B %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # try flexible parse if available
    try:
        import dateutil.parser as dp
        return dp.parse(s, default=_dt.datetime(1900,1,1)).date().isoformat()
    except Exception:
        return None

def _days_ago(n: int) -> str:
    return (_dt.date.today() - _dt.timedelta(days=n)).isoformat()

def extract_date_window_from_query(q: str) -> dict:
    """
    Returns a dict with any of: start_date, end_date, days, time_range
    Prefer start/end when explicit; use days for "last N" queries.
    """
    m = _DATE_RX.search(q or "")
    if not m:
        return {}
    d = m.groupdict()
    # explicit from-to
    if d.get("from"):
        start = _to_yyyy_mm_dd(d["from"])
        end   = _to_yyyy_mm_dd(d.get("to") or "")
        out = {}
        if start: out["start_date"] = start
        if end:   out["end_date"]   = end
        return out
    # after/since
    after = d.get("after") or d.get("after2")
    if after:
        start = _to_yyyy_mm_dd(after)
        return {"start_date": start} if start else {}
    # last N units
    n   = d.get("num") or d.get("num2")
    unit= (d.get("Unit") or d.get("Unit2") or "").lower()
    if n and unit:
        n = int(n)
        # Map unit -> days (for PubMed reldate & Tavily days)
        factor = 1 if "day" in unit else 7 if "week" in unit else 30 if "month" in unit else 365
        return {"days": n * factor}
    return {}

# ---------------------------- RETRIEVER PATCHES -----------------------------
def install_date_filter_hooks(date_filters: dict):
    """
    Patch Tavily / PubMedCentral / arXiv retrievers at runtime to honor date_filters.
    date_filters: may include start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), days (int).
    """
    if not date_filters:
        return

    # 1) Tavily: wrap TavilyClient.search to inject date args
    try:
        import tavily
        from functools import wraps
        _orig_search = tavily.TavilyClient.search

        @wraps(_orig_search)
        def _wrapped_search(self, query: str, **kwargs):
            # Prefer explicit range; else days; else time_range heuristic
            if date_filters.get("start_date"): kwargs.setdefault("start_date", date_filters["start_date"])
            if date_filters.get("end_date"):   kwargs.setdefault("end_date",   date_filters["end_date"])
            if date_filters.get("days"):       kwargs.setdefault("days",       date_filters["days"])
            # Optional: if user said "last week/month/year" map to time_range instead of days
            if not any(k in kwargs for k in ("start_date","end_date","days")) and date_filters.get("time_range"):
                kwargs.setdefault("time_range", date_filters["time_range"])
            return _orig_search(self, query=query, **kwargs)

        tavily.TavilyClient.search = _wrapped_search  # monkey-patch
    except Exception:
        pass

    # 2) PubMed Central: try to patch retriever if available; otherwise post-filter in scraper
    # GPT-Researcher retriever name: PubMedCentralSearch; it usually constructs ESearch URL.
    try:
        from gpt_researcher.retrievers.pubmed_central import pubmed_central as _pmc_mod
        if hasattr(_pmc_mod, "PubMedCentralSearch"):
            _PM = _pmc_mod.PubMedCentralSearch
            if hasattr(_PM, "search"):
                _orig_pmc_search = _PM.search
                def _pmc_search(self, max_results=8):
                    # Attach date params onto the instance if supported, else pass through kwargs
                    # The retriever typically builds query params dict; we provide hints:
                    setattr(self, "_mindate", date_filters.get("start_date"))
                    setattr(self, "_maxdate", date_filters.get("end_date"))
                    setattr(self, "_reldate", date_filters.get("days"))
                    return _orig_pmc_search(self, max_results=max_results)
                _PM.search = _pmc_search
    except Exception:
        pass

    # 3) arXiv: filter by date on results (library doesnt support hard range in all modes)
    try:
        import arxiv, functools
        _orig_results = arxiv.Client.results

        @functools.wraps(_orig_results)
        def _wrapped_results(self, search: "arxiv.Search"):
            for r in _orig_results(self, search):
                pub = getattr(r, "updated", None) or getattr(r, "published", None)
                try:
                    d = pub.date() if hasattr(pub, "date") else pub
                except Exception:
                    d = None
                ok = True
                if d and date_filters.get("start_date"):
                    ok = ok and (d >= _dt.date.fromisoformat(date_filters["start_date"]))
                if d and date_filters.get("end_date"):
                    ok = ok and (d <= _dt.date.fromisoformat(date_filters["end_date"]))
                if date_filters.get("days") and d:
                    ok = ok and (d >= _dt.date.fromisoformat(_days_ago(date_filters["days"])))
                if ok:
                    yield r
        arxiv.Client.results = _wrapped_results
    except Exception:
        pass

# ----------------------- FILE-DRIVEN NEWS HELPERS -----------------------

def _normalize_columns(cols: list[str]) -> dict[str, str]:
    """
    Return a mapping from normalized -> original column name.
    Handles common typos/variants: email_aler vs email_alert, etc.
    """
    norm = {}
    for c in cols:
        cl = c.strip().lower().replace(" ", "_")
        norm[cl] = c
    # gentle aliases
    alias_map = {
        "email_alert": ["email_alert", "email_aler", "alert", "email"],
        "search_type": ["search_type", "type"],
        "subheader":   ["subheader", "section", "group"],
        "user":        ["user", "owner"],
        "keyword":     ["keyword", "key_word", "term"],
        "aliases":     ["aliases", "alias", "synonyms"]
    }
    out = {}
    for want, cands in alias_map.items():
        for cand in cands:
            if cand in norm:
                out[want] = norm[cand]
                break
    return out  # e.g., {"email_alert": "email_aler", "keyword": "keyword", ...}

from functools import lru_cache

def _require_env(name: str, value: str | None):
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is required but missing.")
    return value

@cache_resource(show_spinner=False)
def get_mongo_vector_store_cached(
    db_name: str,
    conn_str: str,
    collection: str,
    has_rag: bool
) -> MongoDBAtlasVectorSearch:
    _require_env("ATLAS_CONNECTION_STR_*", conn_str)
    dims = 3072  # text-embedding-3-large
    client = MongoClient(conn_str)
    if has_rag: index_name= ATLAS_VECTOR_INDEX
    else :index_name= ATLAS_VECTOR_INDEX_REGGINIE
    return MongoPreFilterAdapter(
        collection         = client[db_name][collection],
        embedding          = get_embedding_cached(),
        index_name         = index_name,
        relevance_score_fn = "cosine",
        dimensions         = dims,
        auto_create_index  = True,
        auto_index_timeout = 600,
    )

@lru_cache(maxsize=8)
def list_mongo_collections(conn_str: str, db_name: str) -> list[str]:
    _require_env("ATLAS_CONNECTION_STR_*", conn_str)
    client = MongoClient(conn_str)
    names = client[db_name].list_collection_names()
    # Only show primary text collections (hide *_filemeta and system)
    return sorted([n for n in names if not n.startswith("system.") and not n.endswith("_filemeta")])

def list_filemeta_preview(conn_str: str, db_name: str, coll: str, limit: int = 10) -> list[dict]:
    """
    Read-only peek into <collection>_filemeta. Tries to show newest first.
    """
    _require_env("ATLAS_CONNECTION_STR_*", conn_str)
    client = MongoClient(conn_str)
    meta_col = f"{coll}_filemeta"
    if meta_col not in client[db_name].list_collection_names():
        return []
    cur = client[db_name][meta_col].find({}, {"_id": 0}).sort(
        [("metadata.ingest_ts_iso", -1), ("ingest_ts_iso", -1)]
    ).limit(limit)
    return list(cur)

# ──────────────────────────── STREAMLIT APP ─────────────────────────────────
def app():
    st.set_page_config("SMPA Research AI", page_icon="🔬", layout="wide")
    st.markdown("""
    <style>
    div.stCode, pre, code { max-height: 360px; overflow:auto; }
    </style>
    """, unsafe_allow_html=True)

    if "last" not in st.session_state:
        st.session_state["last"] = {}
    verbose = True

    with st.sidebar:
        tone = st.selectbox("Tone", [
            "objective","formal","analytical","persuasive",
            "informative","explanatory","descriptive"])
        report_format = st.selectbox("Citation style",
            ["APA","MLA","CMS","Harvard","IEEE"])

        mcp_registry = _load_mcp_registry()
        mcp_options = list(mcp_registry.keys())
        default_mcps = st.session_state.get("_mcp_selected_defaults", [])
        default_mcps = [m for m in default_mcps if m in mcp_options]
        if not default_mcps and mcp_options:
            default_mcps = mcp_options[:1]

        mcps_selected = st.multiselect(
            "MCP source(s)",
            options=mcp_options,
            default=default_mcps,
            format_func=lambda key: _format_mcp_label(key, mcp_registry.get(key, {})),
            disabled=not bool(mcp_options),
            help="Select one or more connectors defined in MCP_STATIC_REGISTRY.",
        )
        st.session_state["_mcp_selected_defaults"] = mcps_selected

        normalized_selection = {_normalize_mcp_key(m): m for m in mcps_selected}
        use_rag = any(norm in {"vector_store_rag", "mongo_rag"} for norm in normalized_selection)
        use_reg = any(norm in {"vector_store_reg", "mongo_reg"} for norm in normalized_selection)

        # Only one Mongo backend at a time
        if use_rag and use_reg:
            st.warning("Please select only **one** MongoDB source (RAG or RegGinie).")

        collection = None
        if use_rag and not use_reg:
            choices = list_mongo_collections(_require_env("ATLAS_CONNECTION_STR", ATLAS_CONNECTION_STR_RAG), DB_NAME_RAG)
            collection = st.selectbox("Collection (RAG)", choices, disabled=(not choices))
        elif use_reg and not use_rag:
            choices = list_mongo_collections(_require_env("ATLAS_CONNECTION_STR_REGGINIE", ATLAS_CONNECTION_STR_REGGINIE), DB_NAME_REGGINIE)
            collection = st.selectbox("Collection (RegGinie)", choices, disabled=(not choices))
        else:
            collection = st.selectbox("Collection", [], disabled=True)

        if not mcp_options:
            st.info("No MCP connectors configured in MCP_STATIC_REGISTRY. Update the registry in `gpt_researcher_st.py` to enable selections.")


        report_type = st.selectbox(
            "Report type", [
                "research_report","detailed_report","resource_report",
                "outline_report","custom_report","subtopic_report","deep"
            ], index=0,
        )

        extras: Dict[str, Any] = {}
        if report_type == "deep":
            extras["breadth"] = st.slider("Breadth (branches per level)", 1, 10, 4)
            extras["depth"]   = st.slider("Depth (levels)",               1,  5, 2)
        if report_type == "detailed_report":
            extras["max_subtopics"] = st.slider("Max sub-topics", 3, 10, 5)
        if report_type == "outline_report":
            extras["draft_section_titles"] = st.text_area(
                "Draft section titles (one per line)",
                placeholder="Introduction\nMethods\nFindings\nConclusion",
                height=100,
            )
        if report_type == "subtopic_report":
            extras["subtopic_name"] = st.text_input(
                "Sub-topic name", placeholder="e.g. LLM fine-tuning"
            )
        prompt_label = "Custom prompt (required)" if report_type == "custom_report" else "Custom prompt (optional)"
        extras["custom_prompt"] = st.text_area(
            prompt_label,
            placeholder="e.g. Create an executive summary focused on ROI, ≤500 words",
            height=120 if report_type == "custom_report" else 80,
        )
        st.caption('Write a blog post in a conversational tone using the research. Include headings and a conclusion ')
        st.caption('Create a FAQ section based on the research with at least 5 questions and detailed answers')
        st.caption("Create a report for technical stakeholders, focusing on methodologies and implementation details")

    tab_research, tab_docs, tab_news = st.tabs(["🔬 Research", "📁 Documents", "📰 News & PubMed"])


    # -------- Research Tab --------
    with tab_research:
        hdr = st.container()
        with hdr:
            colA, colB = st.columns((3, 1))
            query = st.text_input(
                "Research question",
                placeholder="e.g. Growth prospects for small-LLM fine-tuning",
                label_visibility="visible"
            )
            with colA:
                url_line = st.text_input(
                    "Custom URLs (comma-separated)",
                    placeholder="pubmed.ncbi.nlm.nih.gov, fiercepharma.com"
                )
            with colB:
                complement = st.checkbox("Also search outside these URLs", value=False)

        run = st.button("Run", use_container_width=True, icon="🚀")

        trace_expander = st.expander("Agent trace (research phase only)", expanded=True)
        with trace_expander:
            # download button (disabled until the handler pumps at least once)
            dl_col, _ = st.columns([1, 3])
            with dl_col:
                file_name = f"agent_trace_{time.strftime('%Y%m%d_%H%M%S')}.md"
                full_md = st.session_state.get("trace_full_md", "")
                st.download_button(
                    "⬇️ Download full trace (MD)",
                    data=(full_md.encode("utf-8") if full_md else b""),
                    file_name=file_name,
                    mime="text/markdown",
                    use_container_width=True,
                    disabled=(not bool(full_md)),
                )
            # the live, tail(50) view:
            trace_box = st.empty()


        col1, col2 = st.columns((2, 1), gap="large")
        with col1:
            mdout = st.empty()
        with col2:
            pdf_download = st.empty()
            docx_download = st.empty()
            md_download = st.empty()
            cost_box = st.empty()
            img_box = st.container()
            src_box = st.container()

        def render(md, costs, imgs, urls, detailed, query, report_type, custom_prompt, already_streamed: bool = False):
            if not already_streamed:
                mdout.markdown(md, unsafe_allow_html=True)
            if costs:
                if isinstance(costs, dict):
                    cost_box.json(costs, expanded=False)
                else:
                    cost_box.code(f"Total Research Cost : {str(costs)}")
            if imgs:
                img_box.image(imgs, width=220)
            else:
                img_box.info("No images returned.")
            refs = urls or [s.get("url","") for s in detailed if s.get("url")]
            if refs:
                src_box.markdown("### References\n" + "\n".join(f"* <{u}>" for u in refs))
            else:
                src_box.info("No references available.")
        
            # Build descriptive filenames that include question, report_type, and custom_prompt
            pdf_name  = _build_report_filename(query, report_type, custom_prompt or "", "pdf")
            docx_name = _build_report_filename(query, report_type, custom_prompt or "", "docx")
            md_name   = _build_report_filename(query, report_type, custom_prompt or "", "md")
        
            pdf_bytes  = md_to_pdf(md)
            docx_bytes = md_to_docx(md)
        
            pdf_download.download_button(
                "⬇️ Download Report - PDF",
                pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True
            )
            docx_download.download_button(
                "⬇️ Download Report - DOCX",
                docx_bytes,
                file_name=docx_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            md_download.download_button(
                "⬇️ Download Report - Markdown",
                md.encode(),
                file_name=md_name,
                mime="text/markdown",
                use_container_width=True
            )


        if run:
            
            if not query.strip():
                st.warning("Please enter a research question."); st.stop()
            if report_type == "custom_report" and not extras["custom_prompt"].strip():
                st.warning("Custom prompt is required for *custom_report*."); st.stop()
            
            if not mcps_selected:
                st.warning("Select at least one MCP source from the sidebar before running."); st.stop()

            selected_mcp_config: Dict[str, Any] = {}
            for name in mcps_selected:
                if name not in mcp_registry:
                    continue
                entry = deepcopy(mcp_registry[name])
                norm = _normalize_mcp_key(name)
                if collection:
                    if norm in {"vector_store_rag", "mongo_rag"}:
                        entry.setdefault("options", {})["collection"] = collection
                    elif norm in {"vector_store_reg", "mongo_reg"}:
                        entry.setdefault("options", {})["collection"] = collection
                selected_mcp_config[name] = entry
            try:
                mcp_servers_json = json.dumps(selected_mcp_config)
            except TypeError as exc:
                st.error(f"Unable to serialize MCP configuration: {exc}"); st.stop()

            cfg = dict(
                selected_mcps=mcps_selected,
                mcp_servers_json=mcp_servers_json,
                mcp_selection_env=",".join(mcps_selected),
                retriever_mode="mcp",
                collection=collection,
                report_type=report_type,
                tone=tone,
                report_format=report_format,
                verbose=verbose,
                urls=[u.strip() for u in url_line.split(",") if u.strip()],
                complement=complement,
                **extras,
            )

            

            total_q = None
            if report_type == "deep":
                b = extras.get("breadth", 4); d = extras.get("depth", 2)
                total_q = b ** d

            prog_bar = st.progress(0.0, text="Starting …")

            def _on_progress(p):
                try:
                    done = p.completed_queries
                    total = p.total_queries or total_q or 1
                    pct   = min(1.0, done / total)
                    prog_bar.progress(pct, text=f"{int(pct*100)} % done")
                except Exception:
                    pass
            st.sidebar.write(cfg)
            md, costs, imgs, urls, src_det = run_sync(
                query,
                {**cfg, "on_progress": _on_progress, "ui": {"trace_box": trace_box, "answer_box": mdout}}
            )
            prog_bar.empty()

            st.session_state["last"] = dict(
                md=md, co=costs, im=imgs, ur=urls, sd=src_det,
                q=query, rt=report_type, cp=extras.get("custom_prompt","")
            )
            render(md, costs, imgs, urls, src_det, query, report_type, extras.get("custom_prompt",""), already_streamed=bool(md))


        elif st.session_state.get("last"):
            d = st.session_state["last"]
            render(
                d["md"], d["co"], d["im"], d["ur"], d.get("sd", []),
                d.get("q",""), d.get("rt",""), d.get("cp","")
            )

        else:
            mdout.info("Fill in the inputs above and click **Run** to start.")

    # -------- Documents Tab --------
    with tab_docs:
        if not (use_rag or use_reg):
            st.info("Select a MongoDB source in the sidebar to view collections.")
            st.stop()

        if use_rag and use_reg:
            st.warning("Please select only one MongoDB source (RAG or RegGinie).")
            st.stop()

        if collection is None:
            st.info("Choose a collection from the sidebar.")
            st.stop()

        is_rag = bool(use_rag)
        db_name = DB_NAME_RAG if is_rag else DB_NAME_REGGINIE
        conn    = ATLAS_CONNECTION_STR_RAG if is_rag else ATLAS_CONNECTION_STR_REGGINIE
        label   = "RAG" if is_rag else "RegGinie"

        st.markdown(f"### Collection (**{label}**): `{collection}` (read-only)")

        rows = list_filemeta_preview(conn, db_name, collection, limit=10)
        if not rows:
            st.info("No file metadata found (or `_filemeta` collection is missing).")
        else:
            st.caption("Showing up to 10 recent items from `<collection>_filemeta`.")
            for doc in rows:
                # Prefer nested metadata if present
                meta = (
                    doc.get("metadata")
                    or (doc.get("doc") or {}).get("metadata")
                    or (doc.get("document") or {}).get("metadata")
                    or {}
                )
                # Normalize
                path  = meta.get("filepath") or meta.get("file_path") or doc.get("filepath") or ""
                fname = meta.get("filename") or meta.get("file_name") or (os.path.basename(path) if path not in ("", "") else "")
                with st.expander(fname, expanded=False):
                    st.write(f"Path: `{path}`")
                    st.write(f"Type: {meta.get('type','')}  |  Size: {meta.get('file_size_bytes','')}  |  SHA256: {meta.get('file_sha256','')}")
                    if meta.get("type") == "pdf":
                        st.write(f"Pages: {meta.get('num_pages','')}  |  OCR pages: {meta.get('ocr_pages','')}  |  Chunks: {meta.get('total_chunks','')}")
                    if meta.get("type") == "excel":
                        st.write(f"Sheets: {meta.get('num_sheets','')}  |  Charts: {meta.get('charts','')}  |  Chunks: {meta.get('total_chunks','')}")
                    if meta.get("type") in ("text","html","xml"):
                        st.write(f"Chunks: {meta.get('total_chunks','')}  |  Chunk size: {meta.get('chunk_size_chars','')}  |  Overlap: {meta.get('chunk_overlap_chars','')}")
                    st.write(f"Ingested: {meta.get('ingest_ts_iso','')}  |  Source: {meta.get('ingest_source','')}")
                    # Optional local file download if exists on disk
                    if path and path not in ("",) and os.path.exists(path):
                        try:
                            with open(path, "rb") as fh:
                                st.download_button(
                                    "Download original", fh.read(),
                                    file_name=os.path.basename(path),
                                    mime="application/octet-stream",
                                    use_container_width=True,
                                )
                        except Exception:
                            st.caption("Original file present but unreadable.")
                    else:
                        st.caption("Original file not available on disk.")

        st.info("Note: Uploading / indexing / deleting is disabled in this build.")

        # -------- News & PubMed (File-Only) --------
    with tab_news:
        st.markdown("### Generate report from CSV/XLSX (file-only)")

        # Date range (end defaults to today)
        colD1, colD2 = st.columns([1,1])
        with colD1:
            start_date = st.date_input("Start date (required)", value=None, format="YYYY-MM-DD")
        with colD2:
            end_date = st.date_input("End date", value=dt.date.today(), format="YYYY-MM-DD")

        # Source selection + optional domain restriction for Tavily
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            use_tv = st.checkbox("Use Tavily (web/news)", value=True)
        with c2:
            use_pm = st.checkbox("Use PubMed", value=True)
        with c3:
            domains_str = st.text_input("Restrict Tavily to domains (optional, comma-separated)",
                                        placeholder="fiercepharma.com, endpointsnews.com, statnews.com")
        domain_list = [d.strip() for d in domains_str.split(",") if d.strip()]

        st.markdown("#### Upload file")
        up = st.file_uploader("CSV/XLSX columns (normalized): keyword, aliases, user, email_alert, search_type, subheader",
                              type=["csv","xlsx"], accept_multiple_files=False)

        sheet_name = None
        df_file = None
        if up is not None:
            try:
                if up.name.lower().endswith(".xlsx"):
                    x = _pd.ExcelFile(up)
                    sheet_name = st.selectbox("Sheet", x.sheet_names, index=0)
                    df_file = x.parse(sheet_name)
                else:
                    df_file = _pd.read_csv(up)
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        chosen_user = None
        colmap = None
        if df_file is not None and not df_file.empty:
            colmap = _normalize_columns(list(df_file.columns))
            req = ["keyword","user","email_alert","search_type","subheader"]
            missing = [r for r in req if r not in colmap]
            if missing:
                st.warning(f"Missing required columns (normalized): {', '.join(missing)}")
            else:
                users = sorted(df_file[colmap["user"]].dropna().astype(str).str.strip().unique().tolist())
                chosen_user = st.selectbox("Select user", users)

        run_btn = st.button("🚀 Generate From File", type="primary", use_container_width=True)

        out_md = st.empty()
        out_table = st.empty()
        dl1, dl2 = st.columns([1,1])

        if run_btn:
            if not start_date:
                st.warning("Please select a **Start date**."); st.stop()
            if end_date and start_date and end_date < start_date:
                st.warning("End date cannot be earlier than Start date."); st.stop()
            if df_file is None or df_file.empty or not chosen_user:
                st.warning("Please upload a valid file and choose a user."); st.stop()
            if not (use_tv or use_pm):
                st.warning("Select at least one source (Tavily or PubMed)."); st.stop()

            s_iso = str(start_date)
            e_iso = str(end_date) if end_date else None

            try:
                sections, df_out, md_report = run_news_from_file_graph(
                    df=df_file,
                    chosen_user=chosen_user,
                    start_date=s_iso,
                    end_date=e_iso,
                    use_tavily=use_tv,
                    use_pubmed=use_pm,
                    domains=[d.strip() for d in domains_str.split(",") if d.strip()],
                    max_results=25,  # you can expose as a slider later
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

            out_md.markdown(md_report)
            if not df_out.empty:
                out_table.dataframe(df_out, use_container_width=True)
                with dl1:
                    st.download_button("⬇️ Download Markdown Report",
                        md_report.encode("utf-8"), "news_pubmed_report.md", "text/markdown", use_container_width=True)
                with dl2:
                    st.download_button("⬇️ Download CSV",
                        df_out.to_csv(index=False).encode("utf-8"), "news_pubmed_results.csv", "text/csv", use_container_width=True)
            else:
                out_table.info("No results in the selected date window after filtering.")


if __name__ == "__main__":
    app()
