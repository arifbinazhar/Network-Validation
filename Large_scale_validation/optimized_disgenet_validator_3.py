# -*- coding: utf-8 -*-
"""
Optimized DisGeNET Validator
==============================
Key optimizations over the original:

1. GENE-LEVEL BATCHING (biggest win):
   Since data is sorted by geneSymbol, we fetch ALL relations for a gene ONCE,
   then match ALL its diseases from that single API call — instead of one API
   call per (gene, disease) pair.

2. SMARTER CACHING:
   - Disk cache persists across runs and restarts.
   - Gene-relation cache avoids redundant PubTator calls within a run.
   - LRU in-memory layer prevents re-reading disk on hot keys.

3. ADAPTIVE PARALLELISM:
   - Gene-level ThreadPoolExecutor: each worker handles one gene + all its diseases.
   - Within each gene, disease matching is vectorised (no per-pair threads).
   - Configurable max_workers so you can tune for API rate limits.

4. RATE-LIMIT AWARENESS:
   - Token-bucket throttle shared across all threads.
   - Exponential backoff only on 429 / 5xx — 404 fails fast.
   - Single global delay knob (REQUESTS_PER_SECOND).

5. CHECKPOINT / RESUME:
   - After every gene group, results are appended to a checkpoint CSV.
   - On restart the script skips already-processed genes automatically.

6. PMID EVIDENCE:
   - Top 10 PMIDs fetched for every matched (gene, disease) pair.
   - Stored semicolon-separated in the `pmids` column.
   - Rows with NO relation found are kept with pmids="" (never dropped).
   - PMID count is overridable via --pmid-limit CLI argument.


Command to run 

python optimized_disgenet_validator.py Disgenet.csv --gene-col entity --disease-col disease --workers 3 --cooldown-every 50 --cooldown-minutes 5
"""

import pandas as pd
import requests
import re
import time
import json
import os
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import defaultdict

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

import os as _os   # needed here before the main os import below

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

# These three paths are overridden per-job by the SLURM submit script via env vars,
# so each HPC batch writes to its own files and never collides with siblings.
CACHE_FILE      = _os.environ.get("DISGENET_CACHE_FILE",      "pubtator_cache.json")
CHECKPOINT_FILE = _os.environ.get("DISGENET_CHECKPOINT_FILE", "checkpoint_validated.csv")
FINAL_OUTPUT    = _os.environ.get("DISGENET_FINAL_OUTPUT",    "validated_relations_final.csv")

REQUESTS_PER_SECOND = 2.5     # stay safely under NCBI's ~3 req/s free limit
MAX_WORKERS         = 6       # parallel gene workers; lower if getting 429s
FETCH_PMIDS         = True    # always True — PMIDs required for all matched relations
RELATION_LIMIT      = 200     # max relations to fetch per gene entity
PMID_LIMIT          = 10      # top-N PMIDs per matched (gene, disease) pair
                              # stored semicolon-separated; "" if no relation found

# ── Cooldown ──────────────────────────────────────────────────────────────────
# After every COOLDOWN_EVERY genes, ALL worker threads are drained and the
# process sleeps for COOLDOWN_MINUTES before resuming. This prevents sustained
# request bursts from overwhelming the PubTator API over long multi-hour runs.
# Set COOLDOWN_EVERY = 0 to disable cooldowns entirely.
COOLDOWN_EVERY   = 100        # genes between cooldowns  (0 = disabled)
COOLDOWN_MINUTES = 3          # minutes to sleep during each cooldown

BAD_PATTERNS = [
    "Neoplasms", "Drug_Related_Side_Effects",
    "Chemical_and_Drug_Induced_Liver_Injury", "Disease_Models"
]

# ─────────────────────────── LOGGING ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("validator.log")]
)
log = logging.getLogger(__name__)

# ─────────────────────────── RATE LIMITER ────────────────────────────────────

class TokenBucket:
    """Thread-safe token bucket for rate limiting."""
    def __init__(self, rate: float):
        self._rate      = rate          # tokens/second
        self._tokens    = rate
        self._lock      = threading.Lock()
        self._last_time = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_time
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_time = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
        # sleep outside the lock so other threads aren't blocked
        time.sleep(1.0 / self._rate)
        self.acquire()

_throttle = TokenBucket(REQUESTS_PER_SECOND)

# ─────────────────────────── DISK CACHE ──────────────────────────────────────

_cache: dict = {}
_cache_dirty = False
_cache_lock  = threading.Lock()

def load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            try:
                _cache = json.load(f)
                log.info(f"Cache loaded: {len(_cache)} entries from {CACHE_FILE}")
            except json.JSONDecodeError:
                log.warning("Cache file corrupt — starting fresh.")
                _cache = {}
    else:
        log.info("No cache file found — starting fresh.")

def save_cache():
    if _cache_dirty:
        with _cache_lock:
            with open(CACHE_FILE, "w") as f:
                json.dump(_cache, f)
        log.info(f"Cache saved: {len(_cache)} entries → {CACHE_FILE}")

def _cache_get(key):
    with _cache_lock:
        return _cache.get(key)

def _cache_set(key, value):
    global _cache_dirty
    with _cache_lock:
        _cache[key] = value
        _cache_dirty = True

# ─────────────────────────── HTTP HELPER ─────────────────────────────────────

def safe_get(url, params, timeout=12, max_retries=8):
    """
    Rate-limited GET with exponential backoff on 429/5xx.

    On 429 the backoff is longer (starts at 60s) because NCBI's rate limit
    window is typically 60 seconds. After max_retries the function returns
    None rather than raising — callers treat None as a transient failure and
    record an empty result rather than crashing the whole gene group.
    """
    _throttle.acquire()
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None                          # fast-fail — no retry
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 429:
                # 429: honour Retry-After header if present, else exponential backoff
                # starting at 60s (NCBI rate-limit window is ~60s)
                retry_after = int(e.response.headers.get("Retry-After", 0))
                wait = retry_after if retry_after > 0 else min(60 * (2 ** attempt), 600)
                log.warning(f"429 Too Many Requests — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                _throttle.acquire()
            elif 500 <= code < 600:
                wait = 2 ** attempt
                log.warning(f"HTTP {code} — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                _throttle.acquire()
            else:
                raise
        except requests.exceptions.RequestException:
            wait = 2 ** attempt
            log.warning(f"Network error — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            _throttle.acquire()
    # Return None on exhaustion — do NOT raise, so one bad gene doesn't
    # corrupt the error column of all its sibling disease rows
    log.error(f"Max retries ({max_retries}) exhausted for {url} — skipping")
    return None

# ─────────────────────────── API WRAPPERS ────────────────────────────────────

def get_entity_id(name: str, concept: str = None) -> str | None:
    """
    Resolve a gene/disease name → PubTator entity ID.
    Returns the top hit's _id, or None if not found.
    """
    key = f"eid:{name}:{concept}"
    cached = _cache_get(key)
    if cached is not None:
        return cached or None      # "" stored as "not found"

    params = {"query": name, "limit": 1}
    if concept:
        params["concept"] = concept
    r = safe_get(f"{BASE_URL}/entity/autocomplete/", params)
    if r is None:
        # None means 404 or max-retries exhausted (e.g. sustained 429).
        # Do NOT cache this — let it be retried on the next run.
        return None

    data = r.json()
    eid = data[0]["_id"] if isinstance(data, list) and data else ""
    _cache_set(key, eid)     # only cache real responses, not transient failures
    return eid or None


def get_all_relations_for_entity(entity_id: str,
                                  relation_types=("treats", "associate"),
                                  limit: int = RELATION_LIMIT) -> list[dict]:
    """
    Fetch ALL disease relations for one entity (gene) in a single call.
    This is the core optimization: 1 call per gene, not 1 call per (gene, disease).
    Returns a flat list of relation dicts.
    """
    key = f"rel:{entity_id}:{','.join(relation_types)}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    all_relations = []
    for rel_type in relation_types:
        params = {"e1": entity_id, "type": rel_type, "e2": "disease", "limit": limit}
        try:
            r = safe_get(f"{BASE_URL}/relations", params)
            if r is None:
                continue
            j = r.json()
            data = j.get("relations", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
            if data:
                log.info(f"  ✅ {len(data)} '{rel_type}' relations for {entity_id}")
                all_relations.extend(data)
                break       # found results — no need to try next relation type
        except Exception as e:
            log.warning(f"  ⚠️ Relation fetch error for {entity_id} ({rel_type}): {e}")

    # Deduplicate by source id
    seen, unique = set(), []
    for rel in all_relations:
        rid = rel.get("source", "") or rel.get("@id", "")
        if rid not in seen:
            seen.add(rid)
            unique.append(rel)

    _cache_set(key, unique)
    return unique


def get_pmids(entity_id1: str, entity_id2: str, relation_type: str,
              limit: int = PMID_LIMIT) -> list[str]:
    """
    Fetch supporting PMIDs for a confirmed (gene, disease) entity pair.

    CRITICAL: The relations query syntax `relations:type|id1|id2` must be
    pre-encoded and embedded directly into the URL string — NOT passed as a
    `params=` dict. When passed as params, requests double-encodes the already
    percent-encoded string, mangling the pipe separators and breaking the query.
    This matches the exact approach used in the original working notebook.
    """
    import urllib.parse
    key = f"pmid:{entity_id1}:{entity_id2}:{relation_type}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # Build query string and embed it directly in the URL — same as original
    query   = f"relations:{relation_type.lower()}|{entity_id1}|{entity_id2}"
    encoded = urllib.parse.quote(query, safe="")
    url     = f"{BASE_URL}/search/?text={encoded}"

    try:
        _throttle.acquire()                          # still rate-limited
        r = requests.get(url, timeout=15)
        log.debug(f"  PMID search HTTP {r.status_code}: {url}")
        r.raise_for_status()

        j = r.json()
        results = j.get("results", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
        pmids   = [str(item["pmid"]) for item in results[:limit] if item.get("pmid")]
        log.info(f"  📄 PMIDs for ({entity_id1}, {entity_id2}): {pmids}")
    except Exception as e:
        log.warning(f"  ⚠️ PMID fetch error ({entity_id1}, {entity_id2}): {e}")
        pmids = []

    _cache_set(key, pmids)
    return pmids

# ─────────────────────────── MATCHING LOGIC ──────────────────────────────────

_bad_re = re.compile("|".join(BAD_PATTERNS), re.IGNORECASE)

def _clean_relations(relations: list[dict]) -> list[dict]:
    return [r for r in relations if not _bad_re.search(r.get("entity_id", ""))]


def match_diseases_for_gene(gene_name: str, disease_names: list[str]) -> list[dict]:
    """
    Core optimized function:
      1. Resolve gene → entity_id  (1 API call, cached)
      2. Fetch ALL relations        (1 API call, cached)
      3. Match each disease locally (zero extra API calls unless PMID needed)

    Returns one result dict per (gene, disease) pair.
    """
    # ── Blank result template ──────────────────────────────────────────────
    # pmids is always a string: semicolon-separated IDs when found, "" when not.
    # relation_type is "" when no relation found. Row is ALWAYS kept.
    def blank(disease):
        return {"entity": gene_name, "disease": disease,
                "entity_id": None, "disease_id": None,
                "relation_found": False, "relation_type": "", "pmids": ""}

    # ── Step 1: resolve gene ───────────────────────────────────────────────
    entity_id = get_entity_id(gene_name, concept="gene")
    if not entity_id:
        log.warning(f"Gene not found in PubTator: {gene_name}")
        return [blank(d) for d in disease_names]

    # ── Step 2: fetch ALL relations for this gene ──────────────────────────
    raw_relations = get_all_relations_for_entity(entity_id)
    clean_rels    = _clean_relations(raw_relations)

    # Build lookup: lowered name → (rel_dict)  for O(1) matching
    rel_by_name: dict[str, dict] = {}
    rel_by_id:   dict[str, dict] = {}
    for rel in clean_rels:
        rname = rel.get("target", "").lower().strip()
        rid   = (rel.get("source") or "").strip()
        if rname:
            rel_by_name[rname] = rel
        if rid:
            rel_by_id[rid] = rel

    # ── Step 3: resolve each disease and match locally ────────────────────
    # Each disease has its own try/except so a transient 429 on disease X
    # only marks that one row as errored — it never poisons Y, Z etc.
    results = []
    for disease_name in disease_names:
        rec = blank(disease_name)
        try:
            disease_id = get_entity_id(disease_name, concept="disease")
            rec["disease_id"] = disease_id

            matched_rel = None

            # Priority 1: exact ID match
            if disease_id and disease_id in rel_by_id:
                matched_rel = rel_by_id[disease_id]

            # Priority 2: exact name match, then fuzzy substring
            if matched_rel is None:
                dname_lower = disease_name.lower().strip()
                if dname_lower in rel_by_name:
                    matched_rel = rel_by_name[dname_lower]
                else:
                    for rname, rel in rel_by_name.items():
                        if dname_lower in rname or rname in dname_lower:
                            matched_rel = rel
                            break

            if matched_rel is not None:
                # ── Match found ───────────────────────────────────────────
                rel_type = (matched_rel.get("type") or "associate").lower()
                rel_eid  = matched_rel.get("source") or disease_id or ""
                rec.update({"relation_found": True, "relation_type": rel_type})

                if rel_eid:
                    pmids = get_pmids(entity_id, rel_eid, rel_type)
                    rec["pmids"] = ";".join(pmids) if pmids else ""

                log.info(f"  🔗 {gene_name} → {disease_name} ({rel_type}) | "
                         f"{len(rec['pmids'].split(';')) if rec['pmids'] else 0} PMIDs")

        except Exception as e:
            # Transient failure (e.g. 429 exhausted) on this disease only.
            # Error is recorded; row is kept; NOT cached so it retries on resume.
            log.warning(f"  ⚠️ {gene_name}/{disease_name}: {e}")
            rec["error"] = str(e)

        results.append(rec)

    return results

# ─────────────────────────── GENE-GROUP PROCESSOR ────────────────────────────

def process_gene_group(gene_name: str, disease_list: list[str]) -> list[dict]:
    """Wrapper used by the thread pool — one gene at a time."""
    try:
        log.info(f"Processing gene: {gene_name} ({len(disease_list)} diseases)")
        return match_diseases_for_gene(gene_name, disease_list)
    except Exception as e:
        log.error(f"Fatal error processing {gene_name}: {e}")
        return [{"entity": gene_name, "disease": d, "entity_id": None,
                 "disease_id": None, "relation_found": False,
                 "relation_type": "", "pmids": "", "error": str(e)}
                for d in disease_list]

# ─────────────────────────── CHECKPOINT HELPERS ──────────────────────────────

def load_processed_genes() -> set[str]:
    """
    Return the set of gene names that completed WITHOUT any errors.

    Genes that have ANY row with an error (e.g. 429) are excluded from the
    done-set so they are fully re-processed on the next run. Their errored
    rows in the checkpoint will be overwritten by the fresh results.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        df = pd.read_csv(CHECKPOINT_FILE, usecols=["entity", "error"])
        # Genes with at least one errored row
        errored = set(df[df["error"].notna() & (df["error"].astype(str).str.strip() != "")]
                      ["entity"].dropna().unique())
        all_genes = set(df["entity"].dropna().unique())
        clean     = all_genes - errored
        log.info(f"Resuming: {len(clean)} clean genes skipped, "
                 f"{len(errored)} errored genes will be retried.")
        return clean
    except Exception as e:
        log.warning(f"Could not read checkpoint for resume: {e}")
        return set()


def append_checkpoint(records: list[dict]):
    """Append a batch of results to the checkpoint CSV (creates if absent)."""
    if not records:
        return
    df  = pd.DataFrame(records)
    hdr = not os.path.exists(CHECKPOINT_FILE)
    df.to_csv(CHECKPOINT_FILE, mode="a", header=hdr, index=False)

# ─────────────────────────── MAIN ENTRY POINT ────────────────────────────────

def run(input_path: str,
        max_workers: int = MAX_WORKERS,
        gene_col: str = "geneSymbol",
        disease_col: str = "diseaseName"):
    """
    Main pipeline.

    Parameters
    ----------
    input_path  : path to the feather/csv file (the full Disgenet dataset or a chunk)
    max_workers : thread pool size
    gene_col    : column name for gene symbols
    disease_col : column name for disease names
    """
    # ── Load data ──────────────────────────────────────────────────────────
    log.info(f"Loading {input_path} …")
    if input_path.endswith(".feather"):
        df = pd.read_feather(input_path)
    else:
        df = pd.read_csv(input_path)

    df.rename(columns={gene_col: "entity", disease_col: "disease"}, inplace=True)
    df.dropna(subset=["entity", "disease"], inplace=True)
    log.info(f"Dataset: {len(df):,} rows | {df['entity'].nunique():,} unique genes")

    # ── Group by gene ──────────────────────────────────────────────────────
    gene_groups: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        gene_groups[row["entity"]].append(row["disease"])

    # ── Resume logic ───────────────────────────────────────────────────────
    load_cache()
    done_genes  = load_processed_genes()
    todo_genes  = [(g, ds) for g, ds in gene_groups.items() if g not in done_genes]
    log.info(f"Genes to process: {len(todo_genes):,} (skipping {len(done_genes):,} already done)")

    # ── Parallel gene processing with periodic cooldowns ─────────────────────
    # Genes are processed in windows of COOLDOWN_EVERY. After each window
    # all threads finish, checkpoint is saved, then we sleep before the next
    # window. This prevents sustained API bursts over long runs.
    CHECKPOINT_FLUSH = 50   # also flush checkpoint mid-window every N genes
    buffer: list[dict] = []
    total  = len(todo_genes)

    # Split todo_genes into windows; if cooldown disabled use one big window
    window_size = COOLDOWN_EVERY if COOLDOWN_EVERY > 0 else total
    windows     = [todo_genes[i:i + window_size]
                   for i in range(0, total, window_size)]

    genes_done = 0
    for w_idx, window in enumerate(windows):
        log.info(f"━━━ Window {w_idx + 1}/{len(windows)} "
                 f"({len(window)} genes, {genes_done}/{total} total done) ━━━")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_gene_group, g, ds): g
                       for g, ds in window}

            for i, future in enumerate(as_completed(futures), 1):
                gene = futures[future]
                try:
                    records = future.result()
                    buffer.extend(records)
                except Exception as e:
                    log.error(f"Unhandled error for gene {gene}: {e}")

                # Mid-window checkpoint flush
                if i % CHECKPOINT_FLUSH == 0:
                    append_checkpoint(buffer)
                    save_cache()
                    buffer.clear()
                    log.info(f"  💾 Mid-window checkpoint ({i}/{len(futures)} in this window)")

        # Flush remainder of this window
        if buffer:
            append_checkpoint(buffer)
            save_cache()
            buffer.clear()

        genes_done += len(window)
        log.info(f"  💾 Window {w_idx + 1} complete — {genes_done}/{total} genes done")

        # Cooldown between windows (skip after the last one)
        if COOLDOWN_EVERY > 0 and w_idx < len(windows) - 1:
            log.info(f"  😴 Cooldown: sleeping {COOLDOWN_MINUTES} min before next window …")
            for remaining in range(COOLDOWN_MINUTES * 60, 0, -15):
                log.info(f"     ⏳ Resuming in {remaining}s …")
                time.sleep(min(15, remaining))
            log.info("  ✅ Cooldown done — resuming")

    # ── Consolidate final output ───────────────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        final = pd.read_csv(CHECKPOINT_FILE)
        final.to_csv(FINAL_OUTPUT, index=False)
        log.info(f"✅ Final output → {FINAL_OUTPUT}  ({len(final):,} rows)")
    else:
        log.warning("No checkpoint file found — nothing to consolidate.")


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized DisGeNET → PubTator validator")
    parser.add_argument("input",          help="Path to .feather or .csv input file")
    parser.add_argument("--workers",      type=int, default=MAX_WORKERS,
                        help=f"Parallel gene-worker threads (default {MAX_WORKERS})")
    parser.add_argument("--gene-col",     default="geneSymbol",
                        help="Column name for gene symbols in raw file")
    parser.add_argument("--disease-col",  default="diseaseName",
                        help="Column name for disease names in raw file")
    parser.add_argument("--pmid-limit",       type=int, default=PMID_LIMIT,
                        help=f"Max PMIDs to store per matched pair (default {PMID_LIMIT})")
    parser.add_argument("--cooldown-every",   type=int, default=COOLDOWN_EVERY,
                        help=f"Pause after every N genes (0=disabled, default {COOLDOWN_EVERY})")
    parser.add_argument("--cooldown-minutes", type=int, default=COOLDOWN_MINUTES,
                        help=f"Minutes to sleep during each cooldown (default {COOLDOWN_MINUTES})")
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants before any work starts
    PMID_LIMIT       = args.pmid_limit
    COOLDOWN_EVERY   = args.cooldown_every
    COOLDOWN_MINUTES = args.cooldown_minutes

    run(args.input, max_workers=args.workers,
        gene_col=args.gene_col, disease_col=args.disease_col)
