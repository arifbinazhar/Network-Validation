import requests
import pandas as pd
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse
import sys
# import Setup

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
CACHE = {}  # in-memory cache for entity lookups
CACHE_FILE = "pubtator_cache.json" # File to persist the cache


# ========== Utility Functions ==========
def load_disk_cache(cache_file):
    """Loads the cache from a JSON file."""
    global CACHE
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                CACHE = json.load(f)
                print(f"✅ Loaded cache from {cache_file} with {len(CACHE)} entries.")
            except json.JSONDecodeError:
                print(f"⚠️ Error decoding JSON from {cache_file}. Starting with empty cache.")
                CACHE = {}
    else:
        print(f"ℹ️ No cache file found at {cache_file}. Starting with empty cache.")

def save_disk_cache(cache_file):
    """Saves the cache to a JSON file."""
    with open(cache_file, "w") as f:
        json.dump(CACHE, f, indent=4)
    print(f"✅ Saved cache to {cache_file} with {len(CACHE)} entries.")

def safe_get(url, params, timeout=10, max_retries=5, delay_between_requests=0.40):
    """Robustly makes GET requests with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            time.sleep(delay_between_requests) # Global delay after successful request
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429 or (r.status_code >= 500 and r.status_code < 600):
                wait_time = 2 ** attempt
                print(f"⚠️ Request failed ({r.status_code}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # For non-retryable HTTP errors (e.g., 404, 400), re-raise immediately
                raise e
        except requests.exceptions.RequestException as e:
            # Catch other request exceptions (e.g., connection errors, timeouts)
            wait_time = 2 ** attempt
            print(f"⚠️ Request failed (network/timeout). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    raise requests.exceptions.RequestException(f"Max retries ({max_retries}) exceeded for {url}")


# ---------- 1️⃣ FIND ENTITY ----------

def find_entity_id(entity, bioconcept=None, limit=5):
    """Get PubTator entity IDs."""
    key = f"entity:{entity}:{bioconcept}:{limit}"
    if key in CACHE:
        return CACHE[key]

    params = {"query": entity, "limit": limit}
    if bioconcept:
        params["concept"] = bioconcept
    url = f"{BASE_URL}/entity/autocomplete/"

    try:
        r = safe_get(url, params=params)

        data = r.json()
        if isinstance(data, list) and data:
            CACHE[key] = data
            print(f"✅ Found {len(data)} entities for '{entity}'")
            return data
        print(f"⚠️ No matches for '{entity}'")
        CACHE[key] = []
        return []
    except requests.RequestException as e:
        print(f"⚠️ Error fetching '{entity}': {e}")
        CACHE[key] = []
        return []


# ------- 2️⃣ FIND RELATED ENTITIES (robust) ------
def find_related_entities_safe(entity_id,
                               entity_type="disease",
                               relation_types=("treats","associate"),
                               limit=100,
                               sleep_time=0.25):
    """Find related entities; handles list/dict responses safely."""
    url = f"{BASE_URL}/relations"
    last_error = None

    for rel_type in relation_types:
        params = {"e1": entity_id, "limit": limit}
        if rel_type:
            params["type"] = rel_type
        if entity_type:
            params["e2"] = entity_type

        try:
            # Use safe_get here, which handles retries and global sleep
            r = safe_get(url, params=params)

            try:
                j = r.json()
            except ValueError:
                print(f"⚠️ Invalid JSON for {entity_id} ({rel_type})")
                continue

            if isinstance(j, dict):
                data = j.get("relations", [])
            elif isinstance(j, list):
                data = j
            else:
                data = []

            if data:
                print(f"✅ Found {len(data)} '{rel_type}' relations for {entity_id}")
                # Removed time.sleep(sleep_time) as it's handled by safe_get
                return {"relations": data, "relation_type": rel_type}
            else:
                print(f"ℹ️ No relations found for {entity_id} ({rel_type})")
        except requests.exceptions.HTTPError as e:
            # Catch specific non-retryable 404s if they occur after max_retries
            if e.response.status_code == 404:
                print(f"ℹ️ No '{rel_type}' data for {entity_id} (after retries)")
                continue
            last_error = e
            print(f"⚠️ Request error for {entity_id} ({rel_type}): {e}")
        except requests.RequestException as e:
            last_error = e
            print(f"⚠️ Request error for {entity_id} ({rel_type}): {e}")

    return {"relations": [], "relation_type": None, "error": str(last_error) if last_error else None}

# ---------- 3️⃣ SEARCH / EVIDENCE ----------
def debug_search_pubtator_evidence(entity_id1, entity_id2, relation_type, max_results=10):
    """
    Debug helper: query PubTator and return structured PMID evidence
    so downstream RE/RAG code can consume it directly.
    """
    import requests
    import urllib.parse

    query = f"relations:{relation_type.lower()}|{entity_id1}|{entity_id2}"
    encoded_query = urllib.parse.quote(query, safe="")
    url = f"{BASE_URL}/search/?text={encoded_query}"

    print(f"\n🔍 Querying PubTator search:\n{url}")

    try:
        r = requests.get(url, timeout=15)
        print(f"↳ HTTP {r.status_code}")
        r.raise_for_status()

        j = r.json()

        # Normalize PubTator response format
        if isinstance(j, dict):
            results = j.get("results", [])
        elif isinstance(j, list):
            results = j
        else:
            results = []

        pmid_data = []
        for r in results[:max_results]:
            pmid = r.get("pmid")
            if not pmid:
                continue

            pmid_data.append({
                "pmid": str(pmid),
                "relation": relation_type,
                "entity_1": entity_id1,
                "entity_2": entity_id2,
                "source": "PubTator3"
            })

        if pmid_data:
            print(f"✅ Found {len(pmid_data)} PMIDs: {[p['pmid'] for p in pmid_data[:5]]}")
        else:
            print("⚠️ No PMIDs returned for this pair.")

        return pmid_data

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return []


# ---------- 4️⃣ CLEAN RELATIONS ----------
def clean_relations(relations):
    """Remove generic or low-value disease terms."""
    bad_patterns = [
        "Neoplasms", "Drug_Related_Side_Effects",
        "Chemical_and_Drug_Induced_Liver_Injury", "Disease_Models"
    ]
    return [
        r for r in relations
        if not any(re.search(p, r.get("entity_id", ""), re.IGNORECASE) for p in bad_patterns)
    ]



# ---------- 5️⃣ VALIDATE ONE PAIR ----------
def validate_pair(entity_name, disease_name):
    """Validate one entity–disease pair."""
    result = {
        "entity": entity_name,
        "disease": disease_name,
        "entity_id": None,
        "disease_id": None,
        "relation_found": False,
        "relation_type": None,
        "pmids": None
    }
    try:
        entity_data = find_entity_id(entity_name)
        disease_data = find_entity_id(disease_name)
        if not entity_data or not disease_data:
            return result

        entity_id = entity_data[0]["_id"]
        disease_id = disease_data[0]["_id"]
        result.update({"entity_id": entity_id, "disease_id": disease_id})

        related = find_related_entities_safe(entity_id, entity_type="disease")
        cleaned = clean_relations(related.get("relations", []))


        for rel in cleaned:
            rel_id = rel.get("source") or ""
            rel_name = rel.get("target", "").lower()
            rel_type = (rel.get("type") or "associate").lower()
            target_name = disease_name.lower()

            if (
                rel_id == disease_id
                or target_name in rel_name
                or rel_name in target_name
            ):
                if not rel_id:  # fallback: try autocomplete lookup for the disease
                    print(f"⚠️ Missing entity_id for matched relation {rel_name}")
                    continue

                print(f"🔗 Matched relation: {entity_name} → {rel_name} ({rel_id})")
                pmids_data = debug_search_pubtator_evidence(entity_id, rel_id, rel_type)

                pmids = [p.get("pmid") for p in pmids_data[:3] if p.get("pmid")]


                result.update({
                    "relation_found": True,
                    "relation_type": rel_type,
                    "pmids": ";".join(pmids) if pmids else None
                })
                break

    except Exception as e:
        result["error"] = str(e)
    return result


# ---------- 6️⃣ MULTI-THREADED RUN ----------
def process_relations_parallel(df, max_workers=4):
    """Run validation in parallel across many pairs."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(validate_pair, row["entity"], row["disease"]): i
            for i, row in df.iterrows()
        }
        for future in as_completed(future_to_row): # Corrected typo here
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print("⚠️ Thread error:", e)
    return pd.DataFrame(results)

# ---------- 7️⃣ DRIVER ----------


if __name__ == "__main__":
    # Setup argument parser for batch processing
    parser = argparse.ArgumentParser(description="Process entity-disease relations in batches.")
    parser.add_argument('--start', type=int, default=None, help='Start row index for batch processing (inclusive).')
    parser.add_argument('--end', type=int, default=None, help='End row index for batch processing (exclusive).')

    # Use parse_known_args() to ignore kernel-specific arguments in notebooks
    args, unknown = parser.parse_known_args()

    load_disk_cache(CACHE_FILE) # Load cache at the start

    input_file = "Folic Acid.csv"

    df = pd.read_csv(input_file)

    # Slice DataFrame based on arguments
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(df)
    df_batch = df.iloc[start_idx:end_idx]

    print(f"🚀 Starting validation for {len(df_batch)} pairs (rows {start_idx} to {end_idx-1})...")

    validated_df = process_relations_parallel(df_batch, max_workers=4)

    # Dynamically generate output filename
    if args.start is not None or args.end is not None:
        output_file = f"validated_relations_pubtator_{start_idx}-{end_idx}.csv"
    else:
        output_file = "validated_relations_pubtator.csv"

    validated_df.to_csv(output_file, index=False)
    print(f"✅ Done! Results saved to {output_file}")

    save_disk_cache(CACHE_FILE)
