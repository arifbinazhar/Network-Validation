import requests
import pandas as pd
import time

# ============================================
# 1️⃣  DISEASE → EFO ID (OLS)
# ============================================
def get_efo_id(disease_name):
    """Fetch the EFO ID for a disease using EBI OLS API."""
    try:
        url = f"https://www.ebi.ac.uk/ols/api/search?q={disease_name}&ontology=efo"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            print(f"⚠️ No EFO match for: {disease_name}")
            return None, None
        label = docs[0]["label"]
        efo_id = docs[0]["obo_id"]
        efo_id = efo_id.replace(":", "_") # Replace colon with underscore
        print(f"✅ {disease_name} → {label} ({efo_id})")
        return efo_id, label
    except Exception as e:
        print(f"⚠️ Error fetching EFO ID for {disease_name}: {e}")
        return None, None


# ============================================
# 2️⃣  GENE → Ensembl ID (Ensembl REST)
# ============================================
def get_ensembl_id(gene_name):
    """Fetch Ensembl gene ID for a given gene symbol (human)."""
    try:
        url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            print(f"⚠️ No Ensembl match for: {gene_name}")
            return None, None
        gene_id = data[0]["id"]
        desc = data[0].get("description", "")
        print(f"✅ {gene_name} → {gene_id}")
        return gene_id, desc
    except Exception as e:
        print(f"⚠️ Error fetching Ensembl ID for {gene_name}: {e}")
        return None, None


# ============================================
# 3️⃣  TARGET–DISEASE EVIDENCE (Open Targets)
# ============================================
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

def ot_target_disease_evidence(ensembl_id, efo_id):
    """
    Query Open Targets for target–disease evidence (association count & avg score).
    """
    query = """
    query targetDiseaseEvidence($ensemblId: String!, $efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        evidences(ensemblIds: [$ensemblId]) {
          count
          rows {
            datasourceId
            resourceScore
          }
        }
      }
    }
    """
    variables = {"ensemblId": ensembl_id, "efoId": efo_id}
    try:
        r = requests.post(
            OT_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        disease_data = r.json().get("data", {}).get("disease", {})
        ev = disease_data.get("evidences", {})
        rows = ev.get("rows", [])
        if rows:
            count = ev.get("count", len(rows))
            avg_score = sum((row.get("resourceScore") or 0) for row in rows) / len(rows)
            sources = ";".join(sorted({r.get("datasourceId") for r in rows if r.get("datasourceId")}))
            return count, round(avg_score, 3), sources
        return 0, None, None
    except Exception as e:
        print(f"⚠️ OT evidence fetch failed for {ensembl_id}-{efo_id}: {e}")
        return None, None, None


# ============================================
# 4️⃣  MERGE ALL IDS + EVIDENCE INTO PUBTATOR
# ============================================
def enrich_pubtator_with_open_targets(pubtator_csv, output_csv="validated_with_OT.csv"):
    df_pub = pd.read_csv(pubtator_csv)
    df_pub.columns = [c.lower().strip() for c in df_pub.columns]

    diseases = sorted(df_pub["disease"].dropna().unique())
    genes = sorted(df_pub["entity"].dropna().unique())

    # Step 1: Get all EFO and Ensembl IDs
    efo_map, ens_map = {}, {}

    for d in diseases:
        efo_id, efo_label = get_efo_id(d)
        efo_map[d] = (efo_id, efo_label)
        time.sleep(0.3)

    for g in genes:
        ens_id, desc = get_ensembl_id(g)
        ens_map[g] = (ens_id, desc)
        time.sleep(0.3)

    # Step 2: Enrich PubTator dataframe
    df_pub["efo_id"] = df_pub["disease"].map(lambda d: efo_map.get(d, (None, None))[0])
    df_pub["efo_label"] = df_pub["disease"].map(lambda d: efo_map.get(d, (None, None))[1])
    df_pub["ensembl_id"] = df_pub["entity"].map(lambda g: ens_map.get(g, (None, None))[0])
    df_pub["gene_desc"] = df_pub["entity"].map(lambda g: ens_map.get(g, (None, None))[1])

    # Step 3: Query Open Targets for associations
    ot_counts, ot_scores, ot_sources = [], [], []
    for i, row in df_pub.iterrows():
        if pd.notna(row["ensembl_id"]) and pd.notna(row["efo_id"]):
            count, avg_score, sources = ot_target_disease_evidence(row["ensembl_id"], row["efo_id"])
        else:
            count, avg_score, sources = None, None, None
        ot_counts.append(count)
        ot_scores.append(avg_score)
        ot_sources.append(sources)
        time.sleep(0.4)

    df_pub["ot_evidence_count"] = ot_counts
    df_pub["ot_avg_score"] = ot_scores
    df_pub["ot_sources"] = ot_sources

    # Step 4: Save enriched data
    df_pub.to_csv(output_csv, index=False)
    print(f"\n✅ Enriched dataset saved to {output_csv}")
    return df_pub


enrich_pubtator_with_open_targets("validated_relations_pubtator.csv")