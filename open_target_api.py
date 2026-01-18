import requests
import pandas as pd
import time

# --------------------------
# 1️⃣  DISEASE → EFO ID
# --------------------------
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
        print(f"✅ {disease_name} → {label} ({efo_id})")
        return efo_id, label
    except Exception as e:
        print(f"⚠️ Error fetching EFO ID for {disease_name}: {e}")
        return None, None


# --------------------------
# 2️⃣  GENE → Ensembl ID
# --------------------------
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


# --------------------------
# 3️⃣  FETCH ALL IDS
# --------------------------
def fetch_all_ids(diseases, genes):
    """Get all EFO and Ensembl IDs and return as DataFrames."""
    disease_records, gene_records = [], []

    for disease in diseases:
        efo_id, efo_label = get_efo_id(disease)
        disease_records.append({"disease": disease, "efo_id": efo_id, "efo_label": efo_label})
        time.sleep(0.4)

    for gene in genes:
        ens_id, desc = get_ensembl_id(gene)
        gene_records.append({"gene": gene, "ensembl_id": ens_id, "gene_desc": desc})
        time.sleep(0.4)

    df_disease = pd.DataFrame(disease_records)
    df_gene = pd.DataFrame(gene_records)
    return df_disease, df_gene


# --------------------------
# 4️⃣  MERGE WITH PUBTATOR OUTPUT
# --------------------------
def merge_with_pubtator(pubtator_csv, output_csv="validated_with_ids.csv"):
    """Merge EFO & Ensembl IDs into PubTator validation results."""
    df_pub = pd.read_csv(pubtator_csv)
    df_pub.columns = [c.lower().strip() for c in df_pub.columns]

    diseases = sorted(df_pub["disease"].dropna().unique())
    genes = sorted(df_pub["entity"].dropna().unique())

    df_disease, df_gene = fetch_all_ids(diseases, genes)

    # Merge IDs
    df_merged = (
        df_pub
        .merge(df_disease, how="left", left_on="disease", right_on="disease")
        .merge(df_gene, how="left", left_on="entity", right_on="gene")
    )

    df_merged.to_csv(output_csv, index=False)
    print(f"\n✅ Merged dataset saved to {output_csv}")
    return df_merged
