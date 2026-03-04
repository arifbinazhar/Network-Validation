import pandas as pd
from PubTator_API import process_relations_parallel
from open_target_api import (
    get_efo_id,
    get_ensembl_id,
    ot_target_disease_evidence
)

def build_network(input_df):
    validated_df = process_relations_parallel(input_df)

    enriched_rows = []

    for _, row in validated_df.iterrows():
        gene = row["entity"]
        disease = row["disease"]

        ensembl_id, _ = get_ensembl_id(gene)
        efo_id, _ = get_efo_id(disease)

        if ensembl_id and efo_id:
            count, avg_score, sources = ot_target_disease_evidence(
                ensembl_id, efo_id
            )
        else:
            count, avg_score, sources = None, None, None

        row["ot_evidence_count"] = count
        row["ot_avg_score"] = avg_score
        row["ot_sources"] = sources

        enriched_rows.append(row)

    enriched_df = pd.DataFrame(enriched_rows)

    return validated_df, enriched_df


def filter_network(enriched_df):
    strong = enriched_df[
        (enriched_df["ot_evidence_count"] > 70) &
        (enriched_df["ot_avg_score"] > 15)
    ].copy()

    strong["association_type"] = "Strong"

    medium = enriched_df[
        (enriched_df["ot_evidence_count"] > 15) &
        (enriched_df["ot_avg_score"] >= 10) &
        (enriched_df["ot_avg_score"] <= 15)
    ].copy()

    medium["association_type"] = "Medium"

    filtered_df = pd.concat([strong, medium])
    filtered_df = filtered_df.drop_duplicates(subset=["entity", "disease"])

    return filtered_df


def extract_pmids(df):
    pmids = set()
    for val in df["pmids"].dropna():
        for p in str(val).split(";"):
            pmids.add(p.strip())
    return list(pmids)