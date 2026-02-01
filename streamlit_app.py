
import streamlit as st
import pandas as pd
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
from pyvis.network import Network
import streamlit.components.v1 as components
import os

# --- Global API URL (ensure this is defined in the Streamlit app's scope)
BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# --- PubTator API Functions (adapted for Streamlit caching and messaging)

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
                st.warning(f"⚠️ Request failed ({r.status_code}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # For non-retryable HTTP errors (e.g., 404, 400), re-raise immediately
                raise e
        except requests.exceptions.RequestException as e:
            # Catch other request exceptions (e.g., connection errors, timeouts)
            wait_time = 2 ** attempt
            st.warning(f"⚠️ Request failed (network/timeout). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    raise requests.exceptions.RequestException(f"Max retries ({max_retries}) exceeded for {url}")

@st.cache_data(show_spinner=False, ttl=3600*24)
def find_entity_id(entity, bioconcept=None, limit=5):
    """Get PubTator entity IDs, cached with Streamlit."""
    params = {"query": entity, "limit": limit}
    if bioconcept:
        params["concept"] = bioconcept
    url = f"{BASE_URL}/entity/autocomplete/"

    try:
        r = safe_get(url, params=params)
        data = r.json()
        if isinstance(data, list) and data:
            return data
        return []
    except requests.RequestException as e:
        st.warning(f"⚠️ Error fetching '{entity}': {e}")
        return []

@st.cache_data(show_spinner=False, ttl=3600*24)
def find_related_entities_safe(entity_id,
                               entity_type="disease",
                               relation_types=("treats","associate"),
                               limit=100):
    """Find related entities; handles list/dict responses safely, cached with Streamlit."""
    url = f"{BASE_URL}/relations"
    last_error = None

    for rel_type in relation_types:
        params = {"e1": entity_id, "limit": limit}
        if rel_type:
            params["type"] = rel_type
        if entity_type:
            params["e2"] = entity_type

        try:
            r = safe_get(url, params=params)
            j = r.json()
            if isinstance(j, dict):
                data = j.get("relations", [])
            elif isinstance(j, list):
                data = j
            else:
                data = []

            if data:
                return {"relations": data, "relation_type": rel_type}

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                continue
            last_error = e
            st.warning(f"⚠️ Request error for {entity_id} ({rel_type}): {e}")
        except requests.exceptions.RequestException as e:
            last_error = e
            st.warning(f"⚠️ Request error for {entity_id} ({rel_type}): {e}")

    return {"relations": [], "relation_type": None, "error": str(last_error) if last_error else None}

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

@st.cache_data(show_spinner=False, ttl=3600*24)
def search_pubtator_evidence(entity_id1, entity_id2, relation_type):
    """Retrieve PubMed IDs mentioning both entities, cached with Streamlit."""
    query = f"relations:{relation_type}|{entity_id1}|{entity_id2}"
    url = f"{BASE_URL}/search/"
    try:
        r = safe_get(url, params={"text": query})
        j = r.json()
        if isinstance(j, dict):
            results = j.get("results", [])
        elif isinstance(j, list):
            results = j
        else:
            results = []
        return results
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Search error for {entity_id1} and {entity_id2}: {e}")
        return []

@st.cache_data(show_spinner=False, ttl=3600*24)
def validate_pair(entity_name, disease_name):
    """Validate one entity–disease pair, cached with Streamlit."""
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
                    st.warning(f"⚠️ Missing entity_id for matched relation {rel_name}")
                    continue

                # st.info(f"🔗 Matched relation: {entity_name} → {rel_name} ({rel_id})")
                pmids_data = search_pubtator_evidence(entity_id, rel_id, rel_type)

                pmids = [str(p.get("pmid")) for p in pmids_data[:3] if p.get("pmid")] # Fixed: cast pmid to str

                result.update({
                    "relation_found": True,
                    "relation_type": rel_type,
                    "pmids": ";".join(pmids) if pmids else None
                })
                break

    except Exception as e:
        result["error"] = str(e)
        st.error(f"An error occurred during validation for {entity_name}-{disease_name}: {e}")
    return result

@st.cache_data(show_spinner=False, ttl=3600*24)
def process_relations_parallel(df, max_workers=4):
    """Run validation in parallel across many pairs with Streamlit progress."""
    results = []
    total_rows = len(df)
    progress_text = "Processing entity-disease pairs..."
    my_bar = st.progress(0, text=progress_text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(validate_pair, row["entity"], row["disease"]): i
            for i, row in df.iterrows()
        }
        for i, future in enumerate(as_completed(future_to_row)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                st.error(f"⚠️ Thread error for row {future_to_row[future]}: {e}")
                results.append({"error": str(e)})

            progress_percentage = (i + 1) / total_rows
            my_bar.progress(progress_percentage, text=f"{progress_text} {i + 1}/{total_rows}")

    my_bar.empty() # Clear the progress bar after completion
    return pd.DataFrame(results)

# --- Open Targets API Functions (adapted for Streamlit caching and messaging)

@st.cache_data(show_spinner=False, ttl=3600*24)
def get_efo_id(disease_name):
    """Fetch the EFO ID for a disease using EBI OLS API, cached with Streamlit."""
    try:
        url = f"https://www.ebi.ac.uk/ols/api/search?q={disease_name}&ontology=efo"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            st.info(f"⚠️ No EFO match for: {disease_name}")
            return None, None
        label = docs[0]["label"]
        efo_id = docs[0]["obo_id"]
        efo_id = efo_id.replace(":", "_") # Replace colon with underscore
        return efo_id, label
    except Exception as e:
        st.warning(f"⚠️ Error fetching EFO ID for {disease_name}: {e}")
        return None, None

@st.cache_data(show_spinner=False, ttl=3600*24)
def get_ensembl_id(gene_name):
    """Fetch Ensembl gene ID for a given gene symbol (human), cached with Streamlit."""
    try:
        url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            st.info(f"⚠️ No Ensembl match for: {gene_name}")
            return None, None
        gene_id = data[0]["id"]
        desc = data[0].get("description", "")
        return gene_id, desc
    except Exception as e:
        st.warning(f"⚠️ Error fetching Ensembl ID for {gene_name}: {e}")
        return None, None

@st.cache_data(show_spinner=False, ttl=3600*24)
def ot_target_disease_evidence(ensembl_id, efo_id):
    """
    Query Open Targets for target–disease evidence (association count & avg score).
    Cached with Streamlit.
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
        st.warning(f"⚠️ OT evidence fetch failed for {ensembl_id}-{efo_id}: {e}")
        return None, None, None

@st.cache_data(show_spinner=False, ttl=3600*24)
def enrich_pubtator_with_open_targets_st(df_pub):
    """Enrich PubTator dataframe with Open Targets data, with Streamlit progress."""
    df_pub_copy = df_pub.copy()
    df_pub_copy.columns = [c.lower().strip() for c in df_pub_copy.columns]

    diseases = sorted(df_pub_copy["disease"].dropna().unique())
    genes = sorted(df_pub_copy["entity"].dropna().unique())

    st.subheader("Fetching EFO and Ensembl IDs...")
    efo_map, ens_map = {}, {}

    efo_progress_bar = st.progress(0, text="Fetching EFO IDs...")
    for i, d in enumerate(diseases):
        efo_map[d] = get_efo_id(d)
        efo_progress_bar.progress((i + 1) / len(diseases), text=f"Fetching EFO IDs: {i+1}/{len(diseases)}")
    efo_progress_bar.empty()

    ens_progress_bar = st.progress(0, text="Fetching Ensembl IDs...")
    for i, g in enumerate(genes):
        ens_map[g] = get_ensembl_id(g)
        ens_progress_bar.progress((i + 1) / len(genes), text=f"Fetching Ensembl IDs: {i+1}/{len(genes)}")
    ens_progress_bar.empty()

    df_pub_copy["efo_id"] = df_pub_copy["disease"].map(lambda d: efo_map.get(d, (None, None))[0])
    df_pub_copy["efo_label"] = df_pub_copy["disease"].map(lambda d: efo_map.get(d, (None, None))[1])
    df_pub_copy["ensembl_id"] = df_pub_copy["entity"].map(lambda g: ens_map.get(g, (None, None))[0])
    df_pub_copy["gene_desc"] = df_pub_copy["entity"].map(lambda g: ens_map.get(g, (None, None))[1])

    st.subheader("Querying Open Targets for associations...")
    ot_counts, ot_scores, ot_sources = [], [], []
    ot_progress_bar = st.progress(0, text="Querying Open Targets...")

    for i, row in df_pub_copy.iterrows():
        if pd.notna(row["ensembl_id"]) and pd.notna(row["efo_id"]):
            count, avg_score, sources = ot_target_disease_evidence(row["ensembl_id"], row["efo_id"])
        else:
            count, avg_score, sources = None, None, None
        ot_counts.append(count)
        ot_scores.append(avg_score)
        ot_sources.append(sources)
        ot_progress_bar.progress((i + 1) / len(df_pub_copy), text=f"Querying Open Targets: {i+1}/{len(df_pub_copy)}")

    ot_progress_bar.empty()

    df_pub_copy["ot_evidence_count"] = ot_counts
    df_pub_copy["ot_avg_score"] = ot_scores
    df_pub_copy["ot_sources"] = ot_sources

    return df_pub_copy

# --- pyvis network graph function ---
def create_network_graph(df, title, filename):
    # Ensure 'entity' and 'disease' columns are strings
    df['entity'] = df['entity'].astype(str)
    df['disease'] = df['disease'].astype(str)

    # Filter out rows where entity or disease is NaN, empty string, or generic placeholders
    df_clean = df.dropna(subset=['entity', 'disease'])
    df_clean = df_clean[df_clean['entity'] != '']
    df_clean = df_clean[df_clean['disease'] != '']

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    net.toggle_physics(True)

    unique_nodes = pd.concat([df_clean['entity'], df_clean['disease']]).unique()
    for node in unique_nodes:
        # Assign colors based on node type
        if node in df_clean['entity'].unique():
            node_color = "#FF0000"  # Red for entities
            node_title = f"Entity: {node}"
        elif node in df_clean['disease'].unique():
            node_color = "#00FF00"  # Green for diseases
            node_title = f"Disease: {node}"
        else:
            node_color = "#FFFFFF"  # Default white for others
            node_title = node
        net.add_node(node, label=node, title=node_title, color=node_color, size=15)

    for _, row in df_clean.iterrows():
        entity = row['entity']
        disease = row['disease']
        relation_type = row.get('relation_type', 'associate') # Default if not found
        score = row.get('ot_avg_score', 0) # Default score
        pmids = row.get('pmids', '') # Default PMIDs

        # Adjust edge width based on score for filtered data
        # For input data, score might be NaN, handle it
        edge_width = score / 5 if pd.notna(score) and score > 0 else 1

        edge_title = f"Relation: {relation_type}<br>Score: {score}<br>PMIDs: {pmids}"
        net.add_edge(entity, disease, title=edge_title, value=edge_width, width=edge_width)

    net.save_graph(filename)
    return filename

# --- Streamlit App Layout ---
st.set_page_config(
    page_title='Potential Drug Target Predictor',
    page_icon=':dna:',
    layout='wide'
)

st.title('Potential Drug Target Predictor')

# File Uploader section
with st.expander('Upload your CSV file here'):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Cache Clearing Mechanism
if st.sidebar.button('Clear Cache'):
    st.cache_data.clear()
    st.sidebar.success('Cache cleared!')

# Main application logic
if uploaded_file is not None:
    # Display Input Data
    st.subheader("Your input data")
    input_df = pd.read_csv(uploaded_file)
    st.dataframe(input_df)

    st.write(f"Loaded {len(input_df)} rows from the CSV file.")

    # Display network graph for input data
    st.subheader("Input Data Network Graph")
    with st.spinner("Generating network graph for input data..."):
        input_graph_html = create_network_graph(input_df, "Input Data Associations", "input_graph.html")
        HtmlFile = open(input_graph_html, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=750)
    HtmlFile.close()

    if st.button("Start Processing >>> "): # Button to trigger processing
        st.subheader("Processing Status (PubTator)") 

        with st.spinner("Running PubTator validation..."):
            validated_df_pubtator = process_relations_parallel(input_df, max_workers=4)
        st.success("PubTator validation complete!")
        st.dataframe(validated_df_pubtator)

        st.subheader("Processing Status (Open Targets)")
        with st.spinner("Enriching with Open Targets data..."):
            enriched_df_ot = enrich_pubtator_with_open_targets_st(validated_df_pubtator)
        st.success("Open Targets enrichment complete!")

        st.subheader("Merged Results")
        final_merged_df = enriched_df_ot.copy()
        st.dataframe(final_merged_df)

        st.subheader("Filtered Associations")
        # Filter for strong associations
        strong_associations = final_merged_df[
            (final_merged_df['ot_evidence_count'] > 70) &
            (final_merged_df['ot_avg_score'] > 15)
        ].copy()
        strong_associations['association_type'] = 'Strong'

        # Filter for medium associations
        medium_associations = final_merged_df[
            (final_merged_df['ot_evidence_count'] > 15) &
            (final_merged_df['ot_avg_score'] >= 10) &
            (final_merged_df['ot_avg_score'] <= 15)
        ].copy()
        medium_associations['association_type'] = 'Medium'

        # Combine strong and medium associations
        filtered_associations_df = pd.concat([strong_associations, medium_associations])
        filtered_associations_df = filtered_associations_df.drop_duplicates(subset=['entity', 'disease'])

        if not filtered_associations_df.empty:
            st.success(f"Found {len(filtered_associations_df)} strong/medium associations.")
            st.dataframe(filtered_associations_df)

            # Display network graph for filtered data
            st.subheader("Final Network Graph")
            with st.spinner("Generating final network graph..."):
                filtered_graph_html = create_network_graph(filtered_associations_df, "Filtered Associations", "filtered_graph.html")
                HtmlFile = open(filtered_graph_html, 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=750)
            HtmlFile.close()

            # --- Visualizations ---
            st.subheader("Visualizations (Altair)")

            # Bar chart of association types
            if not filtered_associations_df.empty:
                st.write("### Distribution of Association Types")
                association_type_counts = filtered_associations_df['association_type'].value_counts().reset_index()
                association_type_counts.columns = ['association_type', 'count'] # Corrected column names
                chart_type = alt.Chart(association_type_counts).mark_bar().encode(
                    x=alt.X('association_type', axis=alt.Axis(title='Association Type')),
                    y=alt.Y('count', axis=alt.Axis(title='Number of Associations')),
                    tooltip=['association_type', 'count'],
                    color=alt.Color('association_type', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']))
                ).properties(
                    title='Number of Strong vs. Medium Associations'
                )
                st.altair_chart(chart_type, use_container_width=True)
            else:
                st.info("No strong or medium associations to visualize.")

            # Scatter plot of ot_evidence_count vs ot_avg_score
            st.write("### Association Strength Overview")
            chart_scatter = alt.Chart(filtered_associations_df).mark_circle(size=60).encode(
                x=alt.X('ot_evidence_count', axis=alt.Axis(title='Open Targets Evidence Count')),
                y=alt.Y('ot_avg_score', axis=alt.Axis(title='Open Targets Average Score')),
                color=alt.Color('association_type', title='Association Type', scale=alt.Scale(range=['#1f77b4', '#ff7f0e'])),
                tooltip=['entity', 'disease', 'ot_evidence_count', 'ot_avg_score', 'association_type', 'ot_sources']
            ).properties(
                title='Open Targets Evidence Count vs. Average Score'
            ).interactive()
            st.altair_chart(chart_scatter, use_container_width=True)

            st.markdown("--- ")
            st.download_button(
                label="Download Filtered Associations CSV",
                data=filtered_associations_df.to_csv(index=False).encode('utf-8'),
                file_name="filtered_associations.csv",
                mime="text/csv",
            )
        else:
            st.info("No strong or medium associations found based on the criteria.")

else:
    st.info("Please upload a CSV file to begin the analysis.")
