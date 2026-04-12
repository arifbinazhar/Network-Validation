# BioKnowledge Explorer
### RAG-Based Drug Target Validation & Gene-Disease Network Analysis

> **M.Sc. Bioinformatics Research Project — SPPU**
> Developed by Arif Bin Azhar | Guided by Dr. Smita Saxena & Dr. Manali Joshi

---

## Overview

**BioKnowledge Explorer** is an end-to-end, literature-validated drug target discovery and gene-disease network analysis platform. It combines multi-source biomedical evidence — PubMed literature, PubTator annotations, and the Open Targets Platform — with a Retrieval-Augmented Generation (RAG) pipeline to allow researchers to interactively explore, validate, and rank gene-disease associations.

The system is currently being used to **clean, curate, and validate over 626,000 gene-disease/polypharmacology relations** through literature-based evidence, establishing it as a large-scale bioinformatics validation framework.

The application is deployed in a **staged, transparent pipeline** format rather than as a black box, allowing users to understand and interact with each step of the analysis before proceeding to the next.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_1.png?raw=true) 

***The interface of BioKnowledge Explorer***


---

## Scientific Motivation

Drug target identification is a critical bottleneck in drug discovery. Existing databases often contain unvalidated or weakly supported associations. This tool addresses that gap by:

- Cross-validating gene-disease associations from primary literature via PubTator *(which uses AIONER and BioRex, a fine tuned BioBERT)*
- Enriching and scoring targets using the Open Targets Platform
- Applying hybrid composite ranking to prioritize the most evidence-supported targets
- Enabling natural language querying of the curated knowledge base via RAG

The project's scale — **626,000+ relations** being curated and validated — demonstrates its utility beyond a single-disease use case, with an initial focus on polypharmacology networks of vitamins and neuropsychiatric conditions such as schizophrenia.

---

## Pipeline Architecture

The pipeline is organized into **11 sequential stages**, each transparent and user-controlled:

### Stage 1 — Data Ingestion & Input
Users can provide gene-disease entity pairs in two ways:
- **Upload a CSV** file with pre-defined gene-disease pairs
- **Natural Language Query** via an integrated LLM (currently Gemini-2.5-Flash) to generate candidate gene-disease pairs automatically

> *Example: "What can be the drug target for schizophrenia?" → automatically generates a curated list of candidate gene-disease pairs*

---

### Stage 2 — Initial Network Visualization
The raw, unvalidated gene-disease network is visualized as an interactive graph. This provides a baseline view of the input network before any filtering or validation is applied.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/uncleaned_network.png?raw=true) 

---

### Stage 3 — PubTator Validation
Each gene-disease pair is submitted to the **NCBI PubTator API**, which returns co-mention evidence from PubMed literature. The stage reports:
- Total pairs submitted
- Relations found in literature
- Relations not found (potential false positives)
- Associated PubMed IDs (PMIDs) per pair

  ![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_4.png?raw=true) 



---

### Stage 4 — PMID Evidence Extraction
All unique PMIDs retrieved during PubTator validation are aggregated and deduplicated. This stage reports the number of unique PMIDs and the per-pair PMID distribution, providing a quantitative measure of literature support for each gene-disease association.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_PubTator.png?raw=true) 



---

### Stage 5 — Open Targets Enrichment
Each validated gene-disease pair is enriched using the **Open Targets Platform GraphQL API**. Enrichment data includes:
- Ensembl gene IDs and MONDO disease IDs
- Open Targets evidence count (`ot_evidence_count`)
- Average Open Targets association score (`ot_avg_score`)
- Evidence sources (e.g., `europepmc`, `clinical_precedence`, `gwas_credible_sets`)

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_6.png?raw=true)
  

---

### Stage 6 — Data Integration
All upstream data — PubTator results, PMID lists, and Open Targets scores — are merged into a unified, integrated dataframe. This consolidated view is the input to the ranking engine.
![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_Data_integration.png?raw=true)

---

### Stage 7 — Hybrid Composite Ranking
A composite ranking score is calculated by combining:
- **Open Targets evidence count** and **average score**
- **PubTator-derived PMID count** per pair
- **Relation found** flag

Targets are categorized using configurable thresholds:
| Category | Criteria |
|----------|----------|
| **Strong** | count ≥ 70, score ≥ 15 |
| **Moderate** | count ≥ 15, score ≥ 0, relation found |
| **Other** | below moderate thresholds |

*We're looking to build a more robust ranking score calculating model using ML based models.*

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_Data_integration.png?raw=true)

---

### Stage 8 — Filter Strong / Moderate Associations
The final curated network retains only **Strong** and **Moderate** associations, removing poorly supported gene-disease pairs. Each retained pair is tagged with its association strength, providing an interpretable output for downstream analysis.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Image_filtering.png?raw=true)

---

### Stage 9 — Interactive Filtered Knowledge Graph
The validated, filtered network is rendered as an **interactive knowledge graph** with color-coded nodes by association strength. Users can drag nodes, hover for details, and visually explore the curated network structure.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Final_cleaned_network.png?raw=true)

---

### Stage 10 — FAISS Vector Index Construction
PubMed abstracts corresponding to the validated PMIDs are fetched and chunked. These chunks are embedded and stored in a **FAISS vector index**, forming the retrieval backbone of the RAG system.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Unique_PMID_FAISS.png?raw=true)
---

### Stage 11 — Biomedical Query Interface (RAG)
Users can query the curated knowledge base using natural language. The system supports:
- **RAG mode** — retrieves relevant FAISS chunks from curated abstracts and generates evidence-grounded answers
- **LLM Search mode** — falls back to the base LLM if RAG context is insufficient

> *Example query: "Explain the role of COMT in schizophrenia"*
> The system returns a structured evidence summary with supporting PMIDs, mechanistic insights, strength of association, and confidence assessment.

![alt text](https://github.com/arifbinazhar/Network-Validation/blob/main/Application_images/Rag_response.png?raw=true)

---

## Dashboard Tools

Alongside the staged pipeline, the dashboard provides utility tools:
- **PMID Lookup (in-pipeline)** — search whether a specific PMID is present in the current pipeline's results
- **PMID Fetch (Live PubMed)** — fetch full abstract details for any PMID directly from PubMed, regardless of whether it's in the pipeline
- **Curation Table** — browse the full enriched, ranked results in tabular format
- **Export** — download all analysis results (full ranked data, filtered data, enriched data) as CSV files

---

## Key Features

- Multi-source validation: PubTator + Open Targets + PubMed
- LLM-assisted gene-disease pair generation (Gemini-2.5-Flash)
- Hybrid composite ranking with configurable thresholds
- Interactive network graph visualization (input and filtered)
- RAG-powered biomedical Q&A grounded in curated literature and explainantion of the pathways.
- FAISS vector store for fast semantic retrieval
- Large-scale validation: 626,000+ gene-disease/polypharmacology relations (***currently under deployment***)
- CSV export of all results
- Staged, transparent pipeline for reproducibility and interpretability

---

## Repository Structure

```
Network-Validation/
│
├── Application/               # Main Streamlit application modules
├── Project/                   # Project documentation and notebooks
│
├── app.py           # Main entry point for the BioKnowledge Explorer UI
├── PubTator_API.py            # PubTator API integration and validation logic
├── open_target_api.py         # Open Targets Platform GraphQL API integration
├── Setup.py                   # Environment setup utilities
├── streamlit_app.py           # Older interface with only network validation without RAG
│
├── requirements.txt           # Python dependencies
├── pubtator_cache.json        # Cached PubTator API responses
│
├── Folic Acid.csv             # Example input: polypharmacology network (vitamins)
├── validated_relations_pubtator.csv   # Example output: PubTator-validated relations
├── validated_with_OT.csv             # Example output: Open Targets enriched relations
│
├── input_graph.html           # Rendered input network graph
├── filtered_graph.html        # Rendered filtered/validated network graph
│
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- A valid **NCBI email** (for PubMed/PubTator API access)
- A valid **Gemini API key** (Google AI Studio)

### Install dependencies

```bash
git clone https://github.com/arifbinazhar/Network-Validation.git
cd Network-Validation
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run Project/app.py
```

### Configuration (in the sidebar)
1. Enter your **Gemini API Key** and select the model (default: `gemini-2.5-flash`)
2. Enter your **NCBI email** for PubMed/PubTator access
3. Adjust **parallel workers** for processing speed

---

## Deployment

The application has been deployed using **Cloudflare** (after migrating from ngrok due to server stability issues). CDAC Ice-Cloud deployment has also been requested for institutional hosting via the CDAC ICE portal.

---

## Scale & Data

| Metric | Value |
|--------|-------|
| Gene-Disease Relations under validation | **626,000+** |
| Initial focus network | Polypharmacology network of vitamins |
| Disease example shown | Schizophrenia (13 gene targets) |
| Unique PMIDs indexed (schizophrenia example) | 21 |
| Validation sources | PubTator, Open Targets, PubMed |

---

## Roadmap

- Integration of custom trained Relation Extraction (RE) model for additional PMIDs not captured by PubTator
- Full automation mode (single-click end-to-end pipeline)
- Support for ChatGPT and other LLMs for pair generation
- Batch processing for the 626K+ relation validation dataset
- Institutional deployment on CDAC Ice-Cloud
- Extended disease and drug-target network coverage.

---

## Team

| Name | Role | Contact |
|------|------|---------|
| **Arif Bin Azhar** | Developer (M.Sc. Bioinformatics, SPPU) | arifbinazhar03@gmail.com |
| **Dr. Smita Saxena** | Guide | saxenasmita.mca@gmail.com |
| **Dr. Manali Joshi** | Co-Guide | manali.joshi@gmail.com |


---

## Technologies Used

- **Frontend/UI**: Streamlit
- **LLM**: Google Gemini 2.5 Flash (via Gemini API)
- **Vector Store**: FAISS
- **Literature Validation**: NCBI PubTator API
- **Target Scoring**: Open Targets Platform (GraphQL API)
- **Abstract Retrieval**: NCBI PubMed API (Entrez)
- **Network Visualization**: PyVis / NetworkX
- **Data Processing**: Pandas, Python
- **Hosting**: Cloudflare (current), CDAC Ice-Cloud (pending)

---

## License

This project is part of an academic research initiative at **Savitribai Phule Pune University (SPPU)**. Please contact the authors before using this tool or data for commercial purposes.

---

> *"Validating the biology, one relation at a time — at 626,000+ scale."*
