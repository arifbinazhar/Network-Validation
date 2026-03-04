import streamlit as st
import pandas as pd
import os
import numpy as np

from network_module import build_network, filter_network, extract_pmids
from retrieval_module import fetch_abstract, chunk_text, build_faiss_index, load_index
from rag_module import build_prompt, ask_gemini, ask_mistral

st.set_page_config(page_title="Integrated Drug Target RAG", layout="wide")

st.title("Integrated Gene–Disease Network + RAG System")

# Sidebar
st.sidebar.header("LLM Configuration")
llm_choice = st.sidebar.radio("Choose LLM", ["Gemini", "Mistral"])
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
mistral_key = st.sidebar.text_input("Mistral API Key", type="password")
email = st.sidebar.text_input("NCBI Email")

uploaded_file = st.file_uploader("Upload gene,disease CSV", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.dataframe(input_df)

    if st.button("Build Network"):
        validated, enriched = build_network(input_df)
        filtered = filter_network(enriched)

        st.subheader("Filtered Network")
        st.dataframe(filtered)

        pmids = extract_pmids(filtered)

        if pmids:
            st.write(f"Building FAISS from {len(pmids)} PMIDs")

            documents = []

            for pmid in pmids:
                abstract = fetch_abstract(pmid, email)
                if abstract:
                    chunks = chunk_text(abstract)
                    for chunk in chunks:
                        documents.append({
                            "text": chunk,
                            "pmid": pmid,
                            "gene": "",
                            "disease": "",
                            "relation_type": "",
                            "ot_score": ""
                        })

            build_faiss_index(documents)
            st.success("FAISS index built and saved.")

            model, index, documents = load_index()

            query = "Summarize the biological significance of the filtered gene-disease network."
            query_embedding = model.encode([query])
            D, I = index.search(query_embedding, 5)
            retrieved_docs = [documents[i] for i in I[0]]

            prompt = build_prompt(query, retrieved_docs)

            if llm_choice == "Gemini" and gemini_key:
                summary = ask_gemini(prompt, gemini_key)
            elif llm_choice == "Mistral" and mistral_key:
                summary = ask_mistral(prompt, mistral_key)
            else:
                summary = "Please provide API key."

            st.subheader("RAG Network Summary")
            st.write(summary)

    st.subheader("Ask Questions About Network")

    user_query = st.text_input("Enter question")

    if st.button("Ask"):
        model, index, documents = load_index()
        query_embedding = model.encode([user_query])
        D, I = index.search(query_embedding, 5)
        retrieved_docs = [documents[i] for i in I[0]]

        prompt = build_prompt(user_query, retrieved_docs)

        if llm_choice == "Gemini" and gemini_key:
            answer = ask_gemini(prompt, gemini_key)
        elif llm_choice == "Mistral" and mistral_key:
            answer = ask_mistral(prompt, mistral_key)
        else:
            answer = "Provide API key."

        st.write(answer)