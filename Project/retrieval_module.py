import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from Bio import Entrez

VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")

def fetch_abstract(pmid, email):
    Entrez.email = email
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        article = records["PubmedArticle"][0]
        abstract = article["MedlineCitation"]["Article"].get("Abstract", {})
        if "AbstractText" in abstract:
            return " ".join(abstract["AbstractText"])
    except:
        return None


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def build_faiss_index(documents):
    os.makedirs(VECTOR_DIR, exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(documents, f)

    return index, documents


def load_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        documents = pickle.load(f)

    return model, index, documents