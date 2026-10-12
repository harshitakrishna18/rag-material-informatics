import os
import numpy as np
import faiss
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

import json

IDX_PATH = "data/index/faiss.index"
META_PATH = "data/index/meta.json"

@st.cache_resource
def load_assets():
    index = faiss.read_index(IDX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = data["texts"]
    metas = data["metas"]
    model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN)
    return index, texts, metas, model

def retrieve(query: str, k: int, index, texts, metas, model):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append((float(score), texts[idx], metas[idx]))
    return results

def build_prompt(question: str, retrieved):
    ctx = []
    for score, chunk, meta in retrieved:
        cite = f"[{meta['doc']} | chunk {meta['chunk']}]"
        ctx.append(f"{cite}\n{chunk}")
    context = "\n\n---\n\n".join(ctx)

    return f"""You are a materials-informatics assistant.
Use ONLY the provided context from arXiv papers.
If the answer is not supported, say: "I don't know based on the provided papers."
Cite sources like [paper_file | chunk N] after key claims.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

st.set_page_config(page_title="arXiv Materials RAG", page_icon="📚", layout="wide")
st.title("📚 arXiv Materials Informatics RAG")

if not HF_TOKEN:
    st.warning("Missing HF_TOKEN. Generation might fail or be limit-restricted.")

try:
    index, texts, metas, model = load_assets()
except Exception as e:
    st.error(f"Index not found / not built yet: {e}\nRun scripts 1→2→3 first.")
    st.stop()

k = st.sidebar.slider("Top-k", 2, 10, 5)
gen_model = st.sidebar.text_input("HF LLM model", value=GEN_MODEL)

q = st.text_input("Ask a question about materials informatics literature")
if q:
    retrieved = retrieve(q, k=k, index=index, texts=texts, metas=metas, model=model)
    with st.expander("Retrieved context"):
        for score, chunk, meta in retrieved:
            st.markdown(f"**{meta['doc']} | chunk {meta['chunk']}** (score={score:.3f})")
            st.text(chunk[:1200] + ("..." if len(chunk) > 1200 else ""))
    prompt = build_prompt(q, retrieved)
    ans = hf_generate(prompt, model=gen_model)
    st.subheader("Answer")
    st.write(ans)
