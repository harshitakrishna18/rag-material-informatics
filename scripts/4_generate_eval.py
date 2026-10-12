import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset
with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)

# Load FAISS index
index = faiss.read_index("data/index/faiss.index")
texts = np.load("data/index/texts.npy", allow_pickle=True)
metas = np.load("data/index/metas.npy", allow_pickle=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# Prepare BM25 index
tokenized_texts = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized_texts)

def get_faiss_context(query, k=5):
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, k)
    return [texts[i] for i in ids[0] if i != -1]

def get_bm25_context(query, k=5):
    tokenized_query = query.lower().split()
    top_n = bm25.get_top_n(tokenized_query, texts, n=k)
    return top_n

def generate_answer(question, context_chunks=[]):
    if context_chunks:
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = f"Use ONLY the provided context from arXiv papers to answer the question.\n\nCONTEXT:\n{context_str}\n\nQUESTION: {question}\n\nANSWER:"
    else:
        prompt = f"Answer the following question about materials informatics:\n\nQUESTION: {question}\n\nANSWER:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

results = []
for i, item in enumerate(eval_data):
    print(f"Processing question {i+1}/{len(eval_data)}...")
    q = item["question"]
    
    # Baseline
    baseline_ans = generate_answer(q)
    
    # BM25 Ablation
    bm25_ctx = get_bm25_context(q)
    bm25_ans = generate_answer(q, bm25_ctx)
    
    # RAG System (FAISS)
    faiss_ctx = get_faiss_context(q)
    rag_ans = generate_answer(q, faiss_ctx)
    
    results.append({
        "question": q,
        "reference": item["reference"],
        "baseline_answer": baseline_ans,
        "ablation_answer": bm25_ans,
        "system_answer": rag_ans
    })

with open("eval_results_mock.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation generation complete. Saved to eval_results_mock.json")
