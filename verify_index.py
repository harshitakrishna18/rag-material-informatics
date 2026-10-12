import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/index/faiss.index")
meta = json.load(open("data/index/meta.json"))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = "What are common challenges in materials informatics?"
qemb = model.encode([q], normalize_embeddings=True).astype(np.float32)

scores, ids = index.search(qemb, 5)
print("Scores:", scores[0])
print("Metas:", [meta["metas"][i] for i in ids[0] if i != -1])
