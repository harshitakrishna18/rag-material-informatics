from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

TXT_DIR = Path("data/text")
IDX_DIR = Path("data/index")
IDX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 2200
OVERLAP = 300

def chunk_text(s: str):
    s = s.replace("\x00", "")
    chunks = []
    i = 0
    while i < len(s):
        j = min(len(s), i + CHUNK_CHARS)
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = j - OVERLAP
        if i < 0:
            i = 0
    return [c for c in chunks if c.strip()]

model = SentenceTransformer(EMBED_MODEL, device="cpu")

texts = []
metas = []

for txt_path in TXT_DIR.glob("*.txt"):
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw)
    for ci, c in enumerate(chunks):
        texts.append(c)
        metas.append({"doc": txt_path.name, "chunk": ci})

print("Total chunks:", len(texts), flush=True)

# Encode one by one to avoid any multiprocessing/loky crashes
embs = []
for i in range(0, len(texts), 32):
    batch = texts[i:i+32]
    batch_emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    embs.append(batch_emb)
    print(f"Encoded {min(i+32, len(texts))}/{len(texts)} chunks...")

emb = np.vstack(embs).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index, str(IDX_DIR / "faiss.index"))
np.array(texts, dtype=object).dump(str(IDX_DIR / "texts.npy"))
np.array(metas, dtype=object).dump(str(IDX_DIR / "metas.npy"))

print("Saved index to data/index/")

