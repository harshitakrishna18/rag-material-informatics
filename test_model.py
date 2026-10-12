from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("Encoding...")
emb = model.encode(["Hello world"])
print("Success:", emb.shape)
