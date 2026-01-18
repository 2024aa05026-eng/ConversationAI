import os
os.environ["TRANSFORMERS_NO_TF"] = "1"


import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

with open("data/corpus_chunks.json") as f:
    data = json.load(f)

texts = [d["text"] for d in data if len(d["text"].strip()) > 20]

print("Total chunks loaded:", len(texts))

# Safety check
if len(texts) == 0:
    raise ValueError("Corpus is empty. Run ingest.py first and verify data.")

embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "data/faiss.index")
np.save("data/embeddings.npy", embeddings)

print("Dense index created successfully")
