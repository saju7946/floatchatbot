import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load text
with open("ocean_knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

documents = text.split("\n\n")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Save embeddings
np.save("embeddings.npy", embeddings)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "faiss_index.index")

print("Index built successfully!")
