import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Load the dataset
data = pd.read_csv("emergency_preparedness_data.csv")  # Replace with the path to your CSV file
if 'Content' not in data.columns:
    raise ValueError("The dataset must contain a 'Content' column.")

texts = data['Content'].tolist()

# Convert all entries in `texts` to strings and filter out any non-string entries
texts = [str(text) for text in texts if isinstance(text, str)]

# Step 2: Load a pre-trained tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Step 3: Function to generate embeddings for each text
def generate_embedding(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Step 4: Generate embeddings for all texts in the dataset
embeddings = [generate_embedding(text) for text in texts]
embedding_matrix = np.vstack(embeddings).astype("float32")  # Convert to float32 for FAISS compatibility

# Optional: Normalize the embeddings
# embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)

# Step 5: Initialize the FAISS index and add embeddings
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance-based index
index.add(embedding_matrix)  # Add embeddings to the index

# Verify the number of embeddings indexed
print(f"Number of embeddings indexed: {index.ntotal}")

# Step 6: Define the retrieval function
def retrieve_similar(query, k=5):
    query_embedding = generate_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results

# Step 7: Test the retrieval function with a sample query
query = "How to prepare for a natural disaster?"
results = retrieve_similar(query, k=3)

print("Query:", query)
print("Top Similar Responses:")
for idx, result in enumerate(results):
    print(f"{idx + 1}: {result}")
