import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Load the dataset and FAISS index
data = pd.read_csv("emergency_preparedness_data.csv")  # Replace with the actual path to your CSV file
texts = data['Content'].tolist()  # Extract the text data

# Load the pre-trained FAISS index
index = faiss.read_index("faiss_index.bin")  # Load the saved FAISS index
print("FAISS index loaded successfully.")

# Step 2: Load a pre-trained tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Step 3: Function to generate embedding for a query
def generate_embedding(text):
    # Tokenize and convert to tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to obtain a single vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Step 4: Define the retrieval function
def retrieve_similar(query, k=5):
    # Generate an embedding for the query
    query_embedding = generate_embedding(query).astype("float32").reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding, k)  # Retrieve top-k results
    
    # Retrieve the top-k texts from the dataset
    results = [texts[i] for i in indices[0]]
    return results

# Step 5: Test the retrieval function with a sample query
query = "How to prepare for a natural disaster?"
results = retrieve_similar(query, k=3)

print("Query:", query)
print("Top Similar Responses:")
for idx, result in enumerate(results):
    print(f"{idx + 1}: {result}")
