import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

# Load your dataset
data = pd.read_csv("emergency_preparedness_data.csv")  # Replace with your actual dataset path
texts = data['Content'].tolist()  # Text content to embed

# Convert all entries in `texts` to strings and filter out any non-string entries
texts = [str(text) for text in texts if isinstance(text, str)]

# Load a pre-trained tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to generate embeddings for each text
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(**inputs)
    # Use mean pooling to get a single vector for the entire text
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Generate embeddings for all texts in the dataset
embeddings = [generate_embedding(text) for text in texts]
embedding_matrix = np.vstack(embeddings).astype("float32")  # Convert to float32 for FAISS compatibility

# Initialize the FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance-based index
index.add(embedding_matrix)  # Add embeddings to the index

# Save the FAISS index to a file
faiss.write_index(index, "faiss_index.bin")

print("FAISS index has been created and saved as 'faiss_index.bin'.")
