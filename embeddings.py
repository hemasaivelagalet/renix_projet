import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Load the dataset with the 'Content' column
data = pd.read_csv("emergency_preparedness_data.csv")  # Replace with the actual path to your CSV file
texts = data['Content'].tolist()  # Text data to embed

# Convert all entries in `texts` to strings and filter out any non-string entries
texts = [str(text) for text in texts if isinstance(text, str)]

# Step 2: Load pre-trained tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Step 3: Function to generate embeddings for each text
def generate_embedding(text):
    # Tokenize and convert text to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Pass through model to get the last hidden state
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to obtain a single embedding vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Step 4: Create embeddings for all texts in the dataset
embeddings = [generate_embedding(text) for text in texts]
embedding_matrix = np.vstack(embeddings).astype("float32")  # Convert to float32 for FAISS compatibility

# Step 5: Initialize the FAISS index and add embeddings
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance-based index
index.add(embedding_matrix)  # Add embeddings to the index

# Step 6: Verify the number of embeddings indexed
print("Generated embeddings for all texts.")
print(f"Number of embeddings indexed: {index.ntotal}")

# Step 7: Save the FAISS index to a file
faiss.write_index(index, "faiss_index.bin")
print("FAISS index has been created and saved as 'faiss_index.bin'.")
