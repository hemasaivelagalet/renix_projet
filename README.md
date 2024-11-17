# Emergency Preparedness Chatbot

Welcome to the Emergency Preparedness Chatbot, designed to assist users with emergency preparedness information. This chatbot leverages a conversational interface to provide context-aware, informative responses on topics like disaster communications, flood preparation, earthquake safety, and fire safety.
Table of Contents:

    Project Overview
    Features
    Setup Instructions
    Configuration
    Usage Guide
    Project Structure
    Testing
    Contributing

### Project Overview

The Emergency Preparedness Chatbot is an AI-powered chatbot designed to provide timely and relevant information to users preparing for or responding to various emergencies. It uses advanced machine learning models and a FAISS index to offer efficient, context-aware responses. The chatbot is developed using FastAPI (or Gradio) and can be hosted locally or on a cloud platform.

### Features and functionalities

    Contextual Awareness: Maintains conversation history to provide responses relevant to previous user queries.
    Efficient Information Retrieval: Uses FAISS indexing to quickly retrieve information from an emergency preparedness dataset.
    Conversational Interface: Access the chatbot through a REST API (FastAPI) or a user-friendly Gradio web interface.
    Robust Search: Supports similarity-based retrieval of information on topics like flood preparation, earthquake safety, and fire response.
     
### Setup Instructions

  1.Download the Repository in your computer
  
    Navigate to the main page of the repository on GitHub.
    Click on the Code button (usually green).
    Select Download ZIP.
    The repository will be downloaded as a .zip file that you can extract on your computer.
  2.Set Up a Virtual Environment
  
    python -m venv venv 
    source venv/bin/activate  # On Windows, use venv\Scripts\activate  
   note: the path you stored the files and the path you opening a virtual environment should be same

  3.Install Dependencies
  
    Install the required packages listed in requirements.txt: 
    pip install -r requirements.txt     

4. Download or Load the Dataset

       Ensure you have the emergency_preparedness_data.csv dataset with the Content and Filename columns. Place it in the project directory.

5. Generate and Save the FAISS Index
   
       If the FAISS index is not already created, run the following script:
       python create_embeddings.py

This will generate embeddings for the dataset and save the FAISS index as faiss_index.bin.
Configuration

Ensure the following configuration variables are set in your environment if needed:

    MODEL_PATH: Path to the language model (if fine-tuned or customized).
    API_KEY (optional): API key for any third-party integrations.

### Usage Guide
Running the Chatbot with FastAPI

  Start the FastAPI server:

    uvicorn main:app --reload
 Access the API documentation at http://127.0.0.1:8000/docs to interact with the chatbot using the /query/ endpoint.

 if OMP: Error #15:
 
    set KMP_DUPLICATE_LIB_OK=TRUE
    
Running the Chatbot with query

 Start the query interface

    python query_interface.py
    
 or else follow the link :- http://127.0.0.1:7860/
 A local web interface will open, where you can interact with the chatbot directly.

Example Queries

    User: "How do I prepare for an earthquake?"

### Project Structure

graphql

    emergency-preparedness-chatbot/
    ├── data/
    │   └── emergency_preparedness_data.csv   # Dataset file
    ├── embeddings/
    │   └── faiss_index.bin                   # Saved FAISS index
    ├── src/
    │   ├── main.py                           # FastAPI application for serving the chatbot
    │   ├── create_embeddings.py              # Script to generate embeddings and save FAISS index
    │   ├── retrieval.py                      # Script for retrieval logic
    │   ├── gradio_interface.py               # Gradio interface for interactive chatbot
    │   └── utils.py                          # Utility functions, including embedding generation
    ├── requirements.txt                      # List of dependencies
    └── README.md                             # Project documentation

### Testing
Manual Testing

    Use the FastAPI documentation at http://127.0.0.1:8000/docs to test API responses.
    For Gradio, enter queries directly in the interface to verify chatbot responses.

Automated Testing (Optional)

For more rigorous testing, consider writing tests using the pytest framework. Sample tests could include checking if responses are returned within a reasonable time and verifying that the chatbot maintains conversation context.
### Contributing

If you would like to contribute to this project, please follow these steps:

    Fork the repository.
    Create a new branch for your feature (git checkout -b feature-name).
    Commit your changes (git commit -m "Add feature").
    Push to the branch (git push origin feature-name).
    Open a Pull Request.

For deployment, consider hosting the chatbot on platforms like Heroku, AWS, or Google Cloud, which support FastAPI applications. Gradio also offers a quick deployment option using Gradio’s shareable links if you’re running the chatbot locally.
