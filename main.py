# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import retrieve_similar

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/query/")
async def query_chatbot(query: Query):
    results = retrieve_similar(query.text)
    response = " ".join(results)
    return {"response": response}

# To run the API, execute this file: uvicorn main:app --reload
