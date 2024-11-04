import gradio as gr
import requests

def chatbot_interface(user_input):
    response = requests.post("http://127.0.0.1:8000/query/", json={"text": user_input}).json()
    return response["response"]

gr.Interface(fn=chatbot_interface, inputs="text", outputs="text", title="RAG-Based Chatbot",).launch(share=True)
