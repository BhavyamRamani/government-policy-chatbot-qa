# qa_system.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage

GEMINI_API_KEY = "["your api key"]"

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def retrieve_relevant_docs(question, k=3):
    vector_db = load_vector_store()
    docs = vector_db.similarity_search(question, k=k)
    return [doc.page_content for doc in docs] if docs else []

def generate_answer(question, context_docs):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

    context_text = "\n\n".join(context_docs) if context_docs else "No relevant info."

    messages = [
        SystemMessage(content="You are a helpful assistant answering questions about government policies."),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)
    return response.content
