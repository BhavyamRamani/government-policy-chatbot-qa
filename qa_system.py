# qa_system.py
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

# ✅ Use HuggingFace for embeddings (local, free)
def load_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def retrieve_relevant_docs(question, k=3):
    vector_db = load_vector_store()
    docs = vector_db.similarity_search(question, k=k)
    return [doc.page_content for doc in docs] if docs else []

# ✅ Still okay to use Gemini for answering (not embeddings)
GEMINI_API_KEY = ""

def generate_answer(question, context_docs):
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

    context_text = "\n\n".join(context_docs) if context_docs else "No relevant info."

    messages = [
        SystemMessage(content="You are a helpful assistant answering questions about government policies."),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)
    return response.content
