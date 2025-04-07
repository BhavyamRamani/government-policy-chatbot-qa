import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set your Gemini API key
GEMINI_API_KEY = "AIzaSyCd-IfRoqVwcnegHUb3UdMhODStOEC_ZMQ"

def load_pdfs(paths):
    """Load and extract text from list of PDF file paths."""
    texts = []
    for path in paths:
        try:
            with pdfplumber.open(path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                if text:
                    texts.append(text)
        except Exception as e:
            print(f"⚠️ Skipped {path} due to error: {e}")
    return texts

def process_and_add_pdfs(pdf_paths):
    """Extract, embed, and add new docs to FAISS index."""
    texts = load_pdfs(pdf_paths)
    if not texts:
        print("⚠️ No valid texts found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(texts)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # If index already exists, update it
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local("faiss_index")
    print("✅ New documents embedded and saved to FAISS.")
