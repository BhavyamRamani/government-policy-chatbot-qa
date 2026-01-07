# Government Policy QA Chatbot

An AI-powered chatbot built with Google Gemini, LangChain, FAISS, and Streamlit to answer questions about government documents like policies, budgets, and reports. Users can upload PDFs and get contextual, natural language answers with transparency on the document source.

## Features

-  Upload and process multiple PDF documents
-  Semantic search using local HuggingFace sentence embeddings with FAISS
-  Context-aware answers generated using Google Gemini (`gemini-2.5-flash`)
-  Retrieves and uses relevant document chunks for grounded responses
-  Simple and interactive Streamlit-based interface

---

## 📸 Demo
![Screenshot 2025-04-07 at 5 43 43 PM](https://github.com/user-attachments/assets/9e27fa7a-8879-4bae-97c9-c2ee95a6b6fb)
![Screenshot 2025-04-07 at 1 48 12 PM](https://github.com/user-attachments/assets/f3553802-a9a3-45ab-a426-1500def3abcf)
![Screenshot 2025-04-07 at 5 28 24 PM](https://github.com/user-attachments/assets/b3a3d56c-303b-4c14-8674-7071d34554ca)




---

## 🧰 Tech Stack

| Component        | Tech Used                                 |
|------------------|------------------------------------       |
| Embeddings       | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store     | FAISS                                     |
| LLM              | Google Gemini (via LangChain)             |
| Frontend         | Streamlit                                 |
| PDF Parsing      | pdfplumber                                |
| Text Splitting   | LangChain RecursiveCharacterTextSplitter  |
| Orchestration    | LangChain                                 |

---

## ⚙️ How It Works
1. PDF documents are uploaded through the Streamlit interface.
2. Text is extracted from PDFs and split into manageable chunks.
3. Document chunks are embedded locally using SentenceTransformer embeddings.
4. Embeddings are stored and updated in a FAISS vector index.
5. User questions are embedded and matched against the vector store.
6. Relevant document chunks are passed as context to Gemini for answer generation.

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/BhavyamRamani/government-policy-qa.git
cd government-policy-qa
pip install -r requirements.txt
streamlit run app.py

---

## Project Structure
```
├── app.py                  # Streamlit frontend
├── qa_system.py            # Handles retrieval and generation
├── preprocess_data.py      # PDF processing and embedding
├── vectorstore/            # FAISS index saved here
├── requirements.txt
└── README.md
```

---
## Acknowledgements
- LangChain  
- SentenceTransformers  
- Google Generative AI  
- Streamlit  
- FAISS
