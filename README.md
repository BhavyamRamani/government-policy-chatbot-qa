# 📜 Government Policy QA Chatbot

An AI-powered chatbot built with Google Gemini, LangChain, FAISS, and Streamlit to answer questions about government documents like policies, budgets, and reports. Users can upload PDFs and get contextual, natural language answers with transparency on the document source.

## 🧠 Features

- 📤 Upload multiple PDF documents
- 🔍 Semantic search using Google `embedding-001` + FAISS
- 💬 Answers generated by Gemini models (e.g., `gemini-1.5-pro-latest`)
- 📄 Shows source documents retrieved
- 🚀 Simple and intuitive Streamlit interface

---

## 📸 Demo
![Screenshot 2025-04-07 at 5 43 43 PM](https://github.com/user-attachments/assets/9e27fa7a-8879-4bae-97c9-c2ee95a6b6fb)
![Screenshot 2025-04-07 at 1 48 12 PM](https://github.com/user-attachments/assets/f3553802-a9a3-45ab-a426-1500def3abcf)
![Screenshot 2025-04-07 at 5 28 24 PM](https://github.com/user-attachments/assets/b3a3d56c-303b-4c14-8674-7071d34554ca)




---

## 🧰 Tech Stack

| Component        | Tech Used                          |
|------------------|------------------------------------|
| Embeddings       | Google Generative AI `embedding-001` |
| Vector Store     | FAISS                              |
| LLM              | Google Gemini (via LangChain)      |
| Frontend         | Streamlit                          |
| PDF Parsing      | PyMuPDF (fitz)                     |
| Orchestration    | LangChain                          |

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/BhavyamRamani/government-policy-qa.git
cd government-policy-qa
pip install -r requirements.txt
streamlit run app.py


├── app.py                  # Streamlit frontend
├── qa_system.py            # Handles retrieval and generation
├── preprocess_data.py      # PDF processing and embedding
├── vectorstore/            # FAISS index saved here
├── requirements.txt
└── README.md

🙌 Acknowledgements
LangChain

Google Generative AI

Streamlit

FAISS

