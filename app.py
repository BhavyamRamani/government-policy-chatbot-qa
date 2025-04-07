import streamlit as st
from tempfile import NamedTemporaryFile

from preprocess_data import process_and_add_pdfs
from qa_system import retrieve_relevant_docs, generate_answer

st.set_page_config(page_title="📜 Government Policy QA", layout="centered")
st.title("📜 Government Policy QA")

# Initialize session state for tracking whether embedding has already been done
if "embedded" not in st.session_state:
    st.session_state.embedded = False

uploaded_files = st.file_uploader("📤 Upload PDF files", type="pdf", accept_multiple_files=True)

# Only embed if new files are uploaded and embedding hasn't been done yet
if uploaded_files and not st.session_state.embedded:
    temp_paths = []
    for file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_paths.append(tmp.name)

    with st.spinner("🔄 Embedding documents..."):
        process_and_add_pdfs(temp_paths)

    st.session_state.embedded = True
    st.success("✅ Uploaded and embedded successfully!")

question = st.text_input("❓ Ask your question")

if question:
    with st.spinner("💬 Generating answer..."):
        docs = retrieve_relevant_docs(question)
        answer = generate_answer(question, docs)

    st.markdown("### 💡 Answer:")
    st.write(answer)

    if docs:
        with st.expander("📄 Retrieved Documents"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Document {i}:**")
                st.write(doc)
                st.markdown("---")
