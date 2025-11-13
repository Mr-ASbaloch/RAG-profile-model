import os
import pickle
import faiss
import gdown
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # ✅ correct import
from groq import Groq

# ==========================================
# CONFIGURATION
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")  # 🔑 Replace manually if not set
PDF_DRIVE_LINK = "https://drive.google.com/uc?id=1wQiWbNi0xI03A_TXGz_4-QimlUhYIZiN"
PDF_FILE = "document.pdf"
VECTOR_DB_PATH = "vector_store.pkl"

client = Groq(api_key=GROQ_API_KEY)


# ==========================================
# FUNCTIONS
# ==========================================
def download_pdf():
    """Download the PDF from Google Drive if not already present."""
    if not os.path.exists(PDF_FILE):
        st.info("📥 Downloading PDF from Google Drive...")
        gdown.download(PDF_DRIVE_LINK, PDF_FILE, quiet=False)
    else:
        st.success("✅ PDF already exists, skipping download.")


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def create_text_chunks(text, chunk_size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def create_embeddings(chunks):
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.embed_documents(chunks)
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    with open(VECTOR_DB_PATH, "wb") as f:
        pickle.dump((index, chunks, model), f)
    st.success("✅ Vector store created and saved.")


def load_or_create_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        st.info("🔁 Loading existing vector store...")
        return
    st.info("⚙️ Processing new PDF...")
    download_pdf()
    text = extract_text_from_pdf(PDF_FILE)
    chunks = create_text_chunks(text)
    create_embeddings(chunks)


def retrieve_context(query, k=3, similarity_threshold=0.6):
    with open(VECTOR_DB_PATH, "rb") as f:
        index, chunks, model = pickle.load(f)

    q_vec = model.embed_query(query)
    D, I = index.search(np.array([q_vec]).astype("float32"), k)

    similarities = 1 / (1 + D[0])  # Convert distances to similarity
    avg_sim = float(np.mean(similarities))

    if avg_sim < similarity_threshold:
        return None, avg_sim

    retrieved_text = "\n".join([chunks[i] for i in I[0]])
    return retrieved_text, avg_sim


def ask_groq(query):
    context, score = retrieve_context(query)
    if not context or score < 0.6:
        return f"⚠️ Sorry, this query is not related to the knowledge base. (Relevance: {round(score*100,1)}%)"

    prompt = f"Use the following document context to answer accurately:\n\n{context}\n\nQuestion: {query}"
    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="RAG Chat — Groq + FAISS",
    page_icon="💬",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#ff4ecd;'>💖 RAG Chat — Strict Knowledge Base Only</h1>
    <p style='text-align:center;'>Ask questions strictly based on the uploaded PDF file — no external data 🌍</p>
    """,
    unsafe_allow_html=True
)

# Automatically prepare vector store
load_or_create_vector_store()

query = st.text_input("💬 Ask your question:")
if st.button("🔮 Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_groq(query)
        st.text_area("🧠 Answer", answer, height=200)
