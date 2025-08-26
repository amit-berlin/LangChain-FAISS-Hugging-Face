import streamlit as st
import os

# Try imports of heavy libraries
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline
    heavy_ok = True
except Exception as e:
    heavy_ok = False

# Always available light libs
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("punkt", quiet=True)

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="AI Solution Designer MVP", layout="wide")
st.title("🧠 AI Solution Designer & Implementor (MVP)")

st.markdown("""
This MVP demonstrates **enterprise AI pipeline (20 steps)**.  
If advanced AI libraries are unavailable, a **human-like fallback Agent** takes over with NLTK + PyMuPDF.
""")

# ---------------- 20 Backend Steps ----------------
steps = [
    "1. User uploads enterprise documents into the AI platform.",
    "2. System ingests and preprocesses text (cleaning, normalization).",
    "3. Documents are split into chunks for better contextual retrieval.",
    "4. Each chunk is converted into embeddings (if advanced mode).",
    "5. Embeddings are stored in FAISS vector DB (if available).",
    "6. Retrieval pipeline fetches top-k chunks (advanced) or keyword search (fallback).",
    "7. Load a Generative AI model (FLAN-T5 small in advanced mode).",
    "8. LangChain orchestrates retrieval + generation (advanced mode).",
    "9. User submits a query from Streamlit frontend.",
    "10. Query converted to embeddings (advanced) or keywords (fallback).",
    "11. Compare query with documents using FAISS or simple match.",
    "12. Retrieve most relevant context.",
    "13. RAG process runs (advanced) or heuristic response (fallback).",
    "14. AI Agent decides how to answer based on available tools.",
    "15. Multi-agent orchestration possible in future (LangGraph).",
    "16. Answer formatted and served via frontend.",
    "17. Logs/metrics tracked (MLflow in enterprise setup).",
    "18. CI/CD automation possible (GitHub Actions).",
    "19. Scalable to Kubernetes + GCP Vertex AI (enterprise).",
    "20. Final output delivered to user."
]

with st.expander("📜 Show 20 Backend Steps"):
    for s in steps:
        st.markdown(f"- {s}")

# ---------------- Document Upload ----------------
st.header("Step 1: Choose a Document")
use_demo = st.checkbox("Use Sample Demo PDF")
uploaded_files = st.file_uploader("Or upload your own .txt or .pdf", type=["txt", "pdf"], accept_multiple_files=True)

docs = []
if use_demo:
    demo_path = os.path.join("demo_docs", "sample.pdf")
    if os.path.exists(demo_path):
        with open(demo_path, "r", errors="ignore") as f:
            docs.append(f.read())
        st.success("✅ Loaded demo PDF text")
    else:
        st.error("Demo PDF missing. Please upload instead.")
elif uploaded_files:
    for f in uploaded_files:
        if f.type == "application/pdf":
            pdf = fitz.open(stream=f.read(), filetype="pdf")
            text = ""
            for page in pdf:
                text += page.get_text()
            docs.append(text)
        else:
            docs.append(f.read().decode("utf-8"))
    st.success("✅ Documents uploaded")

# ---------------- Advanced Pipeline (if available) ----------------
if docs and heavy_ok:
    st.header("Step 2: Advanced AI Mode (LangChain + FAISS + HuggingFace)")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_texts(docs, embeddings)
        retriever = vector_db.as_retriever()
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=200)
        llm = HuggingFacePipeline(pipeline=qa_pipeline)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask your question here:")
        if query:
            with st.spinner("AI thinking..."):
                answer = qa_chain.run(query)
            st.success("✅ Advanced AI Answer ready")
            st.markdown(f"### 🤖 Answer: {answer}")
    except Exception as e:
        st.error("Advanced AI pipeline failed, switching to fallback agent.")

# ---------------- Fallback Agent ----------------
if docs and not heavy_ok:
    st.header("Step 2: Fallback Agent Mode (NLTK + Keyword Search)")
    st.warning("⚠️ Advanced AI libraries are busy. Using lightweight human-like Agent instead.")

    query = st.text_input("Ask your question here:")
    if query:
        text = " ".join(docs)
        sentences = sent_tokenize(text)
        tokens = word_tokenize(query.lower())
        # Simple keyword-based relevance
        matched = [s for s in sentences if any(t in s.lower() for t in tokens)]
        answer = " ".join(matched[:3]) if matched else "I could not find relevant info."
        st.success("✅ Fallback Agent Answer ready")
        st.markdown(f"### 🧑‍💻 Human-like Agent says: {answer}")
