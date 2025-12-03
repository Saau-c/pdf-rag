import streamlit as st
import tempfile
from utils import extract_text_from_pdf, simple_chunk_text

# RAG imports
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# --------------------------
# PAGE SETUP
# --------------------------
st.set_page_config(page_title="PDF Q&A (RAG) Demo", layout="centered")
st.title("📄 PDF Q&A — RAG Demo")

# --------------------------
# FILE UPLOADER
# --------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Load embedder once
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# OpenAI client (optional)
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None  # Will use fallback if key not available

# --------------------------
# MAIN WORKFLOW
# --------------------------
if uploaded_file is not None:
    # Save PDF temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tfile.write(uploaded_file.read())
    tfile.flush()
    path = tfile.name

    # Extract text
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(path)

    st.success("Text extracted!")

    # Preview text
    st.subheader("Preview of extracted text:")
    st.code(text[:1200] + ("..." if len(text) > 1200 else ""), language="text")

    # Chunk text
    chunks = simple_chunk_text(text, max_sentences=5, overlap_sentences=1)
    st.write(f"📌 Total chunks created: {len(chunks)}")

    # --------------------------
    # SHOW ALL CHUNKS CLEANLY
    # --------------------------
    st.subheader("Chunks preview:")
    for i, c in enumerate(chunks):
        if i < 5:
            # Highlight first 5 chunks
            with st.expander(f"Chunk {i+1} (highlighted)"):
                st.write(c)
        else:
            # Remaining chunks
            with st.expander(f"Chunk {i+1}"):
                st.write(c)

    # --------------------------
    # BUILD FAISS INDEX
    # --------------------------
    st.subheader("Building search index...")
    vectors = embedder.encode(chunks)
    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    st.success("Index created!")

    # --------------------------
    # QUESTION INPUT
    # --------------------------
    st.subheader("Ask a question about the PDF:")
    user_question = st.text_input("Enter your question here:")

    if user_question:
        # Embed question
        q_vec = embedder.encode([user_question]).astype("float32")

        # Search top chunks
        D, I = index.search(q_vec, k=3)
        retrieved = "\n\n".join(chunks[i] for i in I[0])

        # Try calling OpenAI if client available
        if client:
            try:
                prompt = f"""
You are an assistant. Answer based only on the PDF content below.
If the answer is not present, say you couldn't find it.

Context:
{retrieved}

Question:
{user_question}

Answer:
"""
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = completion.choices[0].message["content"]

                st.markdown("### ✅ Answer (LLM):")
                st.write(answer)

            except Exception as e:
                st.warning(f"⚠️ OpenAI API not available: {e}\nShowing retrieved chunks only.")
                answer = retrieved
                st.markdown("### ✅ Answer (retrieved chunks only):")
                st.write(answer)
        else:
            # No API key → fallback
            st.info("⚠️ OpenAI API key not found — showing retrieved chunks only.")
            answer = retrieved
            st.markdown("### ✅ Answer (retrieved chunks only):")
            st.write(answer)

        # Always allow user to see retrieved text
        with st.expander("Show retrieved PDF context"):
            st.write(retrieved)
