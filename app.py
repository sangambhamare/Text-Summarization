import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

# ------------------------------------------------------------
# App Title & Description
# ------------------------------------------------------------
st.set_page_config(page_title="Multilingual Document Summarizer", layout="centered")
st.title("Document Summarizer (English + Indian Languages)")
st.write("Upload a PDF, DOCX, or TXT file — or enter text directly. Works for English, Hindi, Marathi, and other Indian languages.")

# ------------------------------------------------------------
# Load Summarization Model (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer():
    # Multilingual model trained on 45+ languages including Indian ones
    return pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", tokenizer="csebuetnlp/mT5_multilingual_XLSum", device=-1)

summarizer = load_summarizer()

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def chunk_text(text, max_words=400):
    """Split text into chunks of roughly max_words words."""
    words = text.split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    text = ""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    elif filename.endswith(".docx"):
        document = Document(uploaded_file)
        for para in document.paragraphs:
            text += para.text + "\n"

    elif filename.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    return text.strip()

# ------------------------------------------------------------
# Input Section
# ------------------------------------------------------------
input_type = st.radio("Select Input Type:", ("Upload File", "Direct Text Input"))

text = ""

if input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        if text:
            st.success("Text extracted successfully.")
        else:
            st.warning("No readable text found in the document.")
else:
    text = st.text_area("Enter your text here:", height=250, placeholder="Type or paste text in English, Hindi, Marathi, etc.")

# ------------------------------------------------------------
# Summarization Section
# ------------------------------------------------------------
if st.button("Summarize"):
    if not text.strip():
        st.error("Please provide text or upload a valid file first.")
    else:
        original_word_count = len(text.split())
        st.write(f"Original Document Word Count: {original_word_count}")

        chunks = chunk_text(text)
        st.write(f"Total Chunks: {len(chunks)}")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            st.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
            try:
                summary = summarizer(chunk, max_length=128, min_length=30, do_sample=False)
                chunk_summaries.append(summary[0]['summary_text'])
            except Exception as e:
                st.warning(f"Chunk {i+1} failed: {e}")

        combined_summary_text = " ".join(chunk_summaries)
        st.subheader("Final Summary")
        st.write(combined_summary_text)

        final_word_count = len(combined_summary_text.split())
        st.write(f"Final Summary Word Count: {final_word_count}")

        st.download_button(
            label="Download Summary as TXT",
            data=combined_summary_text,
            file_name="summary.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.caption("Developed by Sangam Sanjay Bhamare, 2025")
