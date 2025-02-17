import streamlit as st
from transformers import pipeline
import PyPDF2
from docx import Document

st.title("Document Summarizer")
st.write("Upload a PDF, DOCX, or TXT file, or enter text directly below.")

# Cache the summarizer to speed up subsequent runs.
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

def chunk_text(text, max_words=500):
    """
    Splits the text into chunks of roughly max_words words.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_text_from_file(uploaded_file):
    """
    Extracts text from an uploaded file.
    Supports PDF, DOCX, and TXT files.
    """
    text = ""
    filename = uploaded_file.name
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
    return text

# Input type selection on the main screen
input_type = st.radio("Select Input Type:", ("Upload File", "Direct Text Input"))

if input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        text = extract_text_from_file(uploaded_file)
    else:
        text = ""
else:
    text = st.text_area("Enter Text", height=300)

if st.button("Summarize"):
    if not text.strip():
        st.error("No text provided for summarization!")
    else:
        # Split the text into manageable chunks
        chunks = chunk_text(text, max_words=500)
        st.write(f"Total chunks to summarize: {len(chunks)}")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            st.write(f"**Processing chunk {i+1} of {len(chunks)}...**")
            # For the last chunk, preserve all remaining tokens.
            if i == len(chunks) - 1:
                tokens = summarizer.tokenizer(chunk, return_tensors="pt", truncation=False).input_ids[0]
                input_token_length = len(tokens)
                if input_token_length < 130:
                    chunk_summary = chunk
                else:
                    summary = summarizer(
                        chunk,
                        max_length=input_token_length,
                        min_length=30,
                        do_sample=False,
                        truncation=False
                    )
                    chunk_summary = summary[0]['summary_text']
            else:
                summary = summarizer(
                    chunk,
                    max_length=130,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                chunk_summary = summary[0]['summary_text']
            chunk_summaries.append(chunk_summary)
        
        # Combine all chunk summaries into one text
        combined_summary_text = " ".join(chunk_summaries)
        final_summary = summarizer(
            combined_summary_text,
            max_length=130,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        
        st.subheader("Final Summary")
        st.write(final_summary[0]['summary_text'])
        
        st.markdown("---")
        st.markdown("Developed by Sangam Sanjay Bhamare 2025")
