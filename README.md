# Document Summarizer

[Click here for Live Demo](https://text-summarization-6zhc3zpt83i9z6ivib9crj.streamlit.app/)

Document Summarizer is a Streamlit-based web application that uses Facebook's BART model (`facebook/bart-large-cnn`) for summarizing documents. It supports PDF, DOCX, and TXT file formats, as well as direct text input. The app automatically breaks long documents into manageable chunks, summarizes each chunk, and produces one final consolidated summary. It also displays the word counts of the original document and the final summary to highlight the reduction in text.

## Features

- **Multi-format Support:**  
  Upload PDF, DOCX, or TXT files, or enter text directly.

- **Automatic Chunking:**  
  Automatically splits large texts into manageable chunks (around 500 words per chunk) to meet model input limitations.

- **Summarization with Transformers:**  
  Uses the `facebook/bart-large-cnn` model from Hugging Face to generate summaries.

- **Processing Feedback:**  
  Displays the current chunk being processed without showing individual chunk summaries or the full extracted text.

- **Word Count Comparison:**  
  Displays the original document's word count and the final summary's word count to demonstrate text reduction.

- **Clean UI:**  
  The input options are available on the main screen for easy access.


