# PDF RAG Chatbot

This project is a local PDF question-answering chatbot using Retrieval-Augmented Generation (RAG).  
The chatbot allows users to ask questions based on the content of a PDF document and returns answers grounded in the retrieved document context.

## Features

- Read and extract text from PDF documents
- Clean and split text into smaller chunks
- Generate embeddings from text chunks
- Store and search vectors using ChromaDB
- Retrieve relevant chunks based on the user question
- Generate answers using a local LLM
- Simple web interface built with Streamlit

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama / Local LLM
- PyMuPDF

## Project Pipeline

PDF Document  
→ Text Extraction  
→ Text Cleaning  
→ Chunking  
→ Embedding Generation  
→ Vector Storage with ChromaDB  
→ Similarity Search  
→ Local LLM Answer Generation  
→ User Response

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Or run with the batch file on Windows:

```bash
run_app.bat
```

## Purpose

The purpose of this project is to build a customer-support chatbot that can answer questions from a specific PDF document while reducing hallucination by grounding the response in retrieved document content.
