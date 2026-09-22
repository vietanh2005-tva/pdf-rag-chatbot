# Vietnamese Customer Support RAG Chatbot

A local-first Retrieval-Augmented Generation (RAG) prototype that answers Vietnamese customer-support questions from uploaded PDF policies. The application retrieves relevant passages, generates a grounded response with a local Ollama model, and shows the source document and page for human verification.

## Highlights

- Upload and index PDF documents from a Streamlit interface.
- Extract text by page and split it into overlapping chunks.
- Create multilingual embeddings with `paraphrase-multilingual-MiniLM-L12-v2`.
- Store and retrieve vectors with persistent ChromaDB.
- Generate Vietnamese answers locally with Ollama and Qwen 2.5.
- Display retrieved passages, file names, page numbers, and retrieval distance.
- Refuse unsupported answers through a grounded system prompt.
- Include a 15-question retrieval evaluation set.

## Architecture

```text
PDF upload
   |
   v
Text extraction -> Cleaning -> Chunking -> Multilingual embeddings
                                                |
                                                v
                                            ChromaDB
                                                |
User question -> Question embedding -> Top-k retrieval
                                                |
                                                v
                                  Context + grounded prompt
                                                |
                                                v
                                      Local Ollama model
                                                |
                                                v
                              Answer + source pages for review
```

## Technology

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama / Qwen 2.5
- pypdf

## Setup

1. Install Python 3.10+ and [Ollama](https://ollama.com/).
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Download the model with `ollama pull qwen2.5:1.5b`.
5. Copy `.env.example` to `.env` and adjust values if needed.
6. Start the app with `streamlit run app.py`.
7. Upload a text-based PDF, select **Xử lý tài liệu**, and ask a question.

## Evaluation

The repository includes `evaluation/questions.json` with 15 Vietnamese questions. Run the lightweight retrieval check against your own non-sensitive PDF:

```bash
python evaluate_retrieval.py --pdf path/to/policy.pdf
```

The script reports whether expected keywords appear in the retrieved passages. The out-of-scope case is intentionally marked for manual review. This is a starter evaluation, not a complete measure of answer correctness.

## Privacy and repository hygiene

- Uploaded PDFs, the local vector database, `.env`, and virtual environments are excluded by `.gitignore`.
- Do not commit internal policies, customer information, credentials, or copyrighted documents without permission.
- Use synthetic or publicly licensed documents for a public demo.

## Current scope and limitations

This is a portfolio prototype, not a production customer-support system. It currently indexes one active document set at a time and does not include authentication, access control, OCR for scanned PDFs, automated safety moderation, production monitoring, or comprehensive RAG evaluation.

## Suggested portfolio demo

Add one screenshot or short GIF showing the uploaded document, the generated answer, and the expanded source passage. Use a synthetic/public document so the repository remains safe to share.
