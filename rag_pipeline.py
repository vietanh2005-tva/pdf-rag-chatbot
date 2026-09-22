import os
import re
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "customer_support_docs")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("TOP_K", "3"))

_embedding_model = None


def get_embedding_model():
    """Load the embedding model lazily so startup and tests stay lightweight."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def extract_text_from_pdf(pdf_path):
    """Extract non-empty text from each page of a PDF."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append({"page": page_index + 1, "text": text})
    return pages_text


def clean_text(text):
    """Normalize whitespace in extracted document text."""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def split_text_into_chunks(pages_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split page text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    for item in pages_text:
        words = clean_text(item["text"]).split()
        start = 0
        while start < len(words):
            chunk_text = " ".join(words[start : start + chunk_size])
            if len(chunk_text) > 50:
                chunks.append({"text": chunk_text, "page": item["page"]})
            start += chunk_size - overlap
    return chunks


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_vector_database(pdf_path, clear_existing=True):
    """Index one PDF and return the number of chunks stored."""
    pages = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pages)
    if not chunks:
        raise ValueError("Không trích xuất được nội dung văn bản từ PDF.")

    collection = get_chroma_collection()
    if clear_existing:
        existing = collection.get(include=[])
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])

    source = Path(pdf_path).name
    model = get_embedding_model()
    documents = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(documents).tolist()
    ids = [f"{Path(source).stem}_chunk_{index}" for index in range(len(chunks))]
    metadatas = [{"page": chunk["page"], "source": source} for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


def retrieve_relevant_chunks(question, top_k=TOP_K):
    """Retrieve the most relevant chunks and include distance for inspection."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []

    question_embedding = get_embedding_model().encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": document,
                "page": metadata["page"],
                "source": metadata["source"],
                "distance": distance,
            }
        )
    return chunks


def call_ollama(prompt, model=OLLAMA_MODEL_NAME):
    """Generate a response with a locally running Ollama model."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return (
            "Không kết nối được Ollama. Hãy mở Ollama và chạy model: "
            f"ollama run {model}"
        )
    except requests.exceptions.Timeout:
        return "Ollama phản hồi quá lâu. Hãy thử lại hoặc dùng model nhẹ hơn."
    except requests.exceptions.RequestException as exc:
        return f"Ollama trả về lỗi: {exc}"


def generate_answer(question, retrieved_chunks):
    """Generate a grounded Vietnamese answer from retrieved context."""
    context_parts = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Đoạn {index} - Trang {chunk['page']} - File {chunk['source']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
Bạn là chatbot chăm sóc khách hàng của một website bán hàng.

Yêu cầu:
- Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng và lịch sự.
- Chỉ sử dụng thông tin trong tài liệu liên quan bên dưới.
- Không suy đoán hoặc bổ sung kiến thức ngoài tài liệu.
- Nếu tài liệu không đủ thông tin, trả lời chính xác: "Tài liệu chưa cung cấp đủ thông tin để trả lời câu hỏi này."

Tài liệu liên quan:
{context}

Câu hỏi của khách hàng:
{question}

Câu trả lời:
""".strip()
    return call_ollama(prompt)


if __name__ == "__main__":
    default_pdf = BASE_DIR / "data" / "massimo_dutti_dieu_khoan_mua_hang.pdf"
    if not default_pdf.exists():
        raise SystemExit("Không tìm thấy PDF mẫu trong thư mục data.")
    total_chunks = build_vector_database(default_pdf)
    print(f"Đã index {total_chunks} chunks từ {default_pdf.name}.")
