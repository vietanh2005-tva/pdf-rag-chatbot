import os
import re
import requests
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================
# CẤU HÌNH
# =========================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "customer_support_docs"

PDF_PATH = "data/massimo_dutti_dieu_khoan_mua_hang.pdf"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL_NAME = "qwen2.5:1.5b"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# =========================
# 1. ĐỌC PDF
# =========================

def extract_text_from_pdf(pdf_path):
    """
    Đọc file PDF và trích xuất text theo từng trang.
    Trả về list gồm page và text.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages_text.append({
                "page": page_index + 1,
                "text": text
            })

    return pages_text


# =========================
# 2. LÀM SẠCH TEXT
# =========================

def clean_text(text):
    """
    Làm sạch text sau khi lấy từ PDF.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# =========================
# 3. CHIA CHUNK
# =========================

def split_text_into_chunks(pages_text, chunk_size=180, overlap=40):
    """
    Chia text thành chunk theo số từ.
    Mỗi chunk khoảng 180 từ, overlap 40 từ để giữ ngữ cảnh.
    """
    chunks = []

    for item in pages_text:
        page = item["page"]
        text = clean_text(item["text"])
        words = text.split()

        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "text": chunk_text,
                    "page": page
                })

            start = end - overlap

    return chunks


# =========================
# 4. KẾT NỐI CHROMADB
# =========================

def get_chroma_collection():
    """
    Tạo hoặc kết nối tới collection trong ChromaDB.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


# =========================
# 5. TẠO VECTOR DATABASE
# =========================

def build_vector_database(pdf_path):
    """
    Đọc PDF, chia chunk, tạo embedding và lưu vào ChromaDB.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pages)

    collection = get_chroma_collection()

    # Xóa dữ liệu cũ để tránh bị lẫn chunk cũ
    existing = collection.get()
    if existing and len(existing["ids"]) > 0:
        collection.delete(ids=existing["ids"])

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "page": chunk["page"],
            "source": os.path.basename(pdf_path)
        })

        embedding = embedding_model.encode(chunk["text"]).tolist()
        embeddings.append(embedding)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(chunks)


# =========================
# 6. TRUY XUẤT CHUNK LIÊN QUAN
# =========================

def retrieve_relevant_chunks(question, top_k=3):
    """
    Nhận câu hỏi của người dùng,
    tạo embedding cho câu hỏi,
    rồi tìm các chunk liên quan nhất trong ChromaDB.
    """
    collection = get_chroma_collection()

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({
                "text": doc,
                "page": meta["page"],
                "source": meta["source"]
            })

    return retrieved_chunks


# =========================
# 7. GỌI OLLAMA
# =========================

def call_ollama(prompt, model=OLLAMA_MODEL_NAME):
    """
    Gọi Local LLM thông qua Ollama.
    """
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)

        if response.status_code != 200:
            return "Không gọi được Ollama. Hãy kiểm tra Ollama đã chạy chưa."

        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return "Không kết nối được Ollama. Hãy mở Ollama hoặc chạy model bằng lệnh: ollama run qwen2.5:1.5b"

    except requests.exceptions.Timeout:
        return "Ollama phản hồi quá lâu. Hãy thử lại hoặc dùng model nhẹ hơn."


# =========================
# 8. SINH CÂU TRẢ LỜI
# =========================

def generate_answer(question, retrieved_chunks):
    """
    Tạo prompt từ câu hỏi và các chunk tìm được,
    sau đó dùng Local LLM để sinh câu trả lời.
    """
    context = ""

    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[Đoạn {i + 1} - Trang {chunk['page']} - File {chunk['source']}]\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""
Bạn là chatbot chăm sóc khách hàng của một website bán hàng.

Nhiệm vụ của bạn:
- Trả lời bằng tiếng Việt.
- Chỉ trả lời dựa trên nội dung tài liệu được cung cấp.
- Không tự bịa thông tin ngoài tài liệu.
- Nếu tài liệu không có thông tin, hãy nói: "Tài liệu chưa cung cấp đủ thông tin để trả lời câu hỏi này."
- Trả lời ngắn gọn, rõ ràng, lịch sự như nhân viên chăm sóc khách hàng.

Tài liệu liên quan:
{context}

Câu hỏi của khách hàng:
{question}

Câu trả lời:
"""

    answer = call_ollama(prompt)
    return answer


# =========================
# 9. CHẠY TEST
# =========================

if __name__ == "__main__":
    print("Đang xây dựng vector database từ PDF...")

    total_chunks = build_vector_database(PDF_PATH)

    print("Đã tạo embedding và lưu vào ChromaDB")
    print("Tổng số chunk đã lưu:", total_chunks)

    print("=" * 80)

    question = "Khách hàng có bao nhiêu ngày để trả hàng?"

    chunks = retrieve_relevant_chunks(question, top_k=3)

    answer = generate_answer(question, chunks)

    print("Câu hỏi:", question)
    print("=" * 80)
    print("Câu trả lời:")
    print(answer)

    print("=" * 80)
    print("Nguồn tham khảo:")
    for i, chunk in enumerate(chunks):
        print(f"Nguồn {i + 1}: {chunk['source']} - Trang {chunk['page']}")