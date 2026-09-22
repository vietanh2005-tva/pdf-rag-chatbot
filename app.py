from pathlib import Path

import streamlit as st

from rag_pipeline import (
    OLLAMA_MODEL_NAME,
    TOP_K,
    build_vector_database,
    generate_answer,
    retrieve_relevant_chunks,
)

st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Chatbot chăm sóc khách hàng từ tài liệu PDF")
st.caption(
    "RAG prototype chạy local: câu trả lời được tạo từ các đoạn tài liệu truy xuất và hiển thị nguồn để kiểm tra."
)

data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

with st.sidebar:
    st.header("📄 Tài liệu PDF")
    st.caption(f"Local LLM: {OLLAMA_MODEL_NAME} · Top-k: {TOP_K}")
    uploaded_file = st.file_uploader("Upload file PDF", type=["pdf"])

    if uploaded_file is not None:
        safe_name = Path(uploaded_file.name).name
        pdf_path = data_folder / safe_name
        pdf_path.write_bytes(uploaded_file.getbuffer())
        st.success(f"Đã upload: {safe_name}")

        if st.button("Xử lý tài liệu", type="primary"):
            try:
                with st.spinner("Đang đọc PDF, chia chunk và tạo embedding..."):
                    total_chunks = build_vector_database(pdf_path)
                st.success(f"Đã index {total_chunks} chunks.")
            except Exception as exc:
                st.error(f"Không thể xử lý tài liệu: {exc}")

st.divider()
question = st.text_input(
    "Nhập câu hỏi của khách hàng",
    placeholder="Ví dụ: Tôi có bao nhiêu ngày để trả hàng?",
)

if st.button("Hỏi chatbot"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang tìm các đoạn tài liệu liên quan..."):
            chunks = retrieve_relevant_chunks(question)

        if not chunks:
            st.error("Chưa có dữ liệu. Hãy upload và xử lý một PDF trước.")
        else:
            with st.spinner("Đang tạo câu trả lời bằng Local LLM..."):
                answer = generate_answer(question, chunks)

            st.subheader("Câu trả lời")
            st.write(answer)
            st.subheader("Nguồn tham khảo")
            for index, chunk in enumerate(chunks, start=1):
                label = f"Nguồn {index}: {chunk['source']} - Trang {chunk['page']}"
                with st.expander(label):
                    st.write(chunk["text"])
                    st.caption(f"Retrieval distance: {chunk['distance']:.4f}")
