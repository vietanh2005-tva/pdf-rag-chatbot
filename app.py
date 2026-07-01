import os
import streamlit as st

from rag_pipeline import (
    build_vector_database,
    retrieve_relevant_chunks,
    generate_answer
)


st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="💬",
    layout="wide"
)


st.title("💬 Chatbot chăm sóc khách hàng từ tài liệu PDF")
st.write(
    "Chatbot sử dụng RAG để trả lời câu hỏi dựa trên tài liệu chính sách mua hàng, đổi trả, hoàn tiền và giao hàng."
)


DATA_FOLDER = "data"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


st.sidebar.header("📄 Tài liệu PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload file PDF",
    type=["pdf"]
)


if uploaded_file is not None:
    pdf_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Đã upload: {uploaded_file.name}")

    if st.sidebar.button("Xử lý tài liệu"):
        with st.spinner("Đang đọc PDF, chia chunk, tạo embedding và lưu vào ChromaDB..."):
            total_chunks = build_vector_database(pdf_path)

        st.sidebar.success(f"Đã xử lý xong. Tổng số chunk: {total_chunks}")


st.divider()

question = st.text_input("Nhập câu hỏi của khách hàng:")

if st.button("Hỏi chatbot"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang tìm đoạn tài liệu liên quan..."):
            chunks = retrieve_relevant_chunks(question, top_k=3)

        if len(chunks) == 0:
            st.error("Chưa có dữ liệu trong ChromaDB. Hãy upload và xử lý PDF trước.")
        else:
            with st.spinner("Đang sinh câu trả lời bằng Local LLM..."):
                answer = generate_answer(question, chunks)

            st.subheader("✅ Câu trả lời")
            st.write(answer)

            st.subheader("📌 Nguồn tham khảo")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Nguồn {i + 1}: {chunk['source']} - Trang {chunk['page']}"):
                    st.write(chunk["text"])