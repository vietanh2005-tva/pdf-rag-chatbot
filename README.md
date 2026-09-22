# Chatbot RAG hỗ trợ khách hàng bằng tiếng Việt

Đây là nguyên mẫu Retrieval-Augmented Generation (RAG) ưu tiên chạy cục bộ, dùng để trả lời các câu hỏi hỗ trợ khách hàng bằng tiếng Việt dựa trên tài liệu chính sách PDF do người dùng tải lên. Ứng dụng truy xuất những đoạn nội dung liên quan, tạo câu trả lời bám sát tài liệu bằng mô hình Ollama chạy cục bộ, đồng thời hiển thị tài liệu nguồn và số trang để người dùng kiểm chứng.

## Tính năng nổi bật

- Tải lên và lập chỉ mục tài liệu PDF ngay trên giao diện Streamlit.
- Trích xuất văn bản theo từng trang và chia thành các đoạn có phần nội dung chồng lấn.
- Tạo embedding đa ngôn ngữ bằng `paraphrase-multilingual-MiniLM-L12-v2`.
- Lưu trữ và truy xuất vector bằng ChromaDB với dữ liệu được duy trì lâu dài.
- Tạo câu trả lời tiếng Việt cục bộ bằng Ollama và Qwen 2.5.
- Hiển thị các đoạn được truy xuất, tên tệp, số trang và khoảng cách truy xuất.
- Từ chối trả lời khi tài liệu không cung cấp đủ thông tin thông qua system prompt bám sát nguồn.
- Đi kèm bộ 15 câu hỏi để đánh giá khả năng truy xuất.

## Kiến trúc

```text
Tải PDF lên
   |
   v
Trích xuất văn bản -> Làm sạch -> Chia đoạn -> Embedding đa ngôn ngữ
                                                       |
                                                       v
                                                   ChromaDB
                                                       |
Câu hỏi -> Embedding câu hỏi -> Truy xuất top-k
                                                       |
                                                       v
                                      Ngữ cảnh + prompt bám sát nguồn
                                                       |
                                                       v
                                          Mô hình Ollama cục bộ
                                                       |
                                                       v
                                  Câu trả lời + trang nguồn để kiểm chứng
```

## Công nghệ sử dụng

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama / Qwen 2.5
- pypdf

## Cài đặt và chạy ứng dụng

1. Cài đặt Python 3.10+ và [Ollama](https://ollama.com/).
2. Tạo và kích hoạt môi trường ảo.
3. Cài đặt các thư viện bằng lệnh `pip install -r requirements.txt`.
4. Tải mô hình bằng lệnh `ollama pull qwen2.5:1.5b`.
5. Sao chép `.env.example` thành `.env` và điều chỉnh các giá trị nếu cần.
6. Khởi động ứng dụng bằng lệnh `streamlit run app.py`.
7. Tải lên một tệp PDF có thể trích xuất văn bản, chọn **Xử lý tài liệu**, sau đó nhập câu hỏi.


## Quyền riêng tư và vệ sinh repository

- Các tệp PDF được tải lên, cơ sở dữ liệu vector cục bộ, tệp `.env` và môi trường ảo đều được loại trừ trong `.gitignore`.
- Không commit chính sách nội bộ, thông tin khách hàng, thông tin xác thực hoặc tài liệu có bản quyền khi chưa được phép.




