# Đóng góp của Nguyễn Ngọc Hoàng - UI, Docker và demo

## 1. Phạm vi phụ trách

- Giải thích giao diện Gradio và luồng inference.
- Trình bày kiến trúc Docker gồm Web và ChromaDB.
- Kiểm tra HTTP, giao diện và workflow tự động.
- Demo ba kết quả: hồi quy, phân loại và phân cụm.

## 2. File đã đọc và kiểm tra

| File | Nội dung |
|---|---|
| `ui/app.py` | Giao diện, validate input và dự đoán |
| `src/start_app.py` | Đồng bộ vector rồi khởi động Gradio |
| `Dockerfile` | Môi trường Python và lệnh chạy service |
| `docker-compose.yml` | Hai service web/chroma và volume |
| `.github/workflows/verify.yml` | Build, test, HTTP smoke và chụp UI |
| `tests/smoke_http.py` | Kiểm tra server HTTP |
| `tests/capture_ui.py` | Chụp bằng chứng giao diện |

## 3. Luồng giao diện

Giao diện nạp ba model `.joblib`, kiểm tra dữ liệu nhập, biến đổi feature bằng pipeline rồi trả về:

1. Tuổi thọ dự đoán từ mô hình hồi quy.
2. Nhãn tuổi thọ `>=70` hoặc `<70` từ mô hình phân loại.
3. Nhãn cụm từ K-Means.
4. Năm hồ sơ tương đồng từ ChromaDB.

Hồi quy và phân loại học độc lập nên đôi khi có thể bất đồng quanh ngưỡng 70. Đây không mặc định là lỗi chương trình.

## 4. Kiến trúc Docker

- Service `web` chạy Gradio tại cổng 7860.
- Service `chroma` chạy vector database tại cổng nội bộ 8000.
- Web gọi Chroma bằng hostname `chroma` trong mạng Docker Compose.
- Named volume `chroma_data` giữ dữ liệu Chroma khi container khởi động lại.

## 5. Lệnh kiểm tra

Chạy local:

```powershell
.\.venv\Scripts\python.exe src\start_app.py
.\.venv\Scripts\python.exe tests\smoke_http.py
```

Chạy Docker nếu máy hỗ trợ:

```powershell
docker compose up --build -d --wait
docker compose exec -T web python -m pytest tests -q
docker compose exec -T web python tests/smoke_http.py
docker compose down
```

## 6. Kịch bản demo

Nhập một hồ sơ hợp lệ, bấm phân tích và chỉ ba kết quả. Sau đó mở bảng 5 hồ sơ gần nhất, tab trực quan hóa và tab kết quả đánh giá. Cuối cùng thử dân số bằng 0 để chứng minh giao diện chặn dữ liệu sai.

## 7. Hạn chế và câu hỏi bảo vệ

- Dữ liệu chỉ thuộc giai đoạn 1952-2007.
- UI là demo học thuật, không dùng cho quyết định y tế hoặc chính sách.
- Chỉ tuyên bố Docker đạt khi có log hoặc GitHub Actions xanh đúng commit.

**Docker để làm gì?**  
Đóng gói môi trường để máy khác và CI chạy nhất quán.

**start_app.py có train lại model không?**  
Không. Nó kiểm tra DB, đồng bộ vector và mở giao diện bằng model đã lưu.

