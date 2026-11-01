# Atlas Insight — Nhóm 12

Phân tích quốc gia–năm, dự đoán tuổi thọ, phân loại ≥70 năm và phân cụm.

## Chạy nhanh bằng Docker

Cài Docker Desktop, bật Linux containers. Tại thư mục project:

```powershell
docker compose up --build -d --wait
```

Mở http://localhost:7860. Web gọi ChromaDB qua `chroma:8000`; vector lưu trong named volume `chroma_data`. SQLite lưu bảng dữ liệu nền trong project. Khởi động chỉ đồng bộ vector, không huấn luyện lại model.

```powershell
docker compose exec web python -m pytest tests -q
docker compose exec web python tests/smoke_http.py
docker compose logs --tail 60 web
docker compose down
```

Không dùng `down -v` nếu muốn giữ dữ liệu vector. GitHub Actions kiểm tra build, UI, database và restart; ảnh UI chạy Docker nằm trong artifact `docker-ui-evidence` của workflow. **Cấu hình Docker chưa được chạy bằng Docker Engine trong môi trường sửa bài; cần workflow xanh hoặc chạy các lệnh trên trước khi báo cáo Docker thành công.**

## Chạy Python cục bộ

Python 3.12; các package đã ghim phiên bản trong requirements.txt.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/start_app.py
```

`start_app.py` dùng Chroma nhúng khi không có CHROMA_HOST. Không chạy đồng thời hai tiến trình ghi vào cùng thư mục Chroma nhúng. Muốn nhiều tiến trình, dùng Docker/HTTP.

## Tái tạo dữ liệu và model

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
# Thêm --crawl nếu cần tải mới từ World Bank bằng Scrapy.
.\.venv\Scripts\python.exe src/07_build_deliverables.py
```

Pipeline ghi đè đầu ra. Bản kèm theo đã crawl thật 12.768 bản ghi/8 trường; xem `reports/crawl_audit.json`. Sau lọc WB còn 1.400 dòng, nối Gapminder 1.704 thành 3.104 dòng theo nguồn. Chỉ tập 1.704 country-year dùng ML.

## Các sửa đổi chính

- Scrapy thu thập API World Bank, có URL, timestamp, SHA256; đưa vào nhánh 09–14.
- Missingno, thống kê thiếu, so sánh phân phối; IQR clipping feature GDP/dân số sau log.
- Supervised đọc dữ liệu chưa điền thiếu; imputer, scaler, encoder chỉ fit train.
- Chia quốc gia thành fit 90 / validation 23 / test 29. Chọn model bằng validation, refit 113 nước, báo cáo test cuối 29 nước.
- Xuất lại cả 3 model với sklearn 1.8.0; UI chặn sai phiên bản và input không hợp lệ.
- Chroma round-trip DataFrame; UI truy vấn 5 vector gần nhất, không lấy mẫu giả tương đồng.
- Docker Web/Chroma tách riêng, volume, startup retry, healthcheck và CI.

## Báo cáo ngắn

- `docs/Bao_Cao_Final.pdf`: 8 trang, phân công 4 người và code cần mở.
- `slide/Slides_Final_Project.pdf`: slide ngắn, cùng số liệu hiện tại.
- `notebooks/Project_Stage1_Stage2_NHOM12.ipynb`: EDA, 17 biểu đồ và insight; chỉ đọc artifact khi Run All.
- `reports/model_results.json`: số liệu gốc; `reports/validation_benchmark.csv`: căn cứ chọn winner.
- `reports/integration_status.json`: kiểm tra HTTP/DB thật; không phải chứng nhận Docker.

Các DOCX/PPTX dài cũ và ảnh cũ đã bỏ khỏi bản cập nhật để tránh trình bày nhầm metric. Có thể lấy lại từ lịch sử Git. Hai script bonus 05/06 giữ lại làm tham khảo, không dùng chứng minh rubric thu thập.

## Phân công 4 người

| Người | Nội dung | Code |
|---|---|---|
| 1 | Nguồn, Scrapy, concat/merge | 00, 01, 09–14 |
| 2 | SQLite/Chroma, thiếu dữ liệu, EDA | 02, 04, vector_store |
| 3 | Feature, split, 6 thuật toán, metrics | 03, transforms |
| 4 | UI, Docker, demo, kết luận | ui/app.py, start_app, tests, Compose |

Giữ danh sách tác giả gốc: Nguyễn Dương Khang, Đào Gia Huy, Lê Ngọc Quỳnh Hương, Võ Tiến Đạt, Nguyễn Ngọc Hoàng. Nhóm tự gán 4 người báo cáo.

Dữ liệu 1952–2007 và các biến sim_* mô phỏng chỉ phục vụ thực hành. Không suy luận nhân quả hoặc dự báo hiện tại.

Nguồn: [Gapminder](https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv), [World Bank API](https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json).
