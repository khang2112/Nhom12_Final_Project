# Atlas Insight - Nhóm 12

Hệ thống phân tích dữ liệu quốc gia theo năm để:

- Dự đoán tuổi thọ bằng hồi quy.
- Phân loại tuổi thọ từ 70 năm trở lên.
- Phân cụm hồ sơ quốc gia-năm bằng PCA và K-Means.
- Tìm các hồ sơ tương đồng bằng ChromaDB.
- Trình bày kết quả trên giao diện Gradio.

## Dữ liệu

| Nguồn | Quy mô | Mục đích |
|---|---:|---|
| Gapminder | 1.704 dòng | Dữ liệu chính cho EDA và Machine Learning |
| World Bank API | 12.768 bản ghi thô | Nguồn độc lập để kiểm tra và tích hợp |
| World Bank sau chuẩn hóa | 1.400 dòng | So sánh cùng phạm vi với Gapminder |
| Bảng đa nguồn | 3.104 dòng | Giữ riêng từng nguồn, không dùng toàn bộ để train |

Dữ liệu thuộc giai đoạn **1952-2007**. Các biến `sim_*` là dữ liệu mô phỏng phục vụ thực hành.

## Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| PCA | 7 thành phần, giữ 91,25% phương sai |
| K-Means | K=2, Silhouette=0,2879 |
| Hồi quy tốt nhất | Linear Regression, RMSE=6,17068, R²=0,77811 |
| Phân loại tốt nhất | Logistic Regression, Accuracy=0,87069, F1 macro=0,84529 |
| ChromaDB | 1.704 vector, 7 chiều |

Chi tiết nằm trong `reports/model_results.json` và các file benchmark trong `reports/`.

## Chạy bằng Python

Yêu cầu Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe src\start_app.py
```

Mở: http://127.0.0.1:7860

Nếu `.venv` đã tồn tại và test đã đạt thì chỉ cần chạy `src\start_app.py`.

## Chạy bằng Docker

```powershell
docker compose up --build -d --wait
docker compose exec -T web python -m pytest tests -q
docker compose exec -T web python tests/smoke_http.py
```

Mở: http://127.0.0.1:7860

Dừng hệ thống:

```powershell
docker compose down
```

Chỉ chọn một cách chạy tại một thời điểm vì Python local và Docker cùng sử dụng cổng `7860`.

## Tái tạo pipeline

Chỉ chạy khi cần tạo lại dữ liệu, biểu đồ và model:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

Tải lại World Bank bằng Scrapy:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --crawl
```

## Cấu trúc chính

```text
src/        Xử lý dữ liệu, EDA, Machine Learning và ChromaDB
ui/         Giao diện Gradio
data/       Dữ liệu thô và đã xử lý
models/     Model đã huấn luyện
reports/    Kết quả đánh giá và audit
charts/     Biểu đồ
databases/  SQLite và ChromaDB
tests/      Kiểm thử pipeline, database và giao diện
docs/       Báo cáo và đóng góp thành viên
slide/      Slide thuyết trình
```

## Phân công 5 thành viên

| Thành viên | Phần phụ trách |
|---|---|
| Nguyễn Dương Khang | Nguồn dữ liệu và tích hợp |
| Đào Gia Huy | Làm sạch, SQLite và ChromaDB |
| Lê Ngọc Quỳnh Hương | EDA, biểu đồ và insight |
| Võ Tiến Đạt | PCA, K-Means, hồi quy và phân loại |
| Nguyễn Ngọc Hoàng | UI, Docker và demo |

## Giới hạn

- Không suy luận quan hệ nhân quả từ tương quan.
- Không dùng để dự đoán tuổi thọ cá nhân.
- Không dùng dữ liệu 1952-2007 để khẳng định hiện trạng năm 2026.
- Các biến mô phỏng không phải số liệu y tế hoặc giáo dục thực tế.

Nguồn: [Gapminder](https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv) và [World Bank API](https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json).
