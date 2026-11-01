# Đóng góp của Đào Gia Huy - Làm sạch và database

## 1. Phạm vi phụ trách

- Kiểm tra và xử lý dữ liệu thiếu.
- Giải thích cách chống data leakage trong Machine Learning.
- Trình bày SQLite và ChromaDB.
- Chứng minh dữ liệu có thể được đọc ngược thành DataFrame.

## 2. File đã đọc và kiểm tra

| File | Nội dung |
|---|---|
| `src/02_storage_clean_eda.py` | Làm sạch dữ liệu EDA và lưu SQLite |
| `src/13_fill_missing_lifeexp.py` | Điền 6 giá trị tuổi thọ World Bank còn thiếu |
| `src/14_clean_multisource.py` | Làm sạch bảng đa nguồn theo `record_id` |
| `src/transforms.py` | `FeatureCleaner` học thống kê trên tập train |
| `src/04_chromadb_vector.py` | Ghi vector PCA vào ChromaDB |
| `src/vector_store.py` | Đọc DataFrame và tìm vector gần nhất |

## 3. Xử lý dữ liệu thiếu

Trong dữ liệu EDA:

- `sim_health_index` và `sim_education_index` được điền median theo châu lục, sau đó dùng median toàn cục nếu vẫn thiếu.
- `sim_trend_index` được `ffill/bfill` theo chuỗi năm của từng quốc gia.
- `sim_policy_group` thiếu được giữ thành `Unknown`.

Dữ liệu EDA có 375 ô thiếu trước xử lý và 0 ô thiếu sau xử lý.

Trong supervised ML, `FeatureCleaner` chỉ học median và biên IQR từ tập fit/train. Không dùng thống kê học từ toàn bộ dữ liệu vì sẽ gây data leakage.

## 4. Hai database

### SQLite

Lưu các bảng có cấu trúc như:

```text
master_sparse
master_clean
master_with_clusters
```

Code sử dụng `to_sql` để ghi và `read_sql` để đọc ngược.

### ChromaDB

ChromaDB lưu 1.704 vector PCA, mỗi vector có 7 chiều. Hàm `read_frame` dựng lại DataFrame; `nearest_frame` tìm 5 hồ sơ gần nhất.

Vector trong dự án là đặc trưng số sau PCA, không phải embedding văn bản từ LLM.

## 5. Bằng chứng

```text
reports/chroma_status.json
reports/chroma_roundtrip.csv
databases/gapminder_project.sqlite
databases/chroma_db/
```

Trạng thái ChromaDB hiện tại cần kiểm tra:

```text
ingested = true
count = 1704
dimensions = 7
roundtrip_dataframe = true
```

## 6. Lệnh kiểm tra

```powershell
.\.venv\Scripts\python.exe src\02_storage_clean_eda.py
.\.venv\Scripts\python.exe src\04_chromadb_vector.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 7. Hạn chế và câu hỏi bảo vệ

**Vì sao dùng median?**  
Median ít bị ảnh hưởng bởi ngoại lai hơn mean.

**`bfill` có nhìn giá trị tương lai không?**  
Có thể có, nên chỉ dùng cho EDA hồi cứu, không dùng để đánh giá supervised.

**Vì sao cần hai database?**  
SQLite phục vụ dữ liệu bảng và SQL; ChromaDB phục vụ truy vấn vector tương đồng.

