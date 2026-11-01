# Đóng góp của Nguyễn Dương Khang - Nguồn và tích hợp dữ liệu

## 1. Phạm vi phụ trách

- Giới thiệu mục tiêu và luồng tổng thể của dự án Atlas Insight.
- Giải thích hai nguồn Gapminder và World Bank.
- Trình bày quá trình crawl, chuẩn hóa, concat và merge dữ liệu.
- Kiểm tra khóa ghép và giới hạn của dữ liệu.

Phần này tương ứng tiêu chí **Thu thập và tích hợp dữ liệu** trong rubric.

## 2. File đã đọc và kiểm tra

| File | Nội dung |
|---|---|
| `src/00_crawl_worldbank.py` | Crawl World Bank API bằng Scrapy và tạo audit |
| `src/01_prepare_data.py` | Đọc Gapminder, chia feed, sinh `sim_*` và tích hợp |
| `src/09_standardize_worldbank.py` | Chuẩn hóa ISO, năm và kiểu dữ liệu |
| `src/10_compare_sources.py` | Đối chiếu hai nguồn theo ISO + year |
| `src/11_concat_two_sources.py` | Nối dọc hai nguồn và giữ `data_source` |
| `src/12_merge_multisource_dynamic.py` | Ghép thuộc tính mô phỏng theo `record_id` |
| `reports/crawl_audit.json` | URL, thời gian tải, số bản ghi và SHA256 |

## 3. Cách hoạt động

Gapminder có 1.704 dòng, tương ứng 142 quốc gia và 12 mốc năm từ 1952 đến 2007. World Bank được crawl bằng Scrapy, ghi nhận 12.768 bản ghi thô và 8 thuộc tính. Sau khi lọc cùng phạm vi còn 1.400 dòng World Bank.

Nhóm sử dụng:

- `concat` khi nối thêm dòng và giữ cột `data_source`.
- `merge` khi bổ sung thuộc tính theo khóa.
- `left join` khi cần giữ toàn bộ bảng chính.
- `inner join` khi chỉ đối chiếu các cặp có ở cả hai nguồn.
- `validate` để phát hiện quan hệ khóa sai và tránh nhân bản dữ liệu.

Bảng đa nguồn có 3.104 dòng nhưng chỉ có 1.704 `record_id` khác nhau. Không xem toàn bộ 3.104 dòng là các mẫu Machine Learning độc lập.

## 4. Input và output

Input chính:

```text
Gapminder CSV
World Bank API
```

Output chính:

```text
data/raw/worldbank_scrapy_records.csv
data/processed/worldbank_life_expectancy_standardized.csv
data/processed/multisource_with_dynamic.csv
reports/crawl_audit.json
```

## 5. Bằng chứng và kiểm tra

```powershell
.\.venv\Scripts\python.exe src\00_crawl_worldbank.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

Kết quả chỉ ghi là đạt sau khi tự chạy hoặc đối chiếu đúng commit đang báo cáo.

## 6. Hạn chế

- Dữ liệu chỉ thuộc giai đoạn 1952-2007.
- Feed A và Feed B chỉ là hai phần địa lý của Gapminder, không phải hai tổ chức độc lập.
- Các biến `sim_*` là dữ liệu mô phỏng.
- Không dùng dữ liệu này để dự đoán tuổi thọ cá nhân hoặc khẳng định hiện trạng năm 2026.

## 7. Câu hỏi bảo vệ

**Vì sao không ghép bằng tên quốc gia?**  
Tên có thể khác cách viết; ISO kết hợp year ổn định và kiểm tra được.

**Concat khác merge thế nào?**  
Concat thêm dòng; merge thêm cột theo khóa.

**Vì sao 3.104 không phải số mẫu ML?**  
Một quốc gia-năm có thể xuất hiện ở cả hai nguồn.

