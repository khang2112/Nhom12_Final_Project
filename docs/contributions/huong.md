# Đóng góp của Lê Ngọc Quỳnh Hương - EDA và Insights



## 1. Phạm vi phụ trách



- Phân tích khám phá dữ liệu nâng cao.

- Kiểm tra dữ liệu thiếu và ngoại lai.

- Trình bày biểu đồ và insight.

- Giải thích ảnh hưởng của EDA đến thiết kế mô hình.



## 2. File đã đọc và kiểm tra



| File | Nội dung |

|---|---|

| `src/02_storage_clean_eda.py` | Thống kê, làm sạch và tạo biểu đồ |

| `src/08_make_notebook.py` | Sinh notebook tổng hợp |

| `notebooks/Project_Stage1_Stage2_NHOM12.ipynb` | Biểu đồ và phần giải thích |

| `reports/univariate_statistics.csv` | Mean, median, skewness và kurtosis |

| `charts/` | Các hình EDA và mô hình |



## 3. Kết quả EDA



### Dữ liệu thiếu



Các biểu đồ `01`, `02` và `17` cho biết số lượng và cấu trúc dữ liệu thiếu. Dữ liệu EDA có 375 ô thiếu trước xử lý và 0 ô thiếu sau xử lý.



### Phân phối và ngoại lai



Histogram và KDE cho thấy tuổi thọ không phân bố hoàn toàn đồng đều. Boxplot sử dụng quy tắc IQR để đánh dấu ngoại lai. Không tự động xóa các quan sát này vì chúng có thể là dữ liệu lịch sử hợp lệ.



### GDP và tuổi thọ



Heatmap và scatter cho thấy GDP/người và tuổi thọ có quan hệ cùng chiều. Tương quan Spearman xấp xỉ 0,8265. GDP có phân phối lệch nên được biến đổi `log1p` trong mô hình.



Tương quan không chứng minh quan hệ nhân quả.



### Châu lục và thời gian



Các biểu đồ grouped boxplot và stacked bar cho thấy phân phối tuổi thọ khác nhau giữa các châu lục. Vì vậy nhóm giữ `continent` và mã hóa One-Hot Encoding. Biểu đồ lịch sử cho thấy cần giữ biến `year`.



## 4. Insight dẫn sang Stage 2



1\. Biến đổi log cho dân số và GDP/người.

2\. Giữ biến `continent` và `sim_policy_group`.

3\. Xử lý missing/outlier trong pipeline.

4\. Chỉ học median và IQR từ tập train.

5\. Tách train/test theo quốc gia.

6\. Theo dõi F1 macro vì lớp không hoàn toàn cân bằng.



## 5. Bằng chứng



```text

charts/01_missingness.png

charts/03_life_expectancy_hist_kde.png

charts/04_life_expectancy_box_iqr.png

charts/06_correlation_heatmap.png

charts/07_gdp_lifeexp_scatter.png

charts/10_historical_trend.png

reports/univariate_statistics.csv

```



## 6. Lệnh kiểm tra



```powershell

.\\.venv\\Scripts\\python.exe src\\02\_storage\_clean\_eda.py

.\\.venv\\Scripts\\python.exe -m pytest tests -q

```



## 7. Hạn chế và câu hỏi bảo vệ



- Dữ liệu chỉ thuộc giai đoạn 1952-2007.

- `sim_\*` là dữ liệu mô phỏng.

- Tương quan không chứng minh nhân quả.


*\*Vì sao dùng median?\*\* Median ít nhạy với ngoại lai hơn mean.  

*\*Vì sao không xóa mọi outlier?\*\* Chúng có thể là quan sát lịch sử hợp lệ.  

*\*EDA ảnh hưởng đến model thế nào?\*\* EDA dẫn đến log transform, giữ biến châu lục/năm và xử lý trong pipeline.



