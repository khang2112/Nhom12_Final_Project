# Đóng góp của Võ Tiến Đạt - PCA, K-Means và mô hình dự đoán

## 1. Phạm vi phụ trách

- Giải thích feature engineering và preprocessing.
- Trình bày PCA và K-Means.
- So sánh hai mô hình hồi quy và hai mô hình phân loại.
- Giải thích cách chia dữ liệu, chọn winner và đánh giá test.

## 2. File đã đọc và kiểm tra

| File | Nội dung |
|---|---|
| `src/03_model_stage2.py` | Huấn luyện, so sánh và lưu model |
| `src/transforms.py` | Làm sạch feature không gây leakage |
| `reports/model_results.json` | Kết quả mô hình cuối cùng |
| `reports/validation_benchmark.csv` | Kết quả dùng chọn winner |
| `reports/regression_benchmark.csv` | Đánh giá hồi quy trên test |
| `reports/classification_benchmark.csv` | Đánh giá phân loại trên test |

## 3. Pipeline mô hình

- Tạo `log_population` và `log_gdp`.
- Loại `life_expectancy` và nhãn target khỏi feature.
- Điền thiếu và clipping chỉ dựa trên tập train.
- Chuẩn hóa feature số bằng `StandardScaler`.
- Mã hóa biến phân loại bằng `OneHotEncoder`.
- Chia dữ liệu theo quốc gia để cùng một nước không xuất hiện ở cả train và test.

## 4. PCA và K-Means

PCA chọn 7 thành phần, giữ khoảng 91,25% phương sai. K-Means chọn `K=2` trên dữ liệu PCA với silhouette khoảng 0,2879; cùng K trên dữ liệu gốc có silhouette khoảng 0,2638.

PCA dùng để giảm chiều, không trực tiếp dự đoán tuổi thọ. Silhouette 0,2879 cho thấy cụm còn chồng lấn, không được mô tả là tách biệt hoàn toàn.

## 5. Supervised Learning

Nhóm so sánh:

- Hồi quy: Linear Regression và Random Forest Regressor.
- Phân loại: Logistic Regression và Random Forest Classifier.

Winner được chọn bằng validation, sau đó refit trên fit + validation. Test cuối không được dùng để quay lại chọn model.

Kết quả hiện tại:

```text
Linear Regression
RMSE = 6.17068
R2 = 0.77811

Logistic Regression
Accuracy = 0.87069
F1 macro = 0.84529
```

## 6. Bằng chứng

```text
models/regression_winner.joblib
models/classification_winner.joblib
models/clustering_pipeline.joblib
reports/model_results.json
charts/11_pca_cumulative_variance.png
charts/12_kmeans_elbow_silhouette.png
charts/14_regression_actual_vs_predicted.png
charts/16_confusion_matrix.png
```

## 7. Câu hỏi bảo vệ

**Vì sao không random từng dòng?**  
Một quốc gia lặp nhiều năm; random từng dòng dễ làm kết quả quá lạc quan.

**Vì sao Linear/Logistic thắng Random Forest?**  
Trong cấu hình đang thử, chúng đạt validation metric tốt hơn; không có nghĩa luôn tốt hơn mọi mô hình cây.

**Vì sao lưu cả pipeline?**  
Để inference dùng đúng cleaner, scaler, encoder và quy tắc đã học khi train.

