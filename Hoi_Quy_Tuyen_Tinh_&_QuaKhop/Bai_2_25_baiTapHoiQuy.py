import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Tải tập dữ liệu (Tập dữ liệu giá nhà California)
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Chia tập dữ liệu thành Train và Test (tỷ lệ 80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình hồi quy (sử dụng Ridge Regression để tránh quá khớp)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 4. Dự báo trên tập kiểm tra
y_pred = model.predict(X_test)

# 5. Hàm tính toán chỉ số NSE và các chỉ số khác
def nse_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
nse = nse_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print(f"NSE  : {nse:.4f}")