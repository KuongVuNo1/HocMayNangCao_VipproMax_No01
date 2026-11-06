import numpy as np

class Perceptron:
    def __init__(self):
        self.w = None

    def fit(self, X, y, w_init=None, max_iter=1000):
        """
        Huấn luyện mô hình Perceptron.
        
        Tham số:
        X: Ma trận đặc trưng (đã thêm bias). Kích thước (N, d).
        y: Vector nhãn thực tế (+1 hoặc -1). Kích thước (N, ).
        w_init: Vector trọng số khởi tạo. Nếu None, sẽ khởi tạo ngẫu nhiên.
        max_iter: Số vòng lặp tối đa để tránh lặp vô hạn nếu dữ liệu không phân tách tuyến tính.
        """
        N, d = X.shape
        if w_init is None:
            self.w = np.random.randn(d) #[cite: 2]
        else:
            self.w = w_init
            
        for _ in range(max_iter):
            # 1. Dự đoán nhãn hiện tại[cite: 2]
            pred = self.predict(X)
            
            # 2. Tìm các điểm bị phân lớp sai[cite: 2]
            mis_idxs = np.where(np.equal(pred, y) == False)[0]
            num_mis = mis_idxs.shape[0]
            
            # 3. Điều kiện dừng: không còn điểm nào phân lớp lỗi[cite: 2]
            if num_mis == 0:
                break
                
            # 4. Chọn ngẫu nhiên một điểm lỗi để cập nhật trọng số[cite: 2]
            random_id = np.random.choice(mis_idxs, 1)[0]
            
            # 5. Cập nhật trọng số: w = w + y_i * x_i[cite: 2, 3]
            self.w = self.w + y[random_id] * X[random_id]
            
        return self.w

    def predict(self, X):
        """
        Dự báo nhãn cho dữ liệu mới.
        
        Tham số:
        X: Ma trận đặc trưng mới (đã thêm bias). Kích thước (M, d).
        """
        # Áp dụng hàm sgn(w^T x) để dự đoán nhãn[cite: 2, 3]
        return np.sign(X.dot(self.w))