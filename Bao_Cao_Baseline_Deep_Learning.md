# Báo Cáo Cốt Lõi (Baseline) Dự Án Face Anti-Spoofing Cho Môn Học Deep Learning

Báo cáo này mô tả chi tiết toàn bộ kiến trúc gốc (baseline) của hệ thống phát hiện giả mạo khuôn mặt bao gồm: kiến trúc mạng nơ-ron, quy trình tiền xử lý, cấu trúc dữ liệu huấn luyện/kiểm thử, hàm lỗi phối hợp, siêu tham số và tệp trọng số huấn luyện.

---

## 1. Kiến Trúc Mạng Nơ-ron Cốt Lõi (Model Architecture)

Mô hình AI cốt lõi sử dụng mạng **MiniFASNetV2-SE** (thuộc họ mạng MiniFASNet - Mini Face Anti-Spoofing Network) kết hợp với cơ chế tự chú ý kênh màu Squeeze-and-Excitation (SE).

### 1.1. Khối Depthwise Separable Convolution (Trọng tâm tích chập siêu nhẹ)
Để chạy được thời gian thực trên các thiết bị biên, mạng thay thế các lớp tích chập thông thường bằng **Depthwise Separable Convolution**:
- **Depthwise Convolution:** Tích chập không gian trên từng kênh riêng lẻ bằng các bộ lọc kích thước $3 \times 3$ (Số nhóm `groups` bằng số kênh đầu vào).
- **Pointwise Convolution (Linear Block):** Tích chập $1 \times 1$ để tổng hợp thông tin giữa các kênh màu.
- **Hiệu quả:** Giảm khối lượng tính toán và số tham số của mô hình xuống khoảng $8$ đến $9$ lần so với tích chập truyền thống mà không làm giảm độ chính xác.

### 1.2. Khối Tự Chú Ý Kênh Màu (Squeeze-and-Excitation - SE Module)
Khối SEModule giúp mô hình tự học cách gán trọng số tầm quan trọng cho từng kênh đặc trưng:
1. **Squeeze (Ép):** Sử dụng phép toán Global Average Pooling để nén toàn bộ không gian ảnh $H \times W$ của mỗi kênh thành một giá trị đặc trưng duy nhất đại diện cho kênh đó.
2. **Excitation (Kích hoạt):** Đưa qua hai lớp tích chập $1 \times 1$ (tương đương mạng MLP thu nhỏ) và hàm kích hoạt Sigmoid để tính toán hệ số kích hoạt cho từng kênh (từ $0$ đến $1$).
3. **Scale (Tái phân phối):** Nhân trực tiếp hệ số kích hoạt trở lại các kênh đặc trưng ban đầu để làm nổi bật các đặc trưng quan trọng.

---

## 2. Nhánh Phụ Sinh Ảnh Phổ Tần Số Fourier (Fourier Transform Auxiliary Head)

Một đóng góp toán học lớn của baseline này là việc sử dụng **nhánh phụ sinh ảnh tần số phổ Fourier (FTGenerator)** trong quá trình huấn luyện.

### 2.1. Cơ sở lý thuyết của nhánh Fourier
Ảnh chụp người thật và ảnh giả mạo chụp lại từ màn hình hoặc ảnh in có sự khác biệt rất lớn về tần số cao (chi tiết hạt nhiễu moire, hạt mực, độ mịn da). 
- Ảnh thật có phổ tần số Fourier mượt mà, phân bố tự nhiên.
- Ảnh giả mạo xuất hiện các đỉnh năng lượng cao bất thường tại các tần số cụ thể (do lưới pixel màn hình).

### 2.2. Hoạt động của nhánh phụ
- Nhánh phụ `FTGenerator` nhận đầu vào là các đặc trưng trung gian của mô hình và cố gắng tái tạo (sinh ngược lại) ảnh phổ tần số Fourier 2D của khuôn mặt đó.
- **Lưu ý quan trọng:** Nhánh Fourier này **chỉ hoạt động trong quá trình Huấn luyện (Training)** nhằm ép mạng nơ-ron ở các tầng dưới học được các đặc trưng tần số hạt nhiễu. Khi chạy **Demo/Suy luận (Inference/Testing)**, nhánh này bị **loại bỏ hoàn toàn** để giảm thiểu khối lượng tính toán của hệ thống.

---

## 3. Quy Trình Huấn Luyện & Hàm Tổn Hao (Training & Loss Functions)

### 3.1. Cấu trúc tập dữ liệu CelebA-Spoof
Tập dữ liệu huấn luyện được cấu trúc trong JSON metas chứa thông tin nhãn ảnh.
- Nhãn phân loại gốc của CelebA-Spoof bao gồm 10 loại (từ 0 đến 9), đại diện cho ảnh thật và các hình thức tấn công giả mạo khác nhau (in ảnh giấy, chụp qua màn hình điện thoại, ipad, mặt nạ 3D...).
- Mô hình baseline gộp các nhãn này thành bài toán **Phân loại nhị phân (2-class classification)**:
  * **Lớp 0 (Real):** Nhãn gốc [0] (Khuôn mặt thật).
  * **Lớp 1 (Spoof):** Nhãn gốc [1, 2, 3, 7, 8, 9] (Tất cả các hình thức giả mạo).
- Phân chia tập dữ liệu: Tập huấn luyện (Train set) chiếm 80% dữ liệu, tập kiểm tra (Validation set) chiếm 20% dữ liệu.

### 3.2. Hàm tổn hao phối hợp (Combined Loss Function)
Quá trình huấn luyện sử dụng một hàm lỗi kết hợp từ hai nhiệm vụ:
$$Loss_{total} = Loss_{class} + \lambda \cdot Loss_{FT}$$

Trong đó:
1. **$Loss_{class}$ (Hàm lỗi phân loại):** Sử dụng hàm **Cross-Entropy Loss** tiêu chuẩn để đo lường sai số phân loại giữa mặt thật và mặt giả mạo.
   $$Loss_{class} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
2. **$Loss_{FT}$ (Hàm lỗi Fourier):** Sử dụng hàm **Mean Squared Error (MSE) Loss** để đo lường sai số giữa ảnh phổ Fourier do nhánh phụ sinh ra với ảnh phổ Fourier thực tế được tính bằng toán tử FFT2 của ảnh đầu vào.
   $$Loss_{FT} = \frac{1}{H \times W} \sum_{u=1}^{H} \sum_{v=1}^{W} \left( F_{pred}(u,v) - F_{target}(u,v) \right)^2$$
3. **$\lambda$ (Hệ số điều hòa):** Thường được đặt mặc định là $10.0$ để cân bằng độ lớn của hai hàm lỗi.

### 3.3. Siêu tham số huấn luyện (Training Hyperparameters)
Các thông số cấu hình mặc định trong mã nguồn huấn luyện (`src/minifasv2/config.py`):
- **Thuật toán tối ưu (Optimizer):** Stochastic Gradient Descent (SGD) với động lượng (momentum) $0.9$, trọng số suy giảm (weight decay) $0.0005$.
- **Tốc độ học ban đầu (Learning Rate):** $LR = 0.1$.
- **Lịch trình giảm tốc độ học (LR Scheduler):** Giảm tốc độ học đi 10 lần ($\gamma = 0.1$) tại các mốc epoch: $10, 15, 22, 30$.
- **Tổng số Epoch:** $50$.
- **Kích thước lô (Batch size):** $256$.
- **Kích thước ảnh đầu vào:** $128 \times 128$ pixel.

---

## 4. Tệp Trọng Số & Trạng Thái Mô Hình (Checkpoints)

Sau khi huấn luyện thành công 50 epochs, tệp trọng số tốt nhất được lưu lại tại đường dẫn [best_model.pth](file:///c:/Users/ADMIN/Documents/Work/school/DeepL/face-antispoof-onnx-main/face-antispoof-onnx-main/models/best/98.20/best_model.pth):
- Tệp lưu trữ dưới dạng dictionary chứa `model_state_dict` (danh sách các tensor trọng số của từng lớp).
- Dung lượng file PyTorch: **~2.83 MB** (cực kỳ nhẹ, phù hợp triển khai thiết bị di động).
- Độ chính xác kiểm chứng đạt **98.20%** trên tập validation nội bộ CelebA-Spoof.
