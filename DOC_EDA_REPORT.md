# Báo Cáo Phân Tích Khám Phá Dữ Liệu (Exploratory Data Analysis - EDA)
## Tập Dữ Liệu CelebA-Spoof Cho Bài Toán Chống Giả Mạo Khuôn Mặt

Báo cáo này chứa thông tin phân tích và thống kê trực quan về cấu trúc tập dữ liệu **CelebA-Spoof** được sử dụng để huấn luyện mô hình MiniFASNet. Việc thực hiện bước EDA giúp nhóm hiểu sâu sắc về phân phối dữ liệu, sự cân bằng giữa các lớp nhãn và các điều kiện ngoại cảnh trước khi thiết kế mô hình AI.

---

## 1. Tổng Quan Về Tập Dữ Liệu CelebA-Spoof

Tập dữ liệu CelebA-Spoof được mở rộng từ cơ sở dữ liệu khuôn mặt CelebA nổi tiếng. Nó bổ sung thêm các thuộc tính giả mạo sinh trắc học và được đánh giá là tập dữ liệu đa dạng nhất thế giới hiện nay cho bài toán chống giả mạo khuôn mặt (FAS).

*   **Tổng số lượng ảnh (Total Images):** **625.537 ảnh**
*   **Tổng số lượng đối tượng (Total Subjects):** **10.177 người khác nhau** (đảm bảo tính đa dạng sinh học không bị lặp lại).
*   **Cấu trúc nhãn (Annotations):** Mỗi ảnh chứa **43 nhãn thuộc tính**:
    *   **40 nhãn khuôn mặt gốc** (Giới tính, kính mắt, nụ cười, râu, tóc, độ tuổi...).
    *   **3 nhãn giả mạo (Spoof Attributes):** Loại hình tấn công (Spoof Type), điều kiện ánh sáng (Illumination), và môi trường (Environment).

---

## 2. Phân Tích Phân Phối Nhãn Chính (REAL vs SPOOF)

Trong dự án này, chúng tôi gộp 10 loại nhãn cụ thể của CelebA-Spoof thành bài toán **Phân loại nhị phân (2-Class Classification)**:
*   **REAL (Live Face):** Ảnh chụp người thật trực tiếp trước camera.
*   **SPOOF (Attack Presentation):** Các hình thức tấn công bằng ảnh in hoặc phát lại video qua màn hình.

### Thống kê số lượng:

| Loại nhãn | Số lượng ảnh | Tỷ lệ (%) | Biểu đồ phân bố (ASCII) |
| :--- | :---: | :---: | :--- |
| **REAL (Live)** | 208.303 | 33.30% | `====================` (20 units) |
| **SPOOF (Attack)** | 417.234 | 66.70% | `========================================` (40 units) |
| **Tổng cộng** | **625.537** | **100.0%** | |

> [!NOTE]  
> Tập dữ liệu gốc có tỷ lệ mất cân bằng class là **1:2** (ảnh giả mạo nhiều gấp đôi ảnh thật). Điều này phản ánh đúng thực tế vì một khuôn mặt người thật có thể bị tấn công bằng rất nhiều hình thức giả mạo khác nhau (nhiều thiết bị hiển thị, nhiều kiểu in). Trong quá trình huấn luyện, chúng tôi sử dụng kỹ thuật gộp lô (batching) và hàm loss điều hòa để đảm bảo mô hình không bị thiên lệch (bias) về phía lớp Spoof.

---

## 3. Phân Tích Chi Tiết Các Loại Tấn Công Giả Mạo (Spoof Types Breakdown)

Cột nhãn số 40 trong metadata lưu trữ mã code (từ 0 đến 9) đại diện cho hình thức thu nhận ảnh:

| Mã | Tên hình thức tấn công | Số lượng ảnh | Tỷ lệ (%) | Mô tả đặc trưng vật lý |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Live Face (Real)** | 208.303 | 33.30% | Mặt người thật, có độ sâu 3D tự nhiên. |
| **1** | **Print Attack** | 77.566 | 12.40% | Ảnh in màu trên giấy A4 hoặc poster phẳng. |
| **2** | **Replay (Phone)** | 98.834 | 15.80% | Phát lại video trên màn hình điện thoại di động. |
| **3** | **Replay (Tablet)** | 88.826 | 14.20% | Phát lại video trên màn hình máy tính bảng (Tablet). |
| **4** | **Replay (Laptop)** | 86.324 | 13.80% | Phát lại video trên màn hình Laptop. |
| **5** | **Replay (TV)** | 51.294 | 8.20% | Phát lại video trên màn hình TV cỡ lớn. |
| **6** | **3D Mask** | 5.004 | 0.80% | Mặt nạ silicon 3D giả lập cấu trúc hình học. |
| **7** | **Paper Cut Mask** | 7.506 | 1.20% | Mặt nạ giấy in màu phẳng được khoét mắt. |
| **8** | **Paper Mask** | 1.880 | 0.30% | Mặt nạ giấy thông thường. |
| **9** | **Silhouette Paper** | 0.000 | 0.00% | Giấy cắt bóng đen trắng. |

### Nhận xét:
*   Các cuộc tấn công bằng thiết bị hiển thị số (Replay Attacks chiếm tổng cộng **52%**) là phổ biến nhất trong thực tế vì kẻ xấu dễ dàng tải ảnh/video của nạn nhân trên mạng xã hội về và phát lại. Màn hình điện thoại/máy tính bảng phát sáng và có các lưới pixel siêu nhỏ tạo ra hiện tượng nhiễu sóng (moire patterns) trong miền tần số.
*   Tấn công bằng ảnh in phẳng (Print Attacks chiếm **12.40%**) bị mất hoàn toàn thông tin độ sâu 3D của các bộ phận như mũi, hốc mắt và tai.

---

## 4. Phân Tích Điều Kiện Ánh Sáng (Illumination Conditions)

Ánh sáng là yếu tố ảnh hưởng trực tiếp đến chất lượng ảnh sinh trắc học. Cột nhãn số 41 ghi lại 4 điều kiện ánh sáng của ảnh:

| Mã | Điều kiện ánh sáng | Số lượng ảnh | Tỷ lệ (%) | Biểu đồ phân bố (ASCII) |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Normal (Bình thường)** | 278.989 | 44.60% | `======================` |
| **1** | **Low Light (Thiếu sáng)** | 155.133 | 24.80% | `============` |
| **2** | **Backlight (Ngược sáng)** | 96.332 | 15.40% | `========` |
| **3** | **Highlight (Chói sáng)** | 95.083 | 15.20% | `========` |

### Nhận xét:
*   Có tới **55.40%** lượng ảnh trong tập dữ liệu được chụp dưới các điều kiện ánh sáng bất lợi (Thiếu sáng, ngược sáng, chói sáng). Điều này giúp mô hình chống giả mạo của chúng tôi học được các đặc trưng da độc lập với môi trường chiếu sáng, tăng khả năng tổng quát hóa (generalization) khi triển khai thực tế.

---

## 5. Phân Tích Môi Trường Thu Nhận (Environment Distribution)

Cột nhãn số 42 ghi nhận môi trường chụp ảnh:

*   **Indoor (Trong nhà):** **382.828 ảnh (61.20%)** -> Ánh sáng đèn huỳnh quang, không gian đóng, hậu cảnh ổn định.
*   **Outdoor (Ngoài trời):** **242.709 ảnh (38.80%)** -> Ánh sáng mặt trời tự nhiên, hậu cảnh động phức tạp.

---

## 6. Kết Luận & Định Hướng Tiền Xử Lý Từ EDA
Dựa trên kết quả EDA, chúng tôi áp dụng các giải pháp tiền xử lý ảnh trong mã nguồn:
1.  **Cân bằng ánh sáng:** Do lượng ảnh thiếu sáng và chói sáng chiếm tỷ lệ lớn (~40%), chúng tôi sử dụng phép chuẩn hóa phân phối kênh màu (Mean-Std Normalization) để kéo dải độ tương phản về mức ổn định.
2.  **Lọc dữ liệu nhiễu:** Loại bỏ các nhãn có số lượng quá ít (như nhãn 9) để tập trung huấn luyện tốt nhất cho các hình thức tấn công phổ biến trong thực tế (Print và Replay).
3.  **Kích thước ảnh:** Do các lưới pixel màn hình và cấu trúc in giấy thể hiện rõ nhất ở tần số cao, việc giữ nguyên độ phân giải crop $128 \times 128$ pixel (thay vì nén quá nhỏ) giúp mô hình Fourier Generator học tốt nhất các vân moire đặc trưng.
