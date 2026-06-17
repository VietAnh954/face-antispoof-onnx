# CẨM NANG LÝ THUYẾT TỐI ƯU HÓA PHẦN CỨNG AI (DÀNH CHO NGƯỜI MỚI BẮT ĐẦU)

Khi đưa một mô hình Trí tuệ nhân tạo (AI) ra chạy thực tế, đặc biệt là trên các thiết bị biên (Edge Devices như điện thoại, camera thông minh, hoặc máy tính nhúng), chúng ta không chỉ cần mô hình **đoán đúng** mà còn cần mô hình chạy **cực nhanh (Low Latency)** và  **ít tốn điện/tốn bộ nhớ (Resource Efficient)** . Quá trình này được gọi là  **Tối ưu hóa phần cứng (Hardware-Aware Optimization)** .

---

## 1. Nút Cổ Chai CPU-GPU là gì? (CPU-GPU Bottleneck)

### Ẩn dụ thực tế:

Hãy tưởng tượng GPU là một **nhà máy siêu tốc** có hàng ngàn công nhân (nhân CUDA) làm việc cực nhanh. Còn CPU là **giám đốc văn phòng** ngồi ở xa. Mỗi lần nhà máy muốn sản xuất một lô hàng:

1. Giám đốc (CPU) phải viết hướng dẫn, đóng dấu giấy tờ (chuẩn bị dữ liệu).
2. Gửi xe tải (băng thông PCIe) chở nguyên liệu từ văn phòng CPU lên nhà máy GPU.
3. Nhà máy GPU sản xuất xong trong 1 giây, nhưng lại phải chờ xe tải chở sản phẩm về lại văn phòng CPU để giám đốc duyệt.

Nếu giám đốc (CPU) chuẩn bị nguyên liệu quá chậm (bằng OpenCV/Numpy), nhà máy GPU sẽ phải **ngồi chơi xơi nước** (GPU Underutilization). Đó gọi là  **CPU Bottleneck** .

### Giải pháp của chúng ta:

Chúng ta tạo ra lớp `GPUPayloadPreprocessor`. Lệnh nạp ảnh thô lên GPU duy nhất một lần. Mọi việc cắt ảnh (Crop), thay đổi kích thước (Resize), chuẩn hóa (Normalize) đều giao cho các công nhân trên GPU tự làm với nhau. Xe tải PCIe không cần chạy đi chạy lại nữa.

---

## 2. CUDA Graphs là gì? (Loại bỏ phí khởi chạy Kernel)

### Ẩn dụ thực tế:

Mô hình AI gồm hàng trăm lớp nối tiếp nhau. Mỗi lớp là một phép toán (được gọi là một **Kernel** trên GPU).

* **Không dùng CUDA Graphs:** Với mỗi khung hình camera truyền vào, CPU phải gọi GPU: *"Alo, chạy lớp 1 đi"* -> Chờ GPU phản hồi ->  *"Alo, chạy lớp 2 đi"* ... Cuộc hội thoại này (Kernel Launch) tốn từ 3-10 micro-giây cho mỗi lớp. Với hàng trăm lớp, thời gian gọi điện thoại còn lâu hơn thời gian GPU tính toán!
* **Dùng CUDA Graphs:** Ở khung hình đầu tiên, CPU ghi âm lại toàn bộ kịch bản chạy từ lớp 1 đến lớp cuối cùng thành một "bản thiết kế tĩnh" (Static Graph). Từ khung hình thứ 2 trở đi, CPU chỉ cần hét lên một câu duy nhất:  *"Chạy kịch bản cũ!"* . GPU sẽ tự động chạy một mạch từ đầu đến cuối mà không cần CPU can thiệp nữa.

---

## 3. Lượng tử hóa (Quantization) là gì? (FP32 vs FP16 vs INT8)

### Ý tưởng cơ bản:

Máy tính lưu trữ các trọng số (weights) của mô hình AI dưới dạng các con số. Độ chính xác của các con số này quyết định kích thước và tốc độ của mô hình.

1. **FP32 (Độ chính xác đơn - Single Precision):**
   * Mỗi số chiếm **32 bit** (4 byte) bộ nhớ.
   * Ví dụ: Trọng số có dạng `0.12345678`.
   * Rất chính xác nhưng tốn bộ nhớ và tính toán chậm.
2. **FP16 (Độ chính xác bán phần - Half Precision):**
   * Mỗi số chiếm **16 bit** (2 byte) bộ nhớ.
   * Ví dụ: Trọng số viết gọn thành `0.1234`.
   * Dung lượng mô hình giảm đi một nửa ngay lập tức, tốc độ tăng gấp đôi trên các GPU hỗ trợ Tensor Cores.
3. **INT8 (Số nguyên 8-bit - Integer 8-bit):**
   * Mỗi số chiếm **8 bit** (1 byte) bộ nhớ.
   * Trọng số được làm tròn thành các số nguyên từ `-128` đến `127`.
   * Mô hình siêu nhẹ (giảm 4 lần so với FP32), chạy cực nhanh, nhưng dễ bị mất độ chính xác nếu làm tròn sai cách.

### Lượng tử hóa Mixed-Precision (Độ chính xác hỗn hợp):

Lượng tử hóa tất cả mọi thứ sang INT8 sẽ khiến mô hình bị "ngốc" đi (sai số cao). Vì thế, chúng ta giữ các lớp cực kỳ nhạy cảm (như lớp phân loại cuối cùng `logits` và các khối chú ý `se_module`) ở dạng  **FP16** , còn các lớp tích chập thông thường sẽ ép về  **INT8** . Cách này giúp mô hình vừa bé, vừa nhanh, lại vừa thông minh.

---

## 4. Q/DQ Nodes là gì? (Lượng tử hóa tường minh trong TensorRT 11)

Trong các phiên bản TensorRT 11+ mới nhất, nhà phát triển không còn dùng các hàm "đoán mò và tự lượng tử hóa" nữa. Thay vào đó, họ dùng cấu trúc  **Q/DQ (Quantize / Dequantize)** .

* **Node Q (Quantize):** Nhận đầu vào là số thực (Float) và ép nó thành số nguyên (INT8) bằng một hệ số nhân (Scale).
* **Node DQ (Dequantize):** Giải nén số nguyên (INT8) trở lại số thực (Float) trước khi đưa sang lớp tiếp theo nếu lớp đó yêu cầu độ chính xác cao.

Bằng cách chèn trực tiếp các node Q/DQ này vào cấu trúc mô hình ONNX (nhờ bộ công cụ của ONNX Runtime), TensorRT khi đọc mô hình sẽ nhìn thấy rõ ràng sơ đồ:  *"À, chỗ này chạy INT8, chỗ kia chạy FP16"* . Nhờ đó, nó biên dịch ra một Engine chạy ổn định và chính xác tuyệt đối.

### Yêu cầu về Lượng tử hóa Đối xứng (Symmetric Quantization)

Để các nhân Tensor Cores trên GPU chạy INT8 đạt hiệu năng cao nhất, khoảng ánh xạ lượng tử hóa phải đối xứng qua điểm 0. Nghĩa là điểm số thực `0.0` phải tương ứng chính xác với số nguyên `0` (Zero Point = 0).

* Nếu `Zero Point` khác không, GPU sẽ phải thực hiện thêm phép cộng bù lệch (offset) làm chậm quá trình tính toán. Do đó, chúng ta bắt buộc phải cấu hình thiết lập `"ActivationSymmetric": True` và `"WeightSymmetric": True` trong quá trình hiệu chuẩn.
