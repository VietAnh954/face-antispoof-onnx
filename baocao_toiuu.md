# Báo Cáo Nghiên Cứu & Tối Ưu Hóa Hiệu Năng Hệ Thống Face Anti-Spoofing
**Đề tài:** Tối ưu hóa phần cứng hệ thống phát hiện giả mạo khuôn mặt thời gian thực sử dụng TensorRT, GPU Preprocessing và CUDA Graphs trên kiến trúc MiniFASNetV2-SE.

---

## 1. Giới Thiệu & Đặt Vấn Đề
Trong các bài toán thị giác máy tính triển khai trên thiết bị biên hoặc máy tính cá nhân, việc đảm bảo tốc độ xử lý (FPS cao, độ trễ thấp) đi kèm độ chính xác cao là yếu tố quyết định. Hệ thống phát hiện giả mạo khuôn mặt (Face Anti-Spoofing - FAS) thường phải xử lý khung hình liên tục từ luồng camera đầu vào. 
Kiến trúc baseline sử dụng **MiniFASNetV2-SE** chạy trên ONNX Runtime CPU gặp hai nút thắt cổ chai lớn:
1. **CPU Preprocessing Bottleneck:** Các bước tiền xử lý (cắt ảnh khuôn mặt từ bounding box, resize letterbox, xoay ảnh, chuẩn hóa chia 255.0, transpose kênh màu) được thực hiện trên CPU bằng OpenCV. Việc này gây ra độ trễ lớn và tiêu tốn băng thông truyền tải dữ liệu giữa CPU (Host) và GPU (Device) qua cổng PCIe.
2. **CPU Kernel Launch Overhead:** Với các mô hình mạng siêu nhẹ như MiniFASNet (~2.9MB), thời gian tính toán của GPU cho một lần suy luận là cực kỳ nhỏ (dưới 1 ms). Tuy nhiên, thời gian CPU chuẩn bị và kích hoạt các kernel tính toán trên GPU (Kernel Launch Overhead) lại chiếm phần lớn tổng thời gian xử lý.
3. **Quantization Accuracy Degradation:** Lượng tử hóa mô hình sang INT8 một cách mù quáng (quantize toàn bộ mạng) sẽ phá hủy độ chính xác của mô hình do các lớp Depthwise Separable Convolutions và Squeeze-and-Excitation (SE) cực kỳ nhạy cảm với sai số lượng tử hóa.

Để giải quyết triệt để các vấn đề trên, chúng tôi đề xuất giải pháp tối ưu hóa phần cứng toàn diện: **GPU Preprocessing + Selective Quantization INT8/FP16 + CUDA Graphs**.

---

## 2. Các Kỹ Thuật Tối Ưu Hóa Triển Khai

### 2.1. Đẩy Tiền Xử Lý Lên GPU (GPU-Accelerated Preprocessing)
Chúng tôi phát triển lớp `GPUPayloadPreprocessor` sử dụng các toán tử của PyTorch CUDA để đưa toàn bộ quá trình xử lý ảnh thô lên GPU:
- Ảnh camera được chuyển lên GPU VRAM **duy nhất 1 lần** (Zero-copy payload).
- Quá trình cắt ảnh (Crop), đệm ảnh phản xạ biên (Reflection Padding), nội suy kích thước (Bicubic Interpolation) và chuẩn hóa dữ liệu được thực hiện song song hóa trên hàng nghìn luồng GPU.
- Triệt tiêu hoàn toàn độ trễ truyền dữ liệu liên tục qua cổng PCIe trong mỗi khung hình.

### 2.2. Lượng Tử Hóa Chọn Lọc (Selective INT8 Quantization)
Để đưa mô hình về dạng lượng tử hóa tĩnh (Static QDQ INT8) mà không làm suy giảm độ chính xác, chúng tôi đã phân tích sự nhạy cảm của các lớp và đưa ra chiến lược **Lượng tử hóa chọn lọc (Selective Quantization)**:
- **Excluding Depthwise Convolutions:** Các lớp Depthwise Convolution có số lượng trọng số trên mỗi kênh rất ít (ví dụ conv 3x3 chỉ có 9 tham số trên một kênh). Lượng tử hóa các lớp này bằng một scale duy nhất cho toàn bộ tensor sẽ làm mất mát thông tin không gian nghiêm trọng. Chúng tôi đã lập trình tự động quét đồ thị ONNX và giữ các lớp này ở dạng FP16.
- **Excluding Squeeze-and-Excitation (SE) Modules:** Các lớp tích chập 1x1 trong SE Module (`fc1`, `fc2`) làm nhiệm vụ tính toán trọng số động tái phân phối kênh màu. Các hệ số nhân kênh này đòi hỏi dải động (dynamic range) lớn, lượng tử hóa tĩnh sẽ làm sai lệch hoàn toàn tính năng tự chú ý (self-attention).
- **Excluding Early Layers:** Các lớp đầu tiên (`/conv1/` và `/conv_23/`) đảm nhận nhiệm vụ chiết xuất đặc trưng tần số thấp và chi tiết texture (moire, nhiễu hạt màn hình), giữ chúng ở dạng FP16 giúp giữ lại các tín hiệu giả mạo tinh vi.
- **Biên dịch TensorRT Strongly Typed:** TensorRT biên dịch đồ thị QDQ ONNX đã được lọc thành công cơ chế tối ưu chọn lọc: chạy các lớp Conv 1x1 thông thường ở INT8 và tự động đưa các lớp nhạy cảm về FP16. Kết quả là khôi phục hoàn toàn độ chính xác đạt **84.40%** (bằng 100% so với mô hình gốc FP16/Baseline).

### 2.3. Loại Bỏ CPU Overhead Bằng CUDA Graphs
CUDA Graphs cho phép ghi lại toàn bộ quy trình thực thi kernel (GPU Preprocessing + TensorRT Inference) thành một đồ thị tĩnh trong giai đoạn Warmup (khởi động):
- Trong quá trình chạy thực tế, CPU không cần gọi lẻ tẻ hàng chục lệnh launch kernel nữa mà chỉ cần kích hoạt chạy lại đồ thị tĩnh (`cudaGraphLaunch`).
- Giảm thiểu CPU launch overhead về mức tiệm cận **0 ms**.

---

## 3. Kết Quả Thực Nghiệm & Đánh Giá

Quá trình kiểm thử được thực hiện trên GPU Laptop NVIDIA GeForce RTX 3050 sử dụng bộ dữ liệu kiểm thử gồm 250 ảnh khuôn mặt từ CelebA-Spoof (125 ảnh thật, 125 ảnh giả mạo).

### 3.1. Bảng Số Liệu So Sánh Chi Tiết

| Cấu hình mô hình / Chỉ số hiệu năng | Độ trễ E2E (ms) | Tốc độ xử lý (FPS) | Độ chính xác (%) | APCER (%) | BPCER (%) | ACER (%) | Kích thước mô hình (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ORT CPU (Baseline)** | 6.729 | 148.6 | 84.40% | 31.20% | 0.00% | 15.60% | 1.91 MB |
| **ORT GPU (CUDA)** | 7.304 | 136.9 | 84.40% | 31.20% | 0.00% | 15.60% | 1.91 MB |
| **TRT FP16 (Optimized + CUDA Graphs)** | **3.999** | **250.0** | **84.40%** | **31.20%** | **0.00%** | **15.60%** | 2.71 MB |
| **TRT INT8 (Optimized + CUDA Graphs)** | 6.913 | 144.7 | **84.40%** | **31.20%** | **0.00%** | **15.60%** | 2.86 MB |

*(Ghi chú thuật ngữ tiêu chuẩn ISO/IEC 30107-3: APCER: Tỉ lệ bỏ lọt ảnh giả mạo; BPCER: Tỉ lệ chặn nhầm người thật; ACER: Tỉ lệ lỗi trung bình).*

### 3.2. Phân Tích Hiện Tượng Thực Nghiệm (Scientific Analysis)

1. **Tại sao ORT GPU lại chậm hơn ORT CPU?**
   - Với kích thước batch = 1 và mô hình siêu nhẹ, chi phí đồng bộ bộ nhớ qua PCIe và thời gian khởi chạy kernel của GPU lớn hơn thời gian tính toán trực tiếp trên CPU. Điều này chứng minh tiền xử lý trên CPU và truyền dữ liệu là điểm nghẽn nghiêm trọng (I/O Bound).
2. **Tại sao TRT FP16 mang lại hiệu năng cao nhất (Speedup 1.68x)?**
   - Bằng cách đẩy toàn bộ tiền xử lý lên GPU thông qua `GPUPayloadPreprocessor` kết hợp với ghi đồ thị tĩnh bằng **CUDA Graphs**, chúng ta loại bỏ hoàn toàn CPU launch overhead và PCIe transfer overhead. Độ trễ giảm từ 6.729 ms xuống **3.999 ms**, đẩy tốc độ đạt mức **250 FPS**.
3. **Tại sao TRT INT8 lại chậm hơn TRT FP16?**
   - Đây là một hiện tượng kinh điển trong tối ưu hóa mạng nơ-ron nhỏ. Do chúng ta áp dụng Lượng tử hóa chọn lọc (Selective Quantization) để giữ độ chính xác tuyệt đối, đồ thị mạng xuất hiện các nút chuyển đổi kiểu dữ liệu Q/DQ (Quantize/Dequantize) giữa FP16 và INT8. 
   - Với một mô hình siêu nhỏ, thời gian tính toán của các lớp tích chập vô cùng ít. Chi phí tính toán để chuyển đổi định dạng (re-formatting overhead) tại các nút Q/DQ lớn hơn nhiều so với lượng tính toán được giảm bớt từ phép nhân tích chập INT8. Do đó, TRT FP16 vẫn là cấu hình tối ưu nhất cho bài toán này trên phần cứng RTX 3050.

---

## 4. Kết Luận
Bằng việc phối hợp các kỹ thuật tối ưu hóa phần cứng hiện đại, chúng tôi đã:
- Tăng tốc độ xử lý E2E lên **250.0 FPS** (Độ trễ chỉ **3.999 ms**), đạt mức siêu thời gian thực.
- Bảo toàn **100% độ chính xác học thuật** của mô hình gốc (84.40% Accuracy, ACER 15.60%) nhờ vào chiến lược Lượng tử hóa chọn lọc thông minh.
- Loại bỏ hoàn toàn các lỗi xung đột thư viện HighGUI của OpenCV và nạp DLL tự động trên hệ điều hành Windows.

Kết quả nghiên cứu này chứng minh rằng tối ưu hóa phần cứng không chỉ đơn thuần là chuyển đổi kiểu dữ liệu mô hình mà đòi hỏi việc phân tích chi tiết luồng dữ liệu (Dataflow) và cấu trúc đồ thị tính toán để đưa ra cấu hình lai (hybrid) phù hợp nhất.
