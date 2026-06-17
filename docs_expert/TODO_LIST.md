# Checklist Tối Ưu Hóa Cốt Lõi (Peak Performance Optimization Checklist)

Hệ thống tài liệu và checklist theo dõi tiến độ nâng cấp tối ưu hóa phần cứng cho pipeline `face-antispoof-onnx`.

## BƯỚC 1: Khảo sát và Đánh giá Baseline (Audit Phase)
- [x] Đọc và phân tích toàn bộ mã nguồn baseline (`MiniFASNetV2-SE`, `FTGenerator`, `preprocess.py`, `inference.py`, `demo.py`).
- [x] Khởi tạo tài liệu phân tích cấu trúc mạng và hàm loss: `DOC_SYSTEM_ARCHITECTURE.md`.
- [x] Khởi tạo checklist theo dõi tiến độ: `docs_expert/TODO_LIST.md`.
- [x] Khởi tạo tài liệu hướng dẫn tối cao: `docs_expert/FULL_PROJECT_GUIDE.md`.
- [x] Khởi tạo file lưu trữ prompt và xử lý lỗi: `docs_expert/PROMPTS_ARCHIVE.md`.

## BƯỚC 2: Tối ưu hóa Pipeline Tiền xử lý (Data Pipeline Optimization)
- [x] Cài đặt và cấu hình môi trường CUDA / PyTorch GPU / TensorRT.
- [x] Viết module `src/inference/preprocess_cuda.py` với class `GPUPayloadPreprocessor` sử dụng PyTorch CUDA tensors / TorchVision Operations để đưa toàn bộ khâu resize, padding, normalize, transpose lên GPU.
- [x] Cập nhật lý thuyết Zero-copy và GPU Memory Bound vào `docs_expert/FULL_PROJECT_GUIDE.md`.

## BƯỚC 3: Tối ưu hóa Đồ thị Tính toán & Lượng tử hóa (Model Optimization)
- [x] Viết script `scripts/export_tensorrt.py` để biên dịch mô hình PyTorch checkpoint `.pth` sang ONNX và cuối mang sang TensorRT Engine `.engine`.
- [x] Tích hợp Post-Training Quantization (PTQ) với Entropy Calibrator 2 dùng Calibration Dataset để thực hiện Mixed-Precision (kết hợp INT8 và FP16), đảm bảo dung lượng < 500KB và sai số < 0.5%.
- [x] Thiết kế cơ chế ghi và thực thi CUDA Graphs để loại bỏ hoàn toàn CPU Kernel Launch Overhead.
- [x] Viết module suy luận TensorRT tối ưu `src/inference/inference_trt.py`.

## BƯỚC 4: Viết Module Kiểm thử hiệu năng (Benchmarking & Evaluation)
- [x] Viết script kiểm thử `benchmark_hardware.py` đo đạc: FPS, Latency (ms), VRAM, RAM, CPU/GPU % Utilization trước và sau khi tối ưu.
- [x] Xuất bảng so sánh chi tiết và đính kèm báo cáo hiệu năng.
