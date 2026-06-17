# Kho Lưu Trữ Cú Pháp & Troubleshooting (PROMPTS ARCHIVE)

Tài liệu này lưu trữ các câu lệnh cài đặt, cấu hình, và giải quyết lỗi phát sinh trong quá trình tối ưu và triển khai phần cứng.

---

## 1. Cài Đặt Môi Mới & Giải Quyết Dependency

### 1.1. Nâng cấp PyTorch hỗ trợ CUDA 12.4 (Python 3.13)
Do phiên bản PyTorch mặc định khi cài đặt qua `requirements.txt` là bản CPU (`2.8.0+cpu`), ta cần nâng cấp lên bản hỗ trợ CUDA:
```bash
# Gỡ bỏ các thư viện PyTorch CPU cũ
pip uninstall -y torch torchvision

# Cài đặt PyTorch CUDA 12.4 tương thích với Python 3.13
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 1.2. Cài đặt Runtime Libs của NVIDIA & TensorRT 11+
```bash
# Cài đặt CUDA runtime, cuDNN và TensorRT Python API
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 tensorrt tensorrt-cu12 onnxsim
```

---

## 2. Giải Quyết Lỗi Triển Khai (Troubleshooting)

### 2.1. Lỗi thiếu DLL trong ONNX Runtime GPU
**Mô tả lỗi:**
`onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_12.dll" which is missing.`

**Cách xử lý:**
Thêm dòng sau vào mã nguồn chạy Python để tự động liên kết các DLL trong site-packages:
```python
import onnxruntime
try:
    onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True)
except AttributeError:
    pass
```

### 2.2. Lỗi `IInt8EntropyCalibrator2` không tồn tại trong TensorRT 11+
**Mô tả lỗi:**
`AttributeError: module 'tensorrt' has no attribute 'IInt8EntropyCalibrator2'`

**Lý do:**
TensorRT 11 đã loại bỏ hoàn toàn cơ chế lượng tử hóa ngầm định (Implicit Quantization) và các calibrator đi kèm. Nó chuyển hoàn toàn sang **Explicit Quantization** (lượng tử hóa tường minh dựa trên các node Q/DQ chèn vào ONNX).

**Cách xử lý:**
Sử dụng gói `onnxruntime.quantization` để thực hiện static quantization tạo file ONNX chứa Q/DQ nodes trước, sau đó nạp file ONNX này vào TensorRT 11 để tự động biên dịch sang INT8:
```python
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

quantize_static(
    model_input=onnx_file,
    model_output=qdq_onnx_file,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False}
)
```

### 2.3. Lỗi TensorRT Parser từ chối `zero_point` phi-không (Non-zero)
**Mô tả lỗi:**
`[TRT] [E] ModelImporter.cpp:151: ERROR: onnxOpImporters.cpp:1803 In function QuantDequantLinearHelper: [6] Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported.`

**Lý do:**
GPU Tensor Cores chỉ hỗ trợ lượng tử hóa đối xứng (Symmetric Quantization), yêu cầu `zero_point` bắt buộc bằng 0. Mặc định ONNX Runtime dùng asymmetric cho activation.

**Cách xử lý:**
Bắt buộc bật tính năng đối xứng trong `extra_options` của `quantize_static`:
```python
extra_options={"ActivationSymmetric": True, "WeightSymmetric": True}
```

### 2.4. Lỗi Parser báo kiểu dữ liệu Int32 của Bias không được hỗ trợ
**Mô tả lỗi:**
`[TRT] [E] ITensor::getDimensions: Error Code 4: API Usage Error (model.bn.bias_DequantizeLinear: input has type Int32 but must have type FP8, FP4, Int4, or Int8)`

**Lý do:**
ONNX Runtime tự động lượng tử hóa Bias sang Int32. Tuy nhiên, TensorRT parser không hỗ trợ node Dequantize cho Int32 bias.

**Cách xử lý:**
Tắt lượng tử hóa Bias để giữ nguyên Bias ở dạng float (FP32/FP16) trong file ONNX bằng cách cấu hình:
```python
extra_options={"QuantizeBias": False}
```

### 2.5. Lỗi thiếu `EXPLICIT_BATCH` và các check `platform_has_fast_fp16` trong TensorRT 11+
**Mô tả lỗi:**
- `AttributeError: type object 'NetworkDefinitionCreationFlag' has no attribute 'EXPLICIT_BATCH'`
- `AttributeError: 'Builder' object has no attribute 'platform_has_fast_fp16'`

**Lý do:**
TensorRT 11 mặc định chạy explicit batch và coi như mọi GPU hiện đại đều có các nhân Tensor Cores tăng tốc FP16/INT8, nên các flag và check này bị xóa bỏ.

**Cách xử lý:**
- Tạo network bằng cách truyền flag 0: `network = builder.create_network(0)`.
- Thiết lập trực tiếp cấu hình FP16/INT8: `config.set_flag(trt.BuilderFlag.FP16)` hoặc dựa trực tiếp vào kiểu dữ liệu trong file ONNX nạp vào.
