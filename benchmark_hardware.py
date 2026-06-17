"""Benchmark script to evaluate throughput, latency, VRAM, and CPU/GPU utilization."""

import os
import sys
import time
import argparse
import numpy as np
import torch
import psutil
import onnxruntime as ort
from typing import Dict, List, Tuple

# Set directory path
sys.path.insert(0, os.path.dirname(__file__))

# Import preprocessors and wrappers
from src.inference.preprocess import preprocess, crop
from src.inference.preprocess_cuda import GPUPayloadPreprocessor
from src.inference.inference import infer, process_with_logits
from src.inference.inference_trt import TensorRTEngineWrapper


def get_cpu_utilization() -> float:
    """Returns overall CPU utilization percentage."""
    return psutil.cpu_percent(interval=None)


def get_gpu_metrics() -> Tuple[float, float]:
    """Returns GPU utilization and VRAM usage in MB using PyTorch/GPUtil if available."""
    gpu_util = 0.0
    vram_used = 0.0
    
    # Try GPUtil first
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus and len(gpus) > 0:
            gpu = gpus[0]
            gpu_util = gpu.load * 100.0
            vram_used = gpu.memoryUsed
            return gpu_util, vram_used
    except Exception:
        pass
        
    # Fallback to PyTorch VRAM query
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 * 1024)
        
    return gpu_util, vram_used


def run_cpu_onnx_benchmark(
    dummy_frame: np.ndarray, bbox: tuple, model_path: str, num_iters: int = 100
) -> Dict[str, float]:
    """Benchmarks baseline CPU + ONNX Runtime CPU pipeline."""
    print("\nBenchmarking Pipeline 1: CPU Preprocessing + ONNX CPU ...")
    
    # Load ONNX model on CPU
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    # Warmup
    for _ in range(5):
        cropped = crop(dummy_frame, bbox, 1.5)
        prep = preprocess(cropped, 128)
        batch = np.expand_dims(prep, axis=0)
        _ = sess.run([], {input_name: batch})[0]

    # Benchmark loop
    latencies_prep = []
    latencies_infer = []
    latencies_post = []
    latencies_e2e = []
    
    cpu_utils = []
    
    for _ in range(num_iters):
        t_start = time.perf_counter()
        
        # 1. Preprocess (Crop + resize + transpose + normalize on CPU)
        t_prep_start = time.perf_counter()
        cropped = crop(dummy_frame, bbox, 1.5)
        prep = preprocess(cropped, 128)
        batch = np.expand_dims(prep, axis=0)
        t_prep_end = time.perf_counter()
        
        # 2. Inference (ONNX Run on CPU)
        t_infer_start = time.perf_counter()
        logits = sess.run([], {input_name: batch})[0]
        t_infer_end = time.perf_counter()
        
        # 3. Postprocess (Sigmoid / classification threshold)
        t_post_start = time.perf_counter()
        _ = process_with_logits(logits[0], 0.0)
        t_post_end = time.perf_counter()
        
        t_end = time.perf_counter()
        
        latencies_prep.append((t_prep_end - t_prep_start) * 1000.0)
        latencies_infer.append((t_infer_end - t_infer_start) * 1000.0)
        latencies_post.append((t_post_end - t_post_start) * 1000.0)
        latencies_e2e.append((t_end - t_start) * 1000.0)
        
        cpu_utils.append(get_cpu_utilization())

    return {
        "prep_ms": np.mean(latencies_prep),
        "infer_ms": np.mean(latencies_infer),
        "post_ms": np.mean(latencies_post),
        "e2e_ms": np.mean(latencies_e2e),
        "fps": 1000.0 / np.mean(latencies_e2e),
        "cpu_util": np.mean(cpu_utils),
        "gpu_util": 0.0,
        "vram_mb": 0.0,
    }


def run_gpu_onnx_benchmark(
    dummy_frame: np.ndarray, bbox: tuple, model_path: str, num_iters: int = 100
) -> Dict[str, float]:
    """Benchmarks CPU Preprocessing + ONNX Runtime GPU pipeline."""
    print("Benchmarking Pipeline 2: CPU Preprocessing + ONNX GPU (CUDA) ...")
    
    # Load ONNX model on GPU
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    # Warmup
    for _ in range(5):
        cropped = crop(dummy_frame, bbox, 1.5)
        prep = preprocess(cropped, 128)
        batch = np.expand_dims(prep, axis=0)
        _ = sess.run([], {input_name: batch})[0]

    latencies_prep = []
    latencies_infer = []
    latencies_post = []
    latencies_e2e = []
    
    cpu_utils = []
    gpu_utils = []
    vrams = []
    
    for _ in range(num_iters):
        t_start = time.perf_counter()
        
        # 1. Preprocess on CPU
        t_prep_start = time.perf_counter()
        cropped = crop(dummy_frame, bbox, 1.5)
        prep = preprocess(cropped, 128)
        batch = np.expand_dims(prep, axis=0)
        t_prep_end = time.perf_counter()
        
        # 2. Inference on GPU
        t_infer_start = time.perf_counter()
        logits = sess.run([], {input_name: batch})[0]
        t_infer_end = time.perf_counter()
        
        # 3. Postprocess
        t_post_start = time.perf_counter()
        _ = process_with_logits(logits[0], 0.0)
        t_post_end = time.perf_counter()
        
        t_end = time.perf_counter()
        
        latencies_prep.append((t_prep_end - t_prep_start) * 1000.0)
        latencies_infer.append((t_infer_end - t_infer_start) * 1000.0)
        latencies_post.append((t_post_end - t_post_start) * 1000.0)
        latencies_e2e.append((t_end - t_start) * 1000.0)
        
        cpu_utils.append(get_cpu_utilization())
        gpu_util, vram = get_gpu_metrics()
        gpu_utils.append(gpu_util)
        vrams.append(vram)

    return {
        "prep_ms": np.mean(latencies_prep),
        "infer_ms": np.mean(latencies_infer),
        "post_ms": np.mean(latencies_post),
        "e2e_ms": np.mean(latencies_e2e),
        "fps": 1000.0 / np.mean(latencies_e2e),
        "cpu_util": np.mean(cpu_utils),
        "gpu_util": np.mean(gpu_utils),
        "vram_mb": np.mean(vrams),
    }


def run_tensorrt_benchmark(
    dummy_frame: np.ndarray,
    bbox: tuple,
    engine_path: str,
    enable_cuda_graph: bool = True,
    num_iters: int = 100,
) -> Dict[str, float]:
    """Benchmarks fully optimized GPU Preprocessing + TensorRT Engine (with optional CUDA Graph)."""
    graph_str = "with CUDA Graph" if enable_cuda_graph else "without CUDA Graph"
    print(f"Benchmarking Pipeline 3: GPU Preprocessing + TensorRT Engine ({graph_str}) ...")
    
    # Initialize Preprocessor & Engine
    preprocessor = GPUPayloadPreprocessor(device="cuda", model_img_size=128)
    engine = TensorRTEngineWrapper(engine_path, enable_cuda_graph=enable_cuda_graph)
    
    # Upload frame to GPU once (zero copy simulation)
    frame_gpu = torch.from_numpy(dummy_frame).cuda()

    # Warmup
    for _ in range(5):
        prep_gpu = preprocessor.preprocess_batch_gpu(frame_gpu, [bbox], 1.5)
        _ = engine.infer(prep_gpu)

    latencies_prep = []
    latencies_infer = []
    latencies_post = []
    latencies_e2e = []
    
    cpu_utils = []
    gpu_utils = []
    vrams = []
    
    for _ in range(num_iters):
        t_start = time.perf_counter()
        
        # 1. Preprocess entirely on GPU
        t_prep_start = time.perf_counter()
        prep_gpu = preprocessor.preprocess_batch_gpu(frame_gpu, [bbox], 1.5)
        t_prep_end = time.perf_counter()
        
        # 2. Inference on TensorRT Engine
        t_infer_start = time.perf_counter()
        logits = engine.infer(prep_gpu)
        t_infer_end = time.perf_counter()
        
        # 3. Postprocess
        t_post_start = time.perf_counter()
        _ = process_with_logits(logits[0], 0.0)
        t_post_end = time.perf_counter()
        
        t_end = time.perf_counter()
        
        latencies_prep.append((t_prep_end - t_prep_start) * 1000.0)
        latencies_infer.append((t_infer_end - t_infer_start) * 1000.0)
        latencies_post.append((t_post_end - t_post_start) * 1000.0)
        latencies_e2e.append((t_end - t_start) * 1000.0)
        
        cpu_utils.append(get_cpu_utilization())
        gpu_util, vram = get_gpu_metrics()
        gpu_utils.append(gpu_util)
        vrams.append(vram)

    return {
        "prep_ms": np.mean(latencies_prep),
        "infer_ms": np.mean(latencies_infer),
        "post_ms": np.mean(latencies_post),
        "e2e_ms": np.mean(latencies_e2e),
        "fps": 1000.0 / np.mean(latencies_e2e),
        "cpu_util": np.mean(cpu_utils),
        "gpu_util": np.mean(gpu_utils),
        "vram_mb": np.mean(vrams),
    }


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Outputs a nice comparative table in Markdown format."""
    print("\n" + "=" * 90)
    print("                      HARDWARE BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    
    header = f"{'Pipeline / Metrics':<32} | {'Prep (ms)':<9} | {'Infer (ms)':<10} | {'E2E (ms)':<9} | {'FPS':<7} | {'CPU%':<5} | {'GPU%':<5} | {'VRAM(MB)':<8}"
    print(header)
    print("-" * len(header))
    
    for name, metrics in results.items():
        print(
            f"{name:<32} | "
            f"{metrics['prep_ms']:9.3f} | "
            f"{metrics['infer_ms']:10.3f} | "
            f"{metrics['e2e_ms']:9.3f} | "
            f"{metrics['fps']:7.1f} | "
            f"{metrics['cpu_util']:5.1f} | "
            f"{metrics['gpu_util']:5.1f} | "
            f"{metrics['vram_mb']:8.1f}"
        )
    print("=" * 90)
    
    # Calculate speedups
    baseline_fps = results["Baseline (CPU + ORT CPU)"]["fps"]
    ort_gpu_fps = results["Baseline (CPU + ORT GPU)"]["fps"]
    fp16_fps = results["Optimized TRT FP16 + CUDA Graph"]["fps"]
    int8_fps = results["Optimized TRT INT8 + CUDA Graph"]["fps"]
    
    print("\nPerformance Gains (Speedup Factors):")
    print(f"  - ORT GPU vs ORT CPU: {ort_gpu_fps / baseline_fps:.2f}x")
    print(f"  - TRT FP16 (GPU Preprocess + CUDA Graph) vs ORT CPU: {fp16_fps / baseline_fps:.2f}x")
    print(f"  - TRT FP16 (GPU Preprocess + CUDA Graph) vs ORT GPU: {fp16_fps / ort_gpu_fps:.2f}x")
    print(f"  - TRT INT8 (GPU Preprocess + CUDA Graph) vs ORT CPU: {int8_fps / baseline_fps:.2f}x")
    print(f"  - TRT INT8 (GPU Preprocess + CUDA Graph) vs ORT GPU: {int8_fps / ort_gpu_fps:.2f}x")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run liveness detection pipeline benchmarks.")
    parser.add_argument(
        "--onnx",
        type=str,
        default="models/best/98.20/best_model.onnx",
        help="Path to baseline ONNX model",
    )
    parser.add_argument(
        "--engine_fp16",
        type=str,
        default="models/best_model_fp16.engine",
        help="Path to compiled FP16 TRT engine",
    )
    parser.add_argument(
        "--engine_int8",
        type=str,
        default="models/best_model_int8.engine",
        help="Path to compiled INT8 TRT engine",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    # Generate synthetic image for benchmarking
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 300, 300) # [x1, y1, x2, y2]
    
    results = {}
    
    # 1. Baseline CPU + ORT CPU
    try:
        results["Baseline (CPU + ORT CPU)"] = run_cpu_onnx_benchmark(
            dummy_frame, bbox, args.onnx, args.iters
        )
    except Exception as e:
        print(f"Failed to benchmark Baseline CPU + ORT CPU: {e}")
        
    # 2. Baseline CPU + ORT GPU
    try:
        results["Baseline (CPU + ORT GPU)"] = run_gpu_onnx_benchmark(
            dummy_frame, bbox, args.onnx, args.iters
        )
    except Exception as e:
        print(f"Failed to benchmark Baseline CPU + ORT GPU: {e}")

    # 3. Optimized TRT FP16 + CUDA Graph
    try:
        results["Optimized TRT FP16 + CUDA Graph"] = run_tensorrt_benchmark(
            dummy_frame, bbox, args.engine_fp16, enable_cuda_graph=True, num_iters=args.iters
        )
    except Exception as e:
        print(f"Failed to benchmark TRT FP16 + CUDA Graph: {e}")

    # 4. Optimized TRT INT8 + CUDA Graph
    try:
        results["Optimized TRT INT8 + CUDA Graph"] = run_tensorrt_benchmark(
            dummy_frame, bbox, args.engine_int8, enable_cuda_graph=True, num_iters=args.iters
        )
    except Exception as e:
        print(f"Failed to benchmark TRT INT8 + CUDA Graph: {e}")

    if results:
        print_comparison_table(results)
    else:
        print("No benchmarks completed successfully.")
