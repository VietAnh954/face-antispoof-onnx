"""Benchmark script that runs liveness pipelines and plots a beautiful visualization chart."""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

# Insert parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent if '__file__' in locals() else '.'))

from benchmark_hardware import (
    run_cpu_onnx_benchmark,
    run_gpu_onnx_benchmark,
    run_tensorrt_benchmark,
)

def plot_benchmarks(results, output_path="performance_comparison.png"):
    names = list(results.keys())
    latencies = [results[name]["e2e_ms"] for name in names]
    fps_vals = [results[name]["fps"] for name in names]
    
    # Modern styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette
    colors = ['#8A9A86', '#4F6F52', '#3559E0', '#00A9FF'] # Grey-green, Green, Blue, Light Blue
    
    # Latency Bar Chart (Lower is better)
    bars1 = ax1.bar(names, latencies, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax1.set_title("End-to-End Latency per Frame (ms)\n[Lower is Better]", fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.tick_params(axis='x', rotation=15, labelsize=9)
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(latencies)*0.02), f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    # FPS Bar Chart (Higher is better)
    bars2 = ax2.bar(names, fps_vals, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax2.set_title("Throughput (Frames Per Second)\n[Higher is Better]", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel("FPS", fontsize=11)
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    
    # Add values and speedup on top of bars
    baseline_fps = fps_vals[0]
    for bar in bars2:
        yval = bar.get_height()
        speedup = yval / baseline_fps
        label = f"{yval:.1f} FPS\n({speedup:.1f}x)" if speedup > 1.0 else f"{yval:.1f} FPS"
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(fps_vals)*0.02), label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[Visualizer] Beautiful performance chart saved to: {output_path}")

def main():
    # Model Paths
    onnx_model = "models/best/98.20/best_model.onnx"
    engine_fp16 = "models/best_model_fp16.engine"
    engine_int8 = "models/best_model_int8.engine"
    
    # Generate dummy data
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 300, 300)
    iters = 100
    
    print("=== Starting Visual Benchmark Compilation ===")
    results = {}
    
    # 1. Baseline CPU
    try:
        results["ORT CPU\n(Baseline)"] = run_cpu_onnx_benchmark(dummy_frame, bbox, onnx_model, iters)
    except Exception as e:
        print(f"Error benching CPU: {e}")
        
    # 2. ORT GPU
    try:
        results["ORT GPU\n(CUDA)"] = run_gpu_onnx_benchmark(dummy_frame, bbox, onnx_model, iters)
    except Exception as e:
        print(f"Error benching ORT GPU: {e}")
        
    # 3. TRT FP16
    try:
        results["TRT FP16\n(CUDA Graphs)"] = run_tensorrt_benchmark(dummy_frame, bbox, engine_fp16, True, iters)
    except Exception as e:
        print(f"Error benching TRT FP16: {e}")
        
    # 4. TRT INT8
    try:
        results["TRT INT8\n(CUDA Graphs)"] = run_tensorrt_benchmark(dummy_frame, bbox, engine_int8, True, iters)
    except Exception as e:
        print(f"Error benching TRT INT8: {e}")
        
    if results:
        # Plot and save
        plot_benchmarks(results, "performance_comparison.png")
        
        # Display text summary
        print("\n" + "="*80)
        print(f"{'Configuration':<25} | {'E2E Latency (ms)':<18} | {'Throughput (FPS)':<16}")
        print("-"*80)
        for name, metrics in results.items():
            clean_name = name.replace('\n', ' ')
            print(f"{clean_name:<25} | {metrics['e2e_ms']:18.3f} | {metrics['fps']:16.1f}")
        print("="*80)
    else:
        print("Error: No benchmarks completed.")

if __name__ == "__main__":
    from pathlib import Path
    main()
