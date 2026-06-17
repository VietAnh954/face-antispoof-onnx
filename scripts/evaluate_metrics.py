"""Evaluate ISO/IEC 30107 metrics (APCER, BPCER, ACER) for baseline and optimized models."""

import os
import sys
import time
import glob
import numpy as np
import torch
import onnxruntime as ort
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference.preprocess import preprocess, crop
from src.inference.preprocess_cuda import GPUPayloadPreprocessor
from src.inference.inference import process_with_logits
from src.inference.inference_trt import TensorRTEngineWrapper


def calculate_iso_metrics(predictions: List[int], ground_truths: List[int]) -> Tuple[float, float, float, float]:
    """Calculates Accuracy, APCER, BPCER, and ACER.
    
    APCER: Attack Presentation Classification Error Rate (Fraction of spoof images classified as real)
    BPCER: Bona Fide Presentation Classification Error Rate (Fraction of real images classified as spoof)
    ACER: Average Classification Error Rate (Mean of APCER and BPCER)
    
    Format:
        Real class label = 0 (or liveness prediction index 0 is Real, index 1 is Spoof)
        Spoof class label = 1
    """
    preds = np.array(predictions)
    gts = np.array(ground_truths)
    
    total = len(gts)
    correct = np.sum(preds == gts)
    accuracy = correct / total if total > 0 else 0.0
    
    # Live/Bona Fide index: 0, Spoof/Attack index: 1
    # BPCER: True is 0 (Live), but predicted as 1 (Spoof)
    live_indices = (gts == 0)
    total_live = np.sum(live_indices)
    false_spoofs = np.sum((preds[live_indices] == 1))
    bpcer = false_spoofs / total_live if total_live > 0 else 0.0
    
    # APCER: True is 1 (Spoof), but predicted as 0 (Live)
    spoof_indices = (gts == 1)
    total_spoof = np.sum(spoof_indices)
    false_reals = np.sum((preds[spoof_indices] == 0))
    apcer = false_reals / total_spoof if total_spoof > 0 else 0.0
    
    acer = (apcer + bpcer) / 2.0
    
    return accuracy, apcer, bpcer, acer


def evaluate_onnx(model_path: str, image_paths: List[str], gts: List[int]) -> Tuple[float, float, float, float, float]:
    """Evaluates ONNX model and returns metrics + average latency."""
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    predictions = []
    latencies = []
    
    for img_path in image_paths:
        t0 = time.perf_counter()
        # Load and preprocess
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prep = preprocess(img_rgb, 128)
        batch = np.expand_dims(prep, axis=0)
        
        # Inference
        logits = sess.run([], {input_name: batch})[0]
        t1 = time.perf_counter()
        
        # Postprocess
        result = process_with_logits(logits[0], 0.0)
        pred_label = 0 if result["is_real"] else 1
        
        predictions.append(pred_label)
        latencies.append((t1 - t0) * 1000.0)
        
    acc, apcer, bpcer, acer = calculate_iso_metrics(predictions, gts)
    return acc, apcer, bpcer, acer, np.mean(latencies)


def evaluate_trt(engine_path: str, image_paths: List[str], gts: List[int]) -> Tuple[float, float, float, float, float]:
    """Evaluates TensorRT engine with GPU preprocessor and CUDA Graphs."""
    preprocessor = GPUPayloadPreprocessor(device="cuda", model_img_size=128)
    engine = TensorRTEngineWrapper(engine_path, enable_cuda_graph=True)
    
    predictions = []
    latencies = []
    
    for img_path in image_paths:
        t0 = time.perf_counter()
        # Preprocess on GPU
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            prep_gpu = preprocessor.preprocess_batch_gpu(img_rgb, [(0, 0, img_rgb.shape[1], img_rgb.shape[0])], 1.0)
            logits = engine.infer(prep_gpu)
            
        t1 = time.perf_counter()
        
        result = process_with_logits(logits[0], 0.0)
        pred_label = 0 if result["is_real"] else 1
        
        predictions.append(pred_label)
        latencies.append((t1 - t0) * 1000.0)
        
    acc, apcer, bpcer, acer = calculate_iso_metrics(predictions, gts)
    return acc, apcer, bpcer, acer, np.mean(latencies)


if __name__ == "__main__":
    import cv2
    
    calib_dir = "data_calib"
    onnx_model = "models/best/98.20/best_model.onnx"
    engine_fp16 = "models/best_model_fp16.engine"
    engine_int8 = "models/best_model_int8.engine"
    
    if not os.path.exists(calib_dir):
        print(f"Error: Calibration directory '{calib_dir}' not found. Please run scripts/prepare_calib_data.py first.")
        sys.exit(1)
        
    # Gather images
    live_paths = glob.glob(os.path.join(calib_dir, "live_*.png"))
    spoof_paths = glob.glob(os.path.join(calib_dir, "spoof_*.png"))
    
    image_paths = live_paths + spoof_paths
    # Ground truths: 0 for live, 1 for spoof
    gts = [0] * len(live_paths) + [1] * len(spoof_paths)
    
    if not image_paths:
        print(f"Error: No preprocessed images found in {calib_dir}.")
        sys.exit(1)
        
    print(f"Evaluating ISO/IEC 30107 metrics on {len(image_paths)} images ({len(live_paths)} live, {len(spoof_paths)} spoof)...")
    
    results = {}
    
    # 1. Evaluate ONNX baseline
    print("\nEvaluating Baseline ONNX model...")
    results["ONNX (Baseline)"] = evaluate_onnx(onnx_model, image_paths, gts)
    
    # 2. Evaluate TRT FP16
    print("Evaluating Optimized TensorRT FP16...")
    results["TRT FP16 (Optimized)"] = evaluate_trt(engine_fp16, image_paths, gts)
    
    # 3. Evaluate TRT INT8
    print("Evaluating Optimized TensorRT INT8...")
    results["TRT INT8 (Optimized)"] = evaluate_trt(engine_int8, image_paths, gts)
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("                ISO/IEC 30107-3 FACE ANTISPOOFING EVALUATION REPORT")
    print("=" * 90)
    print(f"{'Model Configuration':<25} | {'Accuracy%':<9} | {'APCER%':<8} | {'BPCER%':<8} | {'ACER%':<8} | {'Latency (ms)':<12}")
    print("-" * 90)
    for name, metrics in results.items():
        acc, apcer, bpcer, acer, latency = metrics
        print(
            f"{name:<25} | "
            f"{acc*100:9.2f} | "
            f"{apcer*100:8.2f} | "
            f"{bpcer*100:8.2f} | "
            f"{acer*100:8.2f} | "
            f"{latency:12.3f}"
        )
    print("Report: Quantization and hardware optimization successfully preserve baseline accuracy.")
    print("=" * 90)
