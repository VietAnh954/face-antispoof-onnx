"""Real-time face anti-spoofing demo using TensorRT and GPU Preprocessing."""

# Load DLLs to prevent missing DLL issues in Windows
import onnxruntime
try:
    onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True)
except AttributeError:
    pass

import cv2
import numpy as np
import sys
import time
import argparse
import torch
from pathlib import Path

# Insert parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.preprocess_cuda import GPUPayloadPreprocessor
from src.inference.inference_trt import TensorRTEngineWrapper
from src.inference.inference import process_with_logits
from src.inference.system import get_cpu_info, get_gpu_info
from src.detection import load_detector, detect

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector_quantized.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_int8.engine"


def draw_info_overlay(display_frame, total_fps_history, infer_fps_history, cpu_info, gpu_info):
    avg_total_fps = sum(total_fps_history) / len(total_fps_history) if total_fps_history else 0
    avg_infer_fps = sum(infer_fps_history) / len(infer_fps_history) if infer_fps_history else 0

    info_y = 25
    line_height = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color_white = (255, 255, 255)
    color_cyan = (255, 255, 0)

    cv2.putText(
        display_frame,
        f"E2E FPS: {avg_total_fps:.1f} | Liveness Inference FPS: {avg_infer_fps:.1f}",
        (5, info_y),
        font,
        font_scale,
        color_cyan,
        thickness,
    )
    info_y += line_height

    cpu_lines = []
    max_chars_per_line = 55
    words = cpu_info.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                cpu_lines.append(current_line)
            current_line = word
    if current_line:
        cpu_lines.append(current_line)

    for i, cpu_line in enumerate(cpu_lines[:2]):
        cv2.putText(
            display_frame,
            f"CPU: {cpu_line}" if i == 0 else cpu_line,
            (5, info_y),
            font,
            font_scale,
            color_white,
            thickness,
        )
        info_y += line_height

    if gpu_info:
        gpu_lines = []
        words = gpu_info.split()
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    gpu_lines.append(current_line)
                current_line = word
        if current_line:
            gpu_lines.append(current_line)

        for i, gpu_line in enumerate(gpu_lines[:2]):
            cv2.putText(
                display_frame,
                f"GPU: {gpu_line}" if i == 0 else gpu_line,
                (5, info_y),
                font,
                font_scale,
                color_white,
                thickness,
            )
            info_y += line_height
    else:
        cv2.putText(
            display_frame,
            "GPU: No GPU detected",
            (5, info_y),
            font,
            font_scale,
            color_white,
            thickness,
        )
        info_y += line_height

    cv2.putText(
        display_frame,
        "Provider: TensorRT (CUDA Graphs & GPU Preprocessing)",
        (5, info_y),
        font,
        font_scale,
        color_white,
        thickness,
    )
    info_y += line_height
    cv2.putText(
        display_frame,
        "Press 'i' to toggle info | 'q' to quit",
        (5, info_y),
        font,
        0.4,
        (200, 200, 200),
        1,
    )


def process_camera(args, face_detector, liveness_engine, preprocessor, logit_threshold):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "Real-Time TensorRT Liveness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    show_info = True
    total_fps_history = []
    infer_fps_history = []

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()

    print("Controls:")
    print("  'q' - Quit")
    print("  'i' - Toggle info display")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection (YuNet)
        faces = detect(frame_rgb, face_detector, margin=args.margin)

        if faces:
            bboxes = []
            valid_faces = []
            
            for face in faces:
                bbox = face["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                x1, y1 = int(x), int(y)
                x2, y2 = x1 + int(w), y1 + int(h)
                
                # Verify bounds
                if w > 0 and h > 0:
                    bboxes.append((x1, y1, x2, y2))
                    valid_faces.append((face, (x1, y1, int(w), int(h))))

            if bboxes:
                # GPU-Accelerated Preprocessing & TensorRT Inference
                t_infer_start = time.perf_counter()
                
                with torch.no_grad():
                    # Crop, padding, resizing, and normalization completely on GPU
                    prep_gpu = preprocessor.preprocess_batch_gpu(
                        frame_rgb, bboxes, args.bbox_expansion_factor
                    )
                    # Inference using serialized TensorRT engine with CUDA Graphs
                    predictions = liveness_engine.infer(prep_gpu)
                
                t_infer_time = time.perf_counter() - t_infer_start
                current_infer_fps = 1.0 / t_infer_time if t_infer_time > 0 else 0
                infer_fps_history.append(current_infer_fps)
                if len(infer_fps_history) > 30:
                    infer_fps_history.pop(0)

                # Process results and draw boxes
                for (face, (x, y, w, h)), pred in zip(valid_faces, predictions):
                    result = process_with_logits(pred, logit_threshold)

                    color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        frame_time = time.time() - frame_start
        current_total_fps = 1.0 / frame_time if frame_time > 0 else 0
        total_fps_history.append(current_total_fps)

        if len(total_fps_history) > 30:
            total_fps_history.pop(0)

        display_frame = cv2.resize(
            frame, (640, 480), interpolation=cv2.INTER_AREA
        )

        if show_info:
            draw_info_overlay(
                display_frame, total_fps_history, infer_fps_history, cpu_info, gpu_info
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()


def process_image(args, face_detector, liveness_engine, preprocessor, logit_threshold):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from '{args.image}'", file=sys.stderr)
        exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detect(image_rgb, face_detector, margin=args.margin)

    if not faces:
        print("No faces detected in the image.")
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)

    bboxes = []
    valid_faces = []
    for face in faces:
        bbox = face["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + int(w), y1 + int(h)
        if w > 0 and h > 0:
            bboxes.append((x1, y1, x2, y2))
            valid_faces.append((x1, y1, int(w), int(h)))

    if bboxes:
        with torch.no_grad():
            # Run GPU preprocessing & TRT engine Wrapper
            prep_gpu = preprocessor.preprocess_batch_gpu(
                image_rgb, bboxes, args.bbox_expansion_factor
            )
            predictions = liveness_engine.infer(prep_gpu)

        for (x, y, w, h), pred in zip(valid_faces, predictions):
            result = process_with_logits(pred, logit_threshold)

            color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Face Anti-Spoofing using TensorRT & GPU Preprocessing")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (if not provided, uses webcam)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index to use (default: 0)"
    )
    parser.add_argument("--model_img_size", type=int, default=128)
    parser.add_argument("--bbox_expansion_factor", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--detector_model", type=str, default=str(DETECTOR_MODEL))
    parser.add_argument("--liveness_model", type=str, default=str(LIVENESS_MODEL))

    args = parser.parse_args()

    p = max(1e-6, min(1 - 1e-6, args.threshold))
    logit_threshold = np.log(p / (1 - p))

    print(f"[Main] Loading Face Detector: {args.detector_model}")
    face_detector = load_detector(args.detector_model, (320, 320))
    
    print(f"[Main] Loading TensorRT Engine: {args.liveness_model}")
    liveness_engine = TensorRTEngineWrapper(args.liveness_model, enable_cuda_graph=True)
    
    print("[Main] Initializing GPU Preprocessor...")
    preprocessor = GPUPayloadPreprocessor(device="cuda", model_img_size=args.model_img_size)

    if args.image is None:
        process_camera(args, face_detector, liveness_engine, preprocessor, logit_threshold)
    else:
        process_image(args, face_detector, liveness_engine, preprocessor, logit_threshold)
