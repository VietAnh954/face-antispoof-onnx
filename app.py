"""FastAPI Web Server for Face Registration, Recognition, and Liveness Detection."""

# Load DLLs to prevent missing DLL issues in Windows
import onnxruntime
try:
    onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True)
except AttributeError:
    pass

import sys
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
import json
import base64
import numpy as np
import cv2
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Insert parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.preprocess_cuda import GPUPayloadPreprocessor
from src.inference.inference_trt import TensorRTEngineWrapper
from src.inference.inference import process_with_logits

# Setup FastAPI App
app = FastAPI(title="FaceID and Liveness Demo")

# Directory setups
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Database of registered faces
DATABASE_PATH = BASE_DIR / "models" / "registered_faces.json"
os.makedirs(DATABASE_PATH.parent, exist_ok=True)

registered_faces = {}
if DATABASE_PATH.exists():
    try:
        with open(DATABASE_PATH, "r", encoding="utf-8") as f:
            registered_faces = json.load(f)
        print(f"[Main] Loaded {len(registered_faces)} registered faces.")
    except Exception as e:
        print(f"[Error] Failed to load database: {e}")

def save_database():
    try:
        with open(DATABASE_PATH, "w", encoding="utf-8") as f:
            json.dump(registered_faces, f, indent=4)
    except Exception as e:
        print(f"[Error] Failed to save database: {e}")

# Global AI Models initialization
print("[Main] Initializing Face Detector (YuNet)...")
face_detector = cv2.FaceDetectorYN.create("models/detector_quantized.onnx", "", (320, 320), score_threshold=0.6, nms_threshold=0.3)

print("[Main] Initializing Face Recognizer (SFace)...")
face_recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

print("[Main] Initializing Liveness Engine (TensorRT FP16)...")
liveness_engine = TensorRTEngineWrapper("models/best_model_fp16.engine", enable_cuda_graph=True)

print("[Main] Initializing GPU Preprocessor...")
preprocessor = GPUPayloadPreprocessor(device="cuda", model_img_size=128)

@app.get("/")
async def get_index():
    """Serves the main frontend page."""
    return FileResponse(STATIC_DIR / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Websocket connection for real-time video frames stream."""
    await websocket.accept()
    print("[WS] Client connected.")
    
    try:
        while True:
            # Receive message from browser
            message = await websocket.receive_json()
            msg_type = message.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
                
            frame_data = message.get("image") # base64 string
            if not frame_data:
                continue
                
            # Decode base64 image
            try:
                header, encoded = frame_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
            except Exception as e:
                print(f"[WS Error] Failed to decode frame: {e}")
                continue
                
            img_h, img_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 1. Face Detection (YuNet)
            face_detector.setInputSize((img_w, img_h))
            _, faces = face_detector.detect(img_bgr)
            
            if faces is not None and len(faces) > 0:
                face = faces[0] # Focus on the first detected face
                bbox = face[:4].astype(int)
                x, y, w, h = bbox
                
                # Check for invalid bbox
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    await websocket.send_json({"type": "result", "detected": False})
                    continue
                
                # 2. Face Alignment & Embedding Extraction
                aligned_face = face_recognizer.alignCrop(img_bgr, face)
                feat = face_recognizer.feature(aligned_face) # shape: (1, 128)
                feat_vec = feat[0]
                
                # Handle registration request
                if msg_type == "register":
                    name = message.get("name")
                    if name:
                        # Check for biometric duplication
                        duplicate_name = None
                        max_dup_similarity = 0.0
                        for existing_name, reg_feat_list in registered_faces.items():
                            reg_feat = np.array(reg_feat_list)
                            similarity = np.dot(feat_vec, reg_feat) / (np.linalg.norm(feat_vec) * np.linalg.norm(reg_feat))
                            if similarity > max_dup_similarity:
                                max_dup_similarity = similarity
                                if similarity >= 0.60:
                                    duplicate_name = existing_name
                        
                        if duplicate_name:
                            print(f"[Database] Rejecting registration for '{name}'. Face already registered as '{duplicate_name}' (similarity: {max_dup_similarity:.3f})")
                            await websocket.send_json({
                                "type": "register_status", 
                                "status": "error", 
                                "message": f"Face already registered as '{duplicate_name}'!"
                            })
                        else:
                            registered_faces[name] = feat_vec.tolist()
                            save_database()
                            print(f"[Database] Registered face for: {name}")
                            await websocket.send_json({"type": "register_status", "status": "success", "name": name})
                    else:
                        await websocket.send_json({"type": "register_status", "status": "error", "message": "Name is empty"})
                    continue
                
                # 3. Face Recognition (Cosine Similarity)
                matched_name = "Unknown"
                max_similarity = 0.0
                
                for name, reg_feat_list in registered_faces.items():
                    reg_feat = np.array(reg_feat_list)
                    # Cosine similarity formula: dot(A, B) / (norm(A) * norm(B))
                    similarity = np.dot(feat_vec, reg_feat) / (np.linalg.norm(feat_vec) * np.linalg.norm(reg_feat))
                    if similarity > max_similarity:
                        max_similarity = similarity
                        
                # SFace cosine similarity threshold is typically 0.363
                if max_similarity >= 0.363:
                    matched_name = matched_name if max_similarity < 0.363 else list(registered_faces.keys())[
                        [np.dot(feat_vec, np.array(v)) / (np.linalg.norm(feat_vec) * np.linalg.norm(np.array(v))) for v in registered_faces.values()].index(max_similarity)
                    ]
                
                # 4. Liveness Classification (TensorRT + GPU Preprocessing)
                with torch.no_grad():
                    # Process entirely on GPU
                    prep_gpu = preprocessor.preprocess_batch_gpu(
                        img_rgb, [(x, y, x + w, y + h)], 1.5
                    )
                    logits = liveness_engine.infer(prep_gpu)
                    
                result = process_with_logits(logits[0], 0.0) # threshold 0.5 (logit 0.0)
                is_real = result["is_real"]
                
                # 5. Send results back
                await websocket.send_json({
                    "type": "result",
                    "detected": True,
                    "name": matched_name,
                    "similarity": float(max_similarity),
                    "liveness": "REAL" if is_real else "SPOOF",
                    "bbox": [int(x), int(y), int(w), int(h)]
                })
            else:
                # No face detected
                await websocket.send_json({
                    "type": "result",
                    "detected": False
                })
                
    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS Error] Exception occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
