"""Export PyTorch/ONNX model to TensorRT Engine using TensorRT 11+ explicit precision workflows."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.minifasv2.model import MultiFTNet
from src.minifasv2.config import get_kernel


class PyTorchCalibrationDataReader:
    """ONNX Runtime static quantization calibration data reader using PyTorch tensors."""

    def __init__(self, calibration_data: torch.Tensor, input_name: str):
        self.calibration_data = calibration_data # Shape: [N, C, H, W]
        self.input_name = input_name
        self.current_index = 0
        self.num_samples = len(calibration_data)

    def get_next(self) -> Optional[dict]:
        """Provides the next batch of calibration data."""
        if self.current_index >= self.num_samples:
            return None

        # Return a single sample (batch size 1) as numpy array
        batch = self.calibration_data[self.current_index : self.current_index + 1].cpu().numpy()
        self.current_index += 1
        return {self.input_name: batch}

    def rewind(self) -> None:
        """Resets reader to the beginning."""
        self.current_index = 0


def load_pytorch_model(checkpoint_path: str, input_size: int = 128) -> torch.nn.Module:
    """Loads MiniFASNetV2SE model from pth checkpoint."""
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    kernel_size = get_kernel(input_size, input_size)
    model = MultiFTNet(
        num_channels=3,
        num_classes=2,
        embedding_size=128,
        conv6_kernel=kernel_size,
    )

    # Clean state dict keys
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        new_key = new_key.replace("model.prob", "model.logits")
        new_key = new_key.replace(".prob", ".logits")
        new_key = new_key.replace("model.drop", "model.dropout")
        new_key = new_key.replace(".drop", ".dropout")
        new_key = new_key.replace("FTGenerator.ft.", "FTGenerator.fourier_transform.")
        new_key = new_key.replace("FTGenerator.ft", "FTGenerator.fourier_transform")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module, onnx_path: str, input_size: int = 128, half_precision: bool = False
) -> None:
    """Exports PyTorch model to ONNX format with dynamic batch size."""
    print(f"Exporting PyTorch model to ONNX at: {onnx_path} (FP16: {half_precision})")
    
    if half_precision:
        model = model.half().to("cuda")
        dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float16, device="cuda")
    else:
        model = model.to("cpu")
        dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    # Simplify ONNX
    import onnx
    import onnxsim
    onnx_model = onnx.load(onnx_path)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, onnx_path)
    print("[OK] ONNX model exported and simplified successfully.")


def generate_calibration_data(
    dataset_dir: Optional[str], num_samples: int = 256, input_size: int = 128
) -> torch.Tensor:
    """Generates calibration tensors. Reads from dataset if available, else synthesizes."""
    if dataset_dir and os.path.exists(dataset_dir):
        from PIL import Image
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ])
        
        image_paths = list(Path(dataset_dir).glob("**/*.*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        
        if len(image_paths) > 0:
            print(f"Loading {min(num_samples, len(image_paths))} calibration images from {dataset_dir}")
            samples = []
            for img_path in image_paths[:num_samples]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    samples.append(transform(img))
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
            
            if len(samples) > 0:
                return torch.stack(samples)
                
    # Fallback: Synthesize data using random noise smoothed with Gaussian filter
    print("No valid calibration images found. Synthesizing data using smoothed noise...")
    synthetic_samples = []
    for _ in range(num_samples):
        noise = torch.randn(3, input_size, input_size)
        smoothed = F.interpolate(noise.unsqueeze(0), size=(input_size // 4, input_size // 4), mode='bilinear', align_corners=False)
        smoothed = F.interpolate(smoothed, size=(input_size, input_size), mode='bicubic', align_corners=False).squeeze(0)
        normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
        synthetic_samples.append(normalized)
        
    return torch.stack(synthetic_samples)


def build_tensorrt_engine(onnx_path: str, engine_path: str) -> None:
    """Parses ONNX model (which contains precision definitions) and compiles it to a TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Create network definition (explicit batch is the default in TRT 10/11)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Set workspace memory limit to 1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error), file=sys.stderr)
            raise RuntimeError("Failed to parse ONNX model")
            
    print("ONNX model parsed successfully.")

    # Configure optimization profiles for dynamic batch size
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    # Profile format: name, min_shape, opt_shape, max_shape
    profile.set_shape(input_tensor.name, (1, 3, 128, 128), (1, 3, 128, 128), (16, 3, 128, 128))
    config.add_optimization_profile(profile)

    # In TensorRT 11, we create a STRONGLY TYPED engine based on the parsed ONNX network structure.
    # We do not need to configure global flags. TensorRT automatically uses the precision embedded in the ONNX file
    # (FP16 or Q/DQ nodes for INT8).

    # Compile the model into a serialized engine
    print(f"Building TensorRT Engine at: {engine_path} ... (This can take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"[OK] TensorRT Engine successfully compiled and saved to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch/ONNX model to TensorRT Engine with TensorRT 11+ explicit precision.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/98.20/best_model.pth",
        help="Path to .pth checkpoint or .onnx file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_model.engine",
        help="Path to save output .engine file",
    )
    parser.add_argument(
        "--input_size", type=int, default=128, help="Model input image size (default: 128)"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Enable FP16 precision"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Enable Mixed INT8/FP16 precision via static Q/DQ nodes"
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=None,
        help="Directory containing images for static Q/DQ calibration",
    )

    args = parser.parse_args()

    # Determine input type
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file '{args.model}' does not exist.", file=sys.stderr)
        sys.exit(1)

    temp_onnx = None
    temp_qdq_onnx = None
    
    # We use half precision in ONNX export if FP16 is selected and not doing INT8
    half_precision_onnx = args.fp16 and not args.int8

    if model_path.suffix == ".pth":
        print(f"Loading PyTorch model from checkpoint: {args.model}")
        pytorch_model = load_pytorch_model(str(model_path), args.input_size)
        
        # Export to a temporary ONNX file
        temp_onnx = str(model_path.with_suffix(".temp.onnx"))
        export_to_onnx(pytorch_model, temp_onnx, args.input_size, half_precision=half_precision_onnx)
        onnx_file = temp_onnx
    elif model_path.suffix == ".onnx":
        onnx_file = str(model_path)
    else:
        print(f"Error: Unsupported model format '{model_path.suffix}'. Use .pth or .onnx", file=sys.stderr)
        sys.exit(1)

    # Perform Q/DQ static quantization if INT8 is requested
    if args.int8:
        print("\n=== Performing Static Q/DQ Quantization via ONNX Runtime ===")
        from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
        
        calib_data = generate_calibration_data(args.calib_dir, num_samples=256, input_size=args.input_size)
        reader = PyTorchCalibrationDataReader(calib_data, "input")
        
        temp_qdq_onnx = onnx_file.replace(".onnx", ".qdq.temp.onnx")
        
        # Load ONNX model and find depthwise and other sensitive convolutions to exclude
        import onnx
        onnx_model = onnx.load(onnx_file)
        nodes_to_exclude = []
        for node in onnx_model.graph.node:
            if node.op_type == "Conv":
                # 1. Exclude depthwise convolutions (groups > 1)
                is_depthwise = False
                for attr in node.attribute:
                    if attr.name == "group" and attr.i > 1:
                        is_depthwise = True
                        break
                
                # 2. Exclude SE modules (squeeze-and-excitation)
                is_se = "se_module" in node.name
                
                # 3. Exclude first few layers (conv1 and conv_23 blocks)
                is_early = "/conv1/" in node.name or "/conv_23/" in node.name
                
                if is_depthwise or is_se or is_early:
                    nodes_to_exclude.append(node.name)
        
        print(f"Excluding {len(nodes_to_exclude)} sensitive/depthwise Conv nodes from quantization to preserve accuracy.")
        
        quantize_static(
            model_input=onnx_file,
            model_output=temp_qdq_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv"],
            per_channel=True,
            nodes_to_exclude=nodes_to_exclude,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False},
        )
        print(f"[OK] Static Q/DQ ONNX model saved to temporary file: {temp_qdq_onnx}")
        onnx_file = temp_qdq_onnx

    # Build TensorRT engine
    try:
        build_tensorrt_engine(
            onnx_path=onnx_file,
            engine_path=args.output,
        )
    finally:
        # Clean up temporary ONNX files
        if temp_onnx and os.path.exists(temp_onnx):
            os.remove(temp_onnx)
        if temp_qdq_onnx and os.path.exists(temp_qdq_onnx):
            os.remove(temp_qdq_onnx)
        print("Cleaned up temporary files.")

    print(f"\nDone! Engine compiled. Size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
