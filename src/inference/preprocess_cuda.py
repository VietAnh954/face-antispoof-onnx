"""GPU-accelerated preprocessing pipeline using PyTorch CUDA tensors."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
import numpy as np


class GPUPayloadPreprocessor:
    """GPU-accelerated preprocessor for face liveness detection.
    
    Performs cropping, padding, resizing, and normalization entirely on the GPU
    to eliminate CPU bottlenecks and minimize host-to-device memory transfer overheads.
    """

    def __init__(self, device: str = "cuda", model_img_size: int = 128) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_img_size = model_img_size
        print(f"[GPUPayloadPreprocessor] Initialized on device: {self.device}")

    def safe_pad_reflect(self, tensor: torch.Tensor, pad: List[int]) -> torch.Tensor:
        """Pads tensor with reflection padding. Fallback to replicate mode if pad >= size."""
        # pad format: [left, right, top, bottom]
        _, h, w = tensor.shape
        left, right, top, bottom = pad

        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return tensor

        # PyTorch reflect padding fails if pad >= dimension size.
        if left >= w or right >= w or top >= h or bottom >= h:
            return F.pad(tensor, pad, mode="replicate")
        
        return F.pad(tensor, pad, mode="reflect")

    def crop_gpu(
        self, img_tensor: torch.Tensor, bbox: Tuple[int, int, int, int], bbox_expansion_factor: float
    ) -> torch.Tensor:
        """Extracts square face crop from bbox with expansion. Pads edges with reflection."""
        # Input img_tensor is expected to be [H, W, C] on GPU or CPU
        original_height, original_width = img_tensor.shape[:2]
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")

        max_dim = max(w, h)
        center_x = x1 + w / 2.0
        center_y = y1 + h / 2.0

        crop_size = int(max_dim * bbox_expansion_factor)
        x = int(center_x - crop_size / 2.0)
        y = int(center_y - crop_size / 2.0)

        crop_x1 = max(0, x)
        crop_y1 = max(0, y)
        crop_x2 = min(original_width, x + crop_size)
        crop_y2 = min(original_height, y + crop_size)

        top_pad = max(0, -y)
        left_pad = max(0, -x)
        bottom_pad = max(0, (y + crop_size) - original_height)
        right_pad = max(0, (x + crop_size) - original_width)

        # Slice the face crop
        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            img_crop = img_tensor[crop_y1:crop_y2, crop_x1:crop_x2, :]
        else:
            img_crop = torch.zeros((0, 0, 3), dtype=img_tensor.dtype, device=img_tensor.device)

        # Permute HWC to CHW for PyTorch padding & resizing
        img_crop = img_crop.permute(2, 0, 1) # [C, H, W]

        # Pad using reflection
        pad = [left_pad, right_pad, top_pad, bottom_pad]
        result = self.safe_pad_reflect(img_crop, pad)

        return result

    def preprocess_gpu(self, crop_tensor: torch.Tensor) -> torch.Tensor:
        """Resizes the square crop to model size, normalizes, and returns CHW."""
        # crop_tensor is [C, H_crop, W_crop] on GPU
        # Add batch dimension for interpolate: [1, C, H_crop, W_crop]
        tensor = crop_tensor.unsqueeze(0).float()

        # Resize to model_img_size x model_img_size
        tensor = F.interpolate(
            tensor,
            size=(self.model_img_size, self.model_img_size),
            mode="bicubic",
            align_corners=False,
        )

        # Normalize to [0, 1]
        tensor = tensor / 255.0

        # Remove batch dimension: [C, model_img_size, model_img_size]
        return tensor.squeeze(0)

    def preprocess_batch_gpu(
        self,
        img: Union[np.ndarray, torch.Tensor],
        bboxes: List[Tuple[int, int, int, int]],
        bbox_expansion_factor: float = 1.5,
    ) -> torch.Tensor:
        """Preprocesses multiple face crops from an image into a batched GPU tensor.
        
        Args:
            img: Raw input frame (either numpy.ndarray on CPU or torch.Tensor on GPU/CPU)
            bboxes: List of bboxes in (x1, y1, x2, y2) format
            bbox_expansion_factor: Factor to expand the face bounding boxes
            
        Returns:
            Batched preprocessed tensor of shape [N, 3, model_img_size, model_img_size] on GPU
        """
        if not bboxes:
            raise ValueError("Bboxes list cannot be empty")

        # Transfer image to GPU if it is a numpy array
        if isinstance(img, np.ndarray):
            img_gpu = torch.from_numpy(img).to(device=self.device, non_blocking=True)
        else:
            img_gpu = img.to(device=self.device, non_blocking=True)

        batch_tensors = []
        for bbox in bboxes:
            crop_t = self.crop_gpu(img_gpu, bbox, bbox_expansion_factor)
            prep_t = self.preprocess_gpu(crop_t)
            batch_tensors.append(prep_t)

        # Stack into a batch: [N, 3, model_img_size, model_img_size]
        batched_tensor = torch.stack(batch_tensors, dim=0)
        return batched_tensor
