"""TensorRT Native Inference engine wrapper with CUDA Graphs support."""

import os
import sys
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import tensorrt as trt

# Load DLLs to prevent missing DLL issues in Windows
import onnxruntime
try:
    onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True)
except AttributeError:
    pass


class TensorRTEngineWrapper:
    """High-performance wrapper for executing a serialized TensorRT engine.
    
    Includes native support for GPU preprocessing integration, dynamic shapes,
    and CUDA Graphs execution to eliminate CPU kernel launch overheads.
    """

    def __init__(self, engine_path: str, enable_cuda_graph: bool = True) -> None:
        self.engine_path = engine_path
        self.enable_cuda_graph = enable_cuda_graph
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load and deserialize engine
        print(f"[TensorRT] Loading engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        
        # Parse profiles and tensor I/O
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        self.tensor_dtypes = {}
        
        # Map TensorRT DataType to PyTorch dtype
        self.trt_to_torch_dtype = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
        }
        
        self._parse_engine_tensors()
        
        # CUDA Graph components
        self.cuda_graph = None
        self.graph_input = None
        self.graph_output = None
        self.graph_batch_size = 0
        
        # Warmup and record CUDA graph for default batch size 1
        if self.enable_cuda_graph:
            try:
                self._setup_cuda_graph(batch_size=1)
            except Exception as e:
                print(f"[Warning] Failed to initialize CUDA Graphs: {e}. Falling back to standard execution.")
                self.enable_cuda_graph = False

    def _parse_engine_tensors(self) -> None:
        """Queries the deserialized engine to determine input/output tensor names, shapes, and types."""
        # Query total number of IO tensors (new in TRT 10/11)
        num_io_tensors = self.engine.num_io_tensors
        
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            torch_dtype = self.trt_to_torch_dtype.get(dtype, torch.float32)
            self.tensor_dtypes[name] = torch_dtype
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
                print(f"  [Input] Tensor: '{name}', Shape: {shape}, Type: {torch_dtype}")
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.output_shapes[name] = shape
                print(f"  [Output] Tensor: '{name}', Shape: {shape}, Type: {torch_dtype}")

    def _setup_cuda_graph(self, batch_size: int = 1) -> None:
        """Captures and records a CUDA Graph for execution with a fixed batch size."""
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        
        # Resolve concrete shapes
        input_shape = (batch_size, *self.input_shapes[input_name][1:])
        output_shape = (batch_size, *self.output_shapes[output_name][1:])
        
        input_dtype = self.tensor_dtypes[input_name]
        output_dtype = self.tensor_dtypes[output_name]
        
        # Allocate CUDA device memory using PyTorch tensors
        self.graph_input = torch.empty(input_shape, dtype=input_dtype, device="cuda")
        self.graph_output = torch.empty(output_shape, dtype=output_dtype, device="cuda")
        
        # Register address bindings in Execution Context
        self.context.set_input_shape(input_name, input_shape)
        self.context.set_tensor_address(input_name, self.graph_input.data_ptr())
        self.context.set_tensor_address(output_name, self.graph_output.data_ptr())
        
        # Record execution path on a dedicated CUDA stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # First run for warmup and memory allocation inside TensorRT
            self.context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            
            # Record Graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            self.cuda_graph.capture_begin()
            self.context.execute_async_v3(stream.cuda_stream)
            self.cuda_graph.capture_end()
            
        self.graph_batch_size = batch_size
        print(f"[CUDA Graphs] Captured and instantiated Graph for batch_size={batch_size}")

    def infer(self, preprocessed_batch: torch.Tensor) -> np.ndarray:
        """Runs inference on a preprocessed GPU tensor.
        
        Args:
            preprocessed_batch: Input tensor already on GPU of shape [N, 3, 128, 128]
            
        Returns:
            Numpy array of shape [N, 2] containing model output logits.
        """
        batch_size = preprocessed_batch.shape[0]
        input_name = self.input_names[0]
        output_name = self.output_names[0]
        
        input_dtype = self.tensor_dtypes[input_name]
        output_dtype = self.tensor_dtypes[output_name]

        # Path 1: High-speed CUDA Graph Execution
        if self.enable_cuda_graph:
            # Check if batch size matches the recorded graph batch size
            if batch_size == self.graph_batch_size:
                # Copy input directly into the captured CUDA memory buffer
                self.graph_input.copy_(preprocessed_batch.to(input_dtype))
                # Replay the CUDA graph (zero CPU kernel launch overhead)
                self.cuda_graph.replay()
                # Copy output tensor to CPU
                return self.graph_output.cpu().numpy()
            else:
                # Batch size changed: Re-capture graph for the new batch size
                try:
                    self._setup_cuda_graph(batch_size=batch_size)
                    self.graph_input.copy_(preprocessed_batch.to(input_dtype))
                    self.cuda_graph.replay()
                    return self.graph_output.cpu().numpy()
                except Exception as e:
                    print(f"[Warning] Failed to re-capture CUDA Graph: {e}. Falling back to standard execution.")
                    self.enable_cuda_graph = False

        # Path 2: Standard Asynchronous Context Execution (Fallback / Dynamic Shape path)
        input_shape = (batch_size, *self.input_shapes[input_name][1:])
        output_shape = (batch_size, *self.output_shapes[output_name][1:])
        
        # Ensure input tensor matches the required dtype and is contiguous
        input_gpu = preprocessed_batch.to(input_dtype).contiguous()
        output_gpu = torch.empty(output_shape, dtype=output_dtype, device="cuda")
        
        # Update context bindings
        self.context.set_input_shape(input_name, input_shape)
        self.context.set_tensor_address(input_name, input_gpu.data_ptr())
        self.context.set_tensor_address(output_name, output_gpu.data_ptr())
        
        # Run on current CUDA stream
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        
        return output_gpu.cpu().numpy()
