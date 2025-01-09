
from typing import List, Tuple
from collections import OrderedDict
import torch
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image

logger = trt.Logger(trt.Logger.INFO)

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self) -> None:
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name: str, memory: int, size: int, alignment: int) -> int:
        if tensor_name in self.buffers:
            del self.buffers[tensor_name]

        address = cuda.mem_alloc(size)
        self.buffers[tensor_name] = address

        return int(address)

    def notify_shape(self, tensor_name: str, shape: trt.Dims):
        self.shapes[tensor_name] = tuple(shape)


def get_input_tensor_names(engine: trt.ICudaEngine) -> List[str]:
    input_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(binding)

    return input_tensor_names


def get_output_tensor_names(engine: trt.ICudaEngine) -> List[str]:
    output_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            output_tensor_names.append(binding)

    return output_tensor_names


class InferV3(object):
    def __init__(self, engine: trt.ICudaEngine, gpu_id=0) -> None:
        # device = cuda.Device(0)
        # self.cuda_context = device.make_context()
        # self.cuda_context.push()

        self.engine = engine
        self.output_allocator = OutputAllocator()

        # create execution context
        self.context = engine.create_execution_context()

        # get input and output tensor name
        self.input_tensor_names = get_input_tensor_names(self.engine)
        self.output_tensor_names = get_output_tensor_names(self.engine)

        # create stream
        self.stream = cuda.Stream()

        # create cuda events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    # def __del__(self):
    #     self.cuda_context.pop()

    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def infer(self, inputs: List[np.ndarray]):
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """
        for name, arr in zip(self.input_tensor_names, inputs):
            self.context.set_input_shape(name, arr.shape)

        buffers_host = []
        buffers_device = []

        # host -> cuda
        for name, arr in zip(self.input_tensor_names, inputs):
            host = cuda.pagelocked_empty(arr.shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = cuda.mem_alloc(arr.nbytes)

            host[:] = arr
            cuda.memcpy_htod_async(device, host, self.stream)
            buffers_host.append(host)
            buffers_device.append(device)

        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))

        for name in self.output_tensor_names:
            self.context.set_tensor_address(name, 0)
            self.context.set_output_allocator(name, self.output_allocator)

        self.start_event.record(self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.end_event.record(self.stream)

        output_buffers = OrderedDict()
        for name in self.output_tensor_names:
            arr = cuda.pagelocked_empty(self.output_allocator.shapes[name], dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh_async(arr, self.output_allocator.buffers[name], stream=self.stream)
            output_buffers[name] = arr

        self.stream.synchronize()

        return output_buffers


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def pre_processing(image: np.ndarray,
                   img_size: int = 1024,
                   target_length: int = 1024,
                   pixel_mean: List[float] = [123.675, 116.28, 103.53],
                   pixel_std: List[float] = [58.395, 57.12, 57.375]):
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch.numpy()


def load_engine(path):
    runtime = trt.Runtime(logger)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

image_path = "../images/truck.jpg"
img_inputs = pre_processing(cv2.imread(image_path))

engine = load_engine("./sam_vit_h_encoder.engine")
processer = InferV3(engine)
outputs = processer.infer([img_inputs])
use_time = processer.get_last_inference_time()
print(f"use time: {use_time / 1000}s")

