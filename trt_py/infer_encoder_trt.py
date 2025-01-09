
from typing import List, Tuple
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from torchvision.transforms.functional import resize, to_pil_image


def load_engine(trt_runtime, engine_path):
    trt.init_libnvinfer_plugins(None, "")
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel(object):
    def __init__(self, engine_path, gpu_id=0, max_batch_size=1) -> None:
        print(f"TensorRT Version: {trt.__version__}")
        cuda.init()
        self.cfx = cuda.Device(gpu_id).make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = load_engine(self.runtime, engine_path)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            if binding in ["point_coords", "point_labels"]:
                size = abs(trt.volume(self.engine.get_tensor_shape(binding))) * self.max_batch_size
            else:
                size = abs(trt.volume(self.engine.get_tensor_shape(binding)))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def run(self, inf_in_list, binding_shape_map=None):
        self.cfx.push()

        if binding_shape_map:
            self.context.set_optimization_profile_async
            for binding_name, shape in binding_shape_map.items():
                binding_idx = self.engine[binding_name]
                self.context.set_binding_shape(binding_idx, shape)

        for i in range(len(self.inputs)):
            self.inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(self.inputs[i].device, self.inputs[i].host, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i].host, self.outputs[i].device, self.stream)

        self.stream.synchronize()
        self.cfx.pop()
        return [out.host.copy() for out in self.outputs]

    def __del__(self):
        self.cfx.pop()
        del self.cfx


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

image_path = "../images/truck.jpg"
embedding_inference = TrtModel("./sam_vit_h_encoder.engine")
img_inputs = pre_processing(cv2.imread(image_path))

import time
start = time.time()
image_embedding = embedding_inference.run([img_inputs])
print(f"use time: {time.time() - start}")
