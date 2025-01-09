
from typing import List, Tuple
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import openvino as ov
from torchvision.transforms.functional import resize, to_pil_image


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


core = ov.Core()
ov_encoder = core.compile_model("encoder_ir.xml", device_name='CPU')
infer_request = ov_encoder.create_infer_request()

img = cv2.imread("../images/truck.jpg")
preprocess_img = pre_processing(img)
input_tensor = ov.Tensor(array=preprocess_img, shared_memory=True)


import time
start = time.time()
infer_request.set_input_tensor(input_tensor)
infer_request.start_async()
infer_request.wait()

output = infer_request.get_output_tensor()
output_buffer = output.data
print(f"use time: {time.time() - start}")

# import time
# start = time.time()
# encode_result = ov_encoder(preprocess_img)
# print(f"use time: {time.time() - start}")
