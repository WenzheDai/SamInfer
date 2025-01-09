
from typing import List, Tuple, Union, Dict
from collections import OrderedDict
import copy
import torch
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch.nn.functional as F
import onnxruntime as ort
import time
from torchvision.transforms.functional import resize, to_pil_image


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


class SamEncoderTRT(object):
    def __init__(self, engine_path: str, gpu_id=1) -> None:
        print(f"TensorRT version: {trt.__version__}")
        self.cfx = cuda.Device(gpu_id).make_context()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self.engine = self.load_engine(engine_path, self.runtime)
        self.output_allocator = OutputAllocator()

        # create execution context
        self.context = self.engine.create_execution_context()

        # get input and output tensor name
        self.input_tensor_names = self.get_input_tensor_names(self.engine)
        self.output_tensor_names = self.get_output_tensor_names(self.engine)

        # create stream
        self.stream = cuda.Stream()

        # create cuda events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def run(self, inputs: List[np.ndarray]):
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """
        self.cfx.push()
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
        self.cfx.pop()

        return output_buffers

    def __del__(self):
        self.cfx.pop()
        del self.cfx

    def load_engine(self, path, runtime):
        with open(path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    @staticmethod
    def get_input_tensor_names(engine: trt.ICudaEngine) -> List[str]:
        input_tensor_names = []
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                input_tensor_names.append(binding)

        return input_tensor_names

    @staticmethod
    def get_output_tensor_names(engine: trt.ICudaEngine) -> List[str]:
        output_tensor_names = []
        for binding in engine:
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                output_tensor_names.append(binding)

        return output_tensor_names


class SamDecoderONNX(object):
    target_length = 1024
    def __init__(
        self,
        decoder_model_path: str,
        origin_image_size: Tuple[int, int],
        mask_threshold=0.0,
        cpu_num_thread=None,
    ) -> None:
        self.origin_image_size = origin_image_size
        self.mask_threshold = mask_threshold

        self.opt = ort.SessionOptions()
        if cpu_num_thread is not None:
            self.opt.intra_op_num_threads = cpu_num_thread

        self.providers = ["CPUExecutionProvider"]

        self.decoder_session = ort.InferenceSession(
            decoder_model_path, self.opt, self.providers
        )

    def get_points_coords(
        self,
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

            point_coords = self.apply_coords(
                point_coords, self.origin_image_size, self.target_length
            )
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = (
                self.apply_boxes(boxes, self.origin_image_size, self.target_length)
                .reshape((1, -1, 2))
                .astype(np.float32)
            )
            box_label = np.array(
                [[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32
            ).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        return point_coords, point_labels

    def run_decoder(
        self,
        img_embeddings: np.ndarray,
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        mask_input: np.ndarray = None,
    ) -> None:
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError(
                "Unable to segment, please input at least one box and point"
            )

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")

        point_coords, point_labels = self.get_points_coords(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
        )

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": point_coords.astype(np.float32),
            "point_labels": point_labels.astype(np.float32),
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(self.origin_image_size, dtype=np.float32),
        }
        res = self.decoder_session.run(None, input_dict)

        result_dict = {}
        for i in range(len(res)):
            out_name = self.decoder_session.get_outputs()[i].name
            if out_name == "masks":
                mask = (res[i] > self.mask_threshold).astype(np.int32)
                result_dict[out_name] = mask
            else:
                result_dict[out_name] = res[i]

        return result_dict

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, int], target_length: int
    ) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, target_length)
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, int], target_length: int
    ) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, target_length)
        return boxes.reshape(-1, 4)

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
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


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def infer(
    encoder_engine_path: str,
    decoder_onnx_path: str,
    image_path: str,
    point_coords: Union[list, np.ndarray] = None,
    point_labels: Union[list, np.ndarray] = None,
    boxes: Union[list, np.ndarray] = None,
):
    image = cv2.imread(image_path)
    img_inputs = pre_processing(image)

    t1 = time.time()
    encoder = SamEncoderTRT(encoder_engine_path, 1)
    print(f"load engine time: {time.time() - t1}s\n")

    img_embeddings = encoder.run([img_inputs])
    print(f"encoder time: {encoder.get_last_inference_time() / 1000}s\n")

    origin_image_size = image.shape[:2]
    decoder = SamDecoderONNX(decoder_onnx_path, origin_image_size)

    t3 = time.time()
    res = decoder.run_decoder(
        img_embeddings=img_embeddings["image_embeddings"],
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
    )
    print(f"decoder time: {time.time() - t3}s\n")

    return res


def save_result(result: Dict[str, np.ndarray], image_path: str):
    masks = result["masks"]
    image = cv2.imread(image_path)

    mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
    for m in masks[0, :, :, :]:
        mask[m > 0.0] = [255, 0, 0]
    res_image = cv2.addWeighted(image, 0.8, mask, 0.2, 0)

    cv2.imwrite("./res.jpg", res_image)


if __name__ == "__main__":
    image_path = "../images/truck.jpg"
    encoder_engine_path = "./sam_vit_h_encoder.engine"
    decoder_onnx_path = "../model/sam_vit_h_decoder.onnx"

    points = None
    point_labels = [0]
    boxes = [[421, 613, 706, 861]]

    res = infer(
        encoder_engine_path,
        decoder_onnx_path,
        image_path,
        point_coords=points,
        point_labels=point_labels,
        boxes=boxes,
    )

    save_result(res, image_path)
