import copy
from typing import Dict, Tuple, Union

import time
import cv2
import numpy as np
import onnxruntime as ort
import tqdm


class SamOnnxModel(object):
    target_length = 1024
    def __init__(
        self,
        encoder_model_path: str,
        decoder_model_path: str,
        origin_image_size: Tuple[int, int],
        mask_threshold=0.0,
        cpu_num_thread=None,
        epochs=0,
    ) -> None:
        self.origin_image_size = origin_image_size
        self.mask_threshold = mask_threshold

        self.opt = ort.SessionOptions()
        if cpu_num_thread is not None:
            self.opt.intra_op_num_threads = cpu_num_thread
            # self.opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # self.opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        self.providers = ["CPUExecutionProvider"]

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

        self.encoder_session = ort.InferenceSession(
            encoder_model_path, self.opt, self.providers
        )

        self.decoder_session = ort.InferenceSession(
            decoder_model_path, self.opt, self.providers
        )

        self.input_name = self.encoder_session.get_inputs()[0].name
        self.input_shape = self.encoder_session.get_inputs()[0].shape
        self.output_name = self.encoder_session.get_outputs()[0].name
        self.output_shape = self.encoder_session.get_outputs()[0].shape

        self.warmup(epochs)

    def warmup(self, epochs):
        x = np.random.random(self.input_shape).astype(np.float32)
        print('start warmup')
        for _ in tqdm.tqdm(range(epochs)):
            self.encoder_session.run(None, {self.input_name: x})
        print('warmup finish')

    def transform_image(self, image: cv2.Mat) -> np.ndarray:
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - self.mean) / self.std
        size = max(h, w)
        image = cv2.copyMakeBorder(
            image, 0, (size - h), 0, (size - w), cv2.BORDER_CONSTANT
        ).astype(np.float32)
        image = cv2.resize(image, self.input_shape[2:], interpolation=cv2.INTER_LINEAR)

        batch_image = np.expand_dims(image, axis=0)
        batch_image = np.transpose(batch_image, axes=[0, 3, 1, 2]).astype(np.float32)

        return batch_image


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

            point_coords = apply_coords(
                point_coords, self.origin_image_size, self.target_length
            )
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = (
                apply_boxes(boxes, self.origin_image_size, self.target_length)
                .reshape((1, -1, 2))
                .astype(np.float32)
            )
            box_label = np.array(
                [[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32
            ).reshape((1, -1))

            print(f"boxes: {boxes}")

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

            print(f"point coords: {point_coords}")

        return point_coords, point_labels

    def run_encoder(self, image: cv2.Mat):
        batch_image = self.transform_image(image)
        assert list(batch_image.shape) == self.input_shape
        image_embedding = self.encoder_session.run(
            None, {self.input_name: batch_image}
        )[0]
        assert list(image_embedding.shape) == self.output_shape

        return image_embedding

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
    coords: np.ndarray, original_size: Tuple[int, int], target_length: int
) -> np.ndarray:
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)
    coords = copy.deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(
    boxes: np.ndarray, original_size: Tuple[int, int], target_length: int
) -> np.ndarray:
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, target_length)
    return boxes.reshape(-1, 4)


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def infer(
    encoder_model_path: str,
    decoder_model_path: str,
    image_path: str,
    point_coords: Union[list, np.ndarray] = None,
    point_labels: Union[list, np.ndarray] = None,
    boxes: Union[list, np.ndarray] = None,
):
    image = cv2.imread(image_path)
    origin_image_size = image.shape[:2]

    sam_onnx_model = SamOnnxModel(
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
        origin_image_size=origin_image_size,
        mask_threshold=0.0,
        cpu_num_thread=None,
    )

    start = time.time()
    image_embedding = sam_onnx_model.run_encoder(image)
    encode_time = time.time()
    res = sam_onnx_model.run_decoder(
        img_embeddings=image_embedding,
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
    )
    print(f"encode time: {encode_time - start}")
    print(f"decode time: {time.time() - encode_time}")
    print(f"all time :{time.time() - start}")

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
    # encoder_model_path = "./model/sam_vit_h_encoder_static.onnx"
    encoder_model_path = "./model/sam_vit_h_encoder_static_quntized.onnx"
    decoder_model_path = "./model/sam_vit_h_decoder.onnx"
    image_path = "../images/truck.jpg"

    points = [[575, 750]]
    point_labels = [0]
    boxes = [[425, 600, 700, 875]]

    res = infer(
        encoder_model_path,
        decoder_model_path,
        image_path,
        point_coords=points,
        point_labels=point_labels,
        boxes=boxes,
    )

    save_result(res, image_path)
