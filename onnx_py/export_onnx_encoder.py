
from typing import List
import os
import pathlib
import shutil
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx

from tempfile import mkdtemp
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from onnx.external_data_helper import convert_model_to_external_data

parser = argparse.ArgumentParser(
    description="Export the SAM image encoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="The filename to save the ONNX model to.",
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b', 'mobile']. "
    "Which type of SAM model to export.",
)

parser.add_argument(
    "--use-preprocess",
    action="store_true",
    help=("Embed pre-processing into the model",),
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from "
        "onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)


class ImageEncoderOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the image encoder of Sam, with some functions modified to enable
    model tracing. Also supports extra options controlling what information. See
    the ONNX export script for details.
    """

    DEFAULT_PIXEL_MEAN = [123.675, 116.28, 103.53]
    DEFAULT_PIXEL_STD = [58.395, 57.12, 57.375]

    def __init__(
        self,
        model: Sam,
        use_preprocess: bool,
        pixel_mean: List[float] = None,
        pixel_std: List[float] = None,
    ):
        if pixel_mean is None:
            pixel_mean = self.DEFAULT_PIXEL_MEAN
        if pixel_std is None:
            pixel_std = self.DEFAULT_PIXEL_STD

        super().__init__()
        self.use_preprocess = use_preprocess
        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)
        self.image_encoder = model.image_encoder

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # permute channels
        x = torch.permute(x, (2, 0, 1))

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        # expand channels
        x = torch.unsqueeze(x, 0)
        return x


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
    gelu_approximate: bool = False,
):
    print("Loading model...")
    if model_type == "mobile":
        print("do not support mobile sam")
    else:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = ImageEncoderOnnxModel(
        model=sam,
        use_preprocess=use_preprocess,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    onnx_model.eval()

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = sam.image_encoder.img_size
    if use_preprocess:
        dummy_input = {
            "input_image": torch.randn(
                (image_size, image_size, 3), dtype=torch.float
            )
        }
        dynamic_axes = {
            "input_image": {0: "image_height", 1: "image_width"},
        }
    else:
        dummy_input = {
            "input_image": torch.randn(
                (1, 3, image_size, image_size), dtype=torch.float
            )
        }
        dynamic_axes = None

    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    onnx_base = os.path.splitext(os.path.basename(output))[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        if model_type == "vit_h":
            tmp_dir = mkdtemp()
            tmp_model_path = os.path.join(tmp_dir, f"{onnx_base}.onnx")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                tmp_model_path,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            # Combine the weights into a single file
            pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
            onnx_model = onnx.load(tmp_model_path)
            convert_model_to_external_data(
                onnx_model,
                all_tensors_to_one_file=True,
                location=f"{onnx_base}_data.bin",
                size_threshold=1024,
                convert_attribute=False,
            )
            # Save the model
            onnx.save(onnx_model, output)
            # Cleanup the temporary directory
            shutil.rmtree(tmp_dir)
        else:
            with open(output, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_input.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=list(dummy_input.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    if args.quantize_out is not None:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
