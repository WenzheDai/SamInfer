
from onnxruntime.quantization import QuantType  # type: ignore
from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore


model_input="./model/tmp/sam_vit_h_encoder_static_eval.onnx"
model_output="./model/tmp/sam_vit_h_encoder_static_eval_quantize.onnx"

print(f"Quantizing model and writing to {model_output}...")

quantize_dynamic(
    model_input=model_input,
    model_output=model_output,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
print("Done!")

