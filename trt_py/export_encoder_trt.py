
import tensorrt as trt


onnx_path = "../onnx_py/model/sam_vit_h_encoder_static.onnx"
trt_path = "./sam_vit_h_encoder.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

config = builder.create_builder_config()

workspace = 6
print(f"workspace : {workspace}")
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1 << 30)

flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)

parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(onnx_path):
    raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

for inp in inputs:
    print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
for out in outputs:
    print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

half = True
print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {onnx_path}')
if builder.platform_has_fast_fp16 and half:
    config.set_flag(trt.BuilderFlag.FP16)

serialized_engine = builder.build_serialized_network(network, config)

with open(trt_path, "wb") as f:
    f.write(serialized_engine)

