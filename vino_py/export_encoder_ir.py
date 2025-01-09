

import openvino as ov


core = ov.Core()
onnx_encoder_path = "../model/sam_vit_h_encoder_static.onnx"
ov_encoder_path = "./encoder_ir.xml"

ov_encoder_model = ov.convert_model(onnx_encoder_path)
ov.save_model(ov_encoder_model, str(ov_encoder_path), compress_to_fp16=True)

