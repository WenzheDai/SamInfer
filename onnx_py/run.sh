# export vit h encoder
python export_onnx_encoder.py \
  --checkpoint ../pth_model/sam_vit_h_4b8939.pth \
  --output ./model/tmp/sam_vit_h_encoder_static_eval.onnx \
  --model-type vit_h
#
# export vit h encoder
# python export_onnx_encoder.py \
#   --checkpoint ../pth_model/sam_vit_l_0b3195.pth \
#   --output ./model/sam_vit_l_encoder_static.onnx \
#   --model-type vit_l

# # export vit decoder
# python export_onnx_model.py \
#   --checkpoint ../pth_model/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --output ./model/sam_vit_h_decoder.onnx \
#   --return-single-mask

