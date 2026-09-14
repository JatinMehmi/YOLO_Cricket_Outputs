from ultralytics import YOLO

# Load model
model = YOLO("best.pt")
# second comment added
# Export to ONNX instead of LiteRT/TFLite
model.export(format="onnx", opset=12, dynamic=False)