from ultralytics import YOLO
model = YOLO(".\supermercado.pt")
model.export(format="onnx", opset=12, simplify=True, imgsz=640)
