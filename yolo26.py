from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(data="/datasets/global_daily_object_yolo/Dollar street trial/meta_yolo.yaml", epochs=100, imgsz=640)