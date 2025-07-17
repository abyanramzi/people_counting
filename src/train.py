from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("yolo11n.pt")
model.train(data="dataset.yaml", epochs=100, workers=0, batch=16, device=0)
