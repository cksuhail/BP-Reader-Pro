from ultralytics import YOLO

# Load YOLOv8 SMALL model (best for accuracy)
model = YOLO("yolov8s.pt")   

# Train
model.train(
    data="/home/ubuntu/BP_project/My First Project.v2i.yolov8/data.yaml",        # Path to your dataset YAML
    epochs=50,               # 50 recommended for accuracy
    imgsz=640,               # Image size
    batch=8,                 # Change to 16 if GPU strong
    patience=20,             # Stop early if no improvement
    workers=4,
    device=0                 # GPU (switch to "cpu" if needed)
)
