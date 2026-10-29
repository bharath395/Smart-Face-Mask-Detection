from ultralytics import YOLO

# Path to dataset YAML file in YOLO format
# The dataset YAML should define train, val, and nc (number of classes)
dataset_yaml = 'mask_dataset.yaml'  # Update with your dataset path

# Initialize YOLOv8 model (YOLOv8n for nano model, can use YOLOv8s or YOLOv8m for better accuracy)
model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 nano model

# Train the model on mask dataset
model.train(data=dataset_yaml,
            epochs=10,
            imgsz=640,
            batch=16,
            name='yolov8_mask_detector',
            save=True,
            device='cpu')  # Use CPU explicitly

# After training, the best model weights will be saved in 'runs/train/yolov8_mask_detector/weights/best.pt'
