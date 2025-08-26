import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_yolov8.py datasets/trashnet_yolo")
        sys.exit(1)

    data_root = sys.argv[1]
    data_yaml = os.path.join(data_root, "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found!")
        sys.exit(1)

    print(f"Training YOLOv8 model with dataset: {data_root}")

    cmd = [
        "yolo", "detect", "train",
        f"data={data_yaml}",
        "model=yolov8n.pt",
        "epochs=50",
        "imgsz=640"
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
