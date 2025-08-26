# TrashNet YOLOv8 Object Detection

This repository contains code for training and running real-time object
detection on the **TrashNet dataset** using YOLOv8.

------------------------------------------------------------------------

## 🚀 Features

-   Prepare dataset from TrashNet into YOLO format\
-   Train YOLOv8 on the dataset\
-   Run real-time detection using webcam or video\
-   Save detection results as video

------------------------------------------------------------------------

## 📂 Project Structure

    📂 your-repo/
     ├── detect_realtime.py     # Run detection (webcam/video + save option)
     ├── dataset_prep.py        # Convert TrashNet to YOLO format
     ├── train_yolov8.py        # Train YOLOv8 model
     ├── requirements.txt       # Dependencies
     ├── README.md              # Documentation
     ├── LICENSE                # MIT License

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/sevugamoorthi/trashnet-yolov8.git
cd trashnet-yolov8
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📥 Download Dataset & Weights

To keep the repo lightweight, datasets and weights are hosted on Google
Drive.\
Download everything (dataset + trained weights + base YOLO models) here:

➡️ [Download yolov8n.zip](https://drive.google.com/file/d/1rMeE8buFFiBNSf9_fgFkMMIhPq6NhkyV/view?usp=sharing)

Unzip it and place the contents inside the project root folder.

It contains: - `datasets/trashnet_yolo/` -
`runs/detect/train/weights/` - `yolo11n.pt` - `yolov8n.pt`

------------------------------------------------------------------------

## 🏋️ Training (if you want to train the model, else use the pre-trained model in the runs folder for 'runs/detect/train/weights/best.pt' path to go directly in Detection) 

``` bash
python train_yolov8.py --data datasets/trashnet_yolo/data.yaml --weights yolov8n.pt --epochs 50
```

------------------------------------------------------------------------

## 🎥 Detection (Webcam or Video)

### Run with webcam:

``` bash
python detect_realtime.py --weights runs/detect/train/weights/best.pt --source 0 --save output.mp4
```

### Run with video:

``` bash
python detect_realtime.py --weights runs/detect/train/weights/best.pt --source input_video.mp4 --save output.mp4
```

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.\
See [LICENSE](LICENSE) for details.
