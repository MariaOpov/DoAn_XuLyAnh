# X-ray Object Detection System

A computer vision project for detecting and locating objects in baggage X-ray images using YOLOv8.

The project includes image preprocessing with OpenCV, annotation conversion to YOLO format, model training, and prediction visualization with confidence scores.

## Features

- Preprocesses X-ray images to improve image quality.
- Converts bounding-box annotations to YOLO format.
- Trains a YOLOv8 object detection model.
- Detects and locates objects in X-ray images.
- Displays bounding boxes, class labels, and confidence scores.

## Technologies

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- Google Colab

## Installation

Clone the repository:

```bash
git clone https://github.com/MariaOpov/DoAn_XuLyAnh.git
cd DoAn_XuLyAnh
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Prepare the Dataset

Place the baggage X-ray images and the `BoundingBox.mat` annotation file in the dataset directory.

Then run:

```bash
python convert_to_yolo.py
```

This script will:

- Read the original X-ray images and annotations.
- Apply image preprocessing.
- Convert bounding-box annotations to YOLO format.
- Create the required dataset folders.
- Generate the `xray_config.yaml` configuration file.

### 2. Train the Model

Run the training script:

```bash
python train_yolo.py
```

The script trains a YOLOv8 model using the prepared dataset.

The trained model will be saved in a directory similar to:

```text
runs/train/weights/best.pt
```

The exact directory name may vary depending on the training configuration.

### 3. Run Object Detection

Make sure the trained model path and test image path in `predict.py` are correct.

Then run:

```bash
python predict.py
```

The program will:

- Load the trained YOLOv8 model.
- Detect objects in the selected X-ray image.
- Draw bounding boxes around detected objects.
- Display class labels and confidence scores.

## Project Workflow

```text
Raw X-ray Images
        ↓
Image Preprocessing
        ↓
Annotation Conversion
        ↓
YOLO Dataset Preparation
        ↓
Model Training
        ↓
Object Detection
        ↓
Result Visualization
```
