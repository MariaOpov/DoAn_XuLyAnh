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


