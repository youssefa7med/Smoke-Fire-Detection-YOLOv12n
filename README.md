# 🔥 Smoke & Fire Detection using YOLOv12n

![Smoke Fire Detection Demo](https://europe1.discourse-cdn.com/flex018/uploads/forotuenti/original/1X/f57e560becb08fa4b640f0f85ebd6f23941669ec.gif)

A real-time smoke and fire detection system powered by YOLOv12n (YOLO v12 nano), featuring instant alerts, audio alarms, and Telegram notifications for enhanced safety monitoring.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an advanced computer vision system for detecting smoke and fire in real-time video streams. Built on the lightweight YOLOv12n architecture, it provides fast and accurate detection capabilities suitable for surveillance systems, smart home automation, and industrial safety applications.

### Key Highlights

- ⚡ **Fast Inference**: ~28ms per frame at 384x640 resolution
- 🎯 **High Accuracy**: Trained on custom datasets for optimal performance
- 🔔 **Multi-Channel Alerts**: Audio alarms + Telegram notifications
- 📹 **Real-Time Processing**: Live video stream analysis
- 🎨 **Visual Annotations**: Bounding boxes and confidence scores

---

## ✨ Features

- **Real-Time Detection**: Process video streams frame-by-frame with minimal latency
- **Dual Detection**: Simultaneously detects both smoke and fire
- **Audio Alarms**: Automatic alarm sound playback when threats are detected
- **Telegram Integration**: Instant photo alerts sent to Telegram with timestamps
- **Visual Feedback**: Annotated video output with bounding boxes and labels
- **Cooldown System**: Prevents notification spam with configurable delays
- **Model Training**: Complete training pipeline with validation metrics

---

## 🛠️ Technologies Used

- **YOLOv12n**: Ultralytics YOLO v12 nano model
- **Python 3.x**: Core programming language
- **OpenCV (cv2)**: Video processing and image handling
- **PyGame**: Audio alarm playback
- **Requests**: HTTP requests for Telegram API
- **Ultralytics**: YOLO model inference and training
- **NumPy**: Numerical computations
- **Pillow**: Image processing utilities

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- Webcam or video file for testing

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Smoke-Fire-Detection-YOLOv12n.git
cd Smoke-Fire-Detection-YOLOv12n
```

### Step 2: Install Dependencies

```bash
pip install ultralytics opencv-python pygame requests numpy pillow
```

Or install from requirements.txt (if available):

```bash
pip install -r requirements.txt
```

### Step 3: Download Model Weights

Ensure you have the `best.pt` model file in the project root directory. This is the trained YOLOv12n model for smoke and fire detection.

---

## 🚀 Usage

### Basic Detection (Video File)

```python
import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Process video file
video_path = "Test Video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Display results
    cv2.imshow("Fire Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Full-Featured Detection (with Alarms & Telegram)

See `notebook.ipynb` for the complete implementation with:
- Audio alarm system
- Telegram notifications
- Cooldown management
- Timestamped alerts

### Telegram Setup

1. Create a Telegram bot using [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Get your chat ID (use [@userinfobot](https://t.me/userinfobot))
4. Update the credentials in your script:

```python
token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
```

---

## 📁 Project Structure

```
Smoke-Fire-Detection-YOLOv12n/
│
├── README.md                 # Project documentation
├── notebook.ipynb            # Main implementation notebook
├── smoke-fire-detection.ipynb # Alternative notebook
├── best.pt                   # Trained model weights
├── alarm.mp3                 # Alarm sound file
├── minion_fire_alarm.mp3     # Alternative alarm sound
├── Test Video.mp4            # Test video file
│
├── train 01/                 # Training run 1 results
│   ├── args.yaml            # Training configuration
│   ├── results.png          # Training metrics
│   ├── confusion_matrix.png # Confusion matrix
│   └── weights/             # Model checkpoints
│       ├── best.pt
│       └── last.pt
│
└── train 02/                 # Training run 2 results
    ├── args.yaml
    ├── results.png
    └── weights/
        ├── best.pt
        └── last.pt
```

---

## 🎓 Model Training

### Training Configuration

The model was trained with the following parameters:

- **Model**: YOLOv12n (nano variant)
- **Input Size**: 640x640 pixels
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: AdamW (auto)
- **Learning Rate**: 0.001 (initial)
- **Device**: GPU (CUDA)

### Training Results

Training metrics and visualizations are available in the `train 01/` and `train 02/` directories, including:
- Precision-Recall curves
- F1-score curves
- Confusion matrices
- Training/validation loss plots

### Custom Training

To train on your own dataset:

1. Prepare your dataset in YOLO format
2. Create a `data.yaml` configuration file
3. Run training:

```python
from ultralytics import YOLO

model = YOLO("yolo12n.pt")  # Start from pre-trained weights
results = model.train(
    data="path/to/data.yaml",
    epochs=40,
    imgsz=640,
    batch=8,
    device=0
)
```

---

## ⚙️ Configuration

### Detection Thresholds

Adjust confidence and IoU thresholds in your inference code:

```python
results = model(frame, conf=0.25, iou=0.45)
```

### Alarm Settings

- **Alarm Sound**: Change `alarm_sound` path to use different audio files
- **Cooldown**: Adjust `cooldown_seconds` to control notification frequency

### Video Source

Switch between video file and webcam:

```python
# Video file
cap = cv2.VideoCapture("path/to/video.mp4")

# Webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
```

---

## 📊 Performance Metrics

- **Inference Speed**: ~28ms per frame (384x640 resolution)
- **Preprocessing**: ~2-3ms
- **Postprocessing**: ~4-8ms
- **FPS**: ~35-40 frames per second

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv12 implementation
- The open-source computer vision community
- Contributors and testers of this project

---

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for Safety and Security**

⭐ Star this repo if you find it helpful!

</div>

