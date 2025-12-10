# 🔥 Smoke & Fire Detection using YOLOv12n

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v12n-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)

![Smoke Fire Detection Demo](https://europe1.discourse-cdn.com/flex018/uploads/forotuenti/original/1X/f57e560becb08fa4b640f0f85ebd6f23941669ec.gif)

A state-of-the-art real-time smoke and fire detection system powered by YOLOv12n (YOLO v12 nano), featuring instant alerts, audio alarms, and Telegram notifications for enhanced safety monitoring. This project demonstrates advanced computer vision capabilities for critical safety applications.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Model Training](#-model-training)
- [Configuration](#-configuration)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a production-ready computer vision system for detecting smoke and fire in real-time video streams. Built on the lightweight yet powerful YOLOv12n architecture, it provides exceptional speed and accuracy suitable for:

- 🏢 **Surveillance Systems**: Continuous monitoring of indoor and outdoor environments
- 🏠 **Smart Home Automation**: Integration with home security systems
- 🏭 **Industrial Safety**: Early warning systems for manufacturing facilities
- 🚨 **Emergency Response**: Automated alert systems for fire departments

### Key Highlights

- ⚡ **Ultra-Fast Inference**: ~28ms per frame at 384x640 resolution
- 🎯 **High Accuracy**: mAP50 of **76.97%** and mAP50-95 of **45.93%**
- 🔔 **Multi-Channel Alerts**: Audio alarms + Telegram notifications
- 📹 **Real-Time Processing**: Live video stream analysis with minimal latency
- 🎨 **Visual Annotations**: Bounding boxes with confidence scores
- 🔄 **Production Ready**: Robust error handling and cooldown mechanisms

---

## ✨ Key Features

### Detection Capabilities
- **Dual-Class Detection**: Simultaneously detects both smoke and fire
- **Real-Time Processing**: Frame-by-frame analysis with minimal latency
- **High Precision**: Optimized thresholds for reduced false positives

### Alert System
- **Audio Alarms**: Automatic alarm sound playback when threats are detected
- **Telegram Integration**: Instant photo alerts with timestamps and location data
- **Cooldown Management**: Configurable delays to prevent notification spam
- **Visual Feedback**: Real-time annotated video output

### Model Training
- **Comprehensive Training Pipeline**: Complete training and validation workflow
- **Multiple Training Runs**: Two separate training experiments for optimal results
- **Detailed Metrics**: Precision-Recall curves, confusion matrices, and loss plots

---

## 📊 Model Performance

### Training Experiments

This project involved **two separate training runs** (not sequential) to optimize model performance:

#### Training Run 1: Initial Experiment
- **Epochs**: 40
- **Purpose**: Baseline model training
- **Final Metrics**:
  - mAP50: 73.24%
  - mAP50-95: 42.37%
  - Precision: 73.70%
  - Recall: 65.79%

#### Training Run 2: Optimized Model ⭐ **BEST RESULTS**
- **Epochs**: 100
- **Purpose**: Extended training for improved accuracy
- **Final Metrics**:
  - **mAP50: 76.97%** ⬆️ (+3.73%)
  - **mAP50-95: 45.93%** ⬆️ (+3.56%)
  - **Precision: 74.51%** ⬆️ (+0.81%)
  - **Recall: 70.56%** ⬆️ (+4.77%)

**Note**: The model weights from Training Run 2 (`train 02/weights/best.pt`) are used in production as they achieved superior performance across all metrics.

### Performance Comparison

| Metric | Training Run 1 (40 epochs) | Training Run 2 (100 epochs) | Improvement |
|--------|---------------------------|----------------------------|-------------|
| **mAP50** | 73.24% | **76.97%** | +3.73% |
| **mAP50-95** | 42.37% | **45.93%** | +3.56% |
| **Precision** | 73.70% | **74.51%** | +0.81% |
| **Recall** | 65.79% | **70.56%** | +4.77% |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **YOLOv12n** | Latest | Object detection model |
| **Python** | 3.8+ | Core programming language |
| **OpenCV** | 4.x | Video processing and image handling |
| **PyGame** | Latest | Audio alarm playback |
| **Requests** | Latest | HTTP requests for Telegram API |
| **Ultralytics** | Latest | YOLO model inference and training |
| **NumPy** | Latest | Numerical computations |
| **Pillow** | Latest | Image processing utilities |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- Webcam or video file for testing
- 4GB+ RAM
- 2GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Smoke-Fire-Detection-YOLOv12n.git
cd Smoke-Fire-Detection-YOLOv12n
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install ultralytics opencv-python pygame requests numpy pillow
```

Or install from requirements.txt (if available):

```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Weights

Ensure you have the `best.pt` model file in the project root directory. This is the trained YOLOv12n model from Training Run 2 (100 epochs) with the best performance metrics.

---

## 🚀 Quick Start

### Basic Detection (Video File)

```python
import cv2
from ultralytics import YOLO

# Load the trained model
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

---

## 💻 Usage Examples

### Full-Featured Detection (with Alarms & Telegram)

See `notebook.ipynb` for the complete implementation with:
- Audio alarm system
- Telegram notifications
- Cooldown management
- Timestamped alerts
- Error handling

### Telegram Bot Setup

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Use `/newbot` command and follow instructions
   - Save your bot token

2. **Get Your Chat ID**:
   - Message [@userinfobot](https://t.me/userinfobot) on Telegram
   - Copy your chat ID

3. **Configure in Script**:
   ```python
   token = "YOUR_BOT_TOKEN"
   chat_id = "YOUR_CHAT_ID"
   ```

### Webcam Detection

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📁 Project Structure

```
Smoke-Fire-Detection-YOLOv12n/
│
├── README.md                    # Project documentation
├── notebook.ipynb               # Main implementation notebook
├── smoke-fire-detection.ipynb   # Alternative notebook
├── best.pt                      # Best model weights (from Training Run 2)
├── alarm.mp3                    # Alarm sound file
├── minion_fire_alarm.mp3        # Alternative alarm sound
├── Test Video.mp4               # Test video file
│
├── train 01/                    # Training Run 1 (40 epochs)
│   ├── args.yaml               # Training configuration
│   ├── results.csv              # Detailed training metrics
│   ├── results.png              # Training metrics visualization
│   ├── confusion_matrix.png     # Confusion matrix
│   ├── confusion_matrix_normalized.png
│   ├── BoxP_curve.png          # Precision curve
│   ├── BoxR_curve.png          # Recall curve
│   ├── BoxF1_curve.png         # F1-score curve
│   ├── BoxPR_curve.png         # Precision-Recall curve
│   └── weights/                # Model checkpoints
│       ├── best.pt             # Best weights from Run 1
│       └── last.pt             # Last epoch weights
│
└── train 02/                    # Training Run 2 (100 epochs) ⭐ BEST
    ├── args.yaml               # Training configuration
    ├── results.csv             # Detailed training metrics
    ├── results.png             # Training metrics visualization
    ├── confusion_matrix.png    # Confusion matrix
    ├── confusion_matrix_normalized.png
    ├── BoxP_curve.png         # Precision curve
    ├── BoxR_curve.png         # Recall curve
    ├── BoxF1_curve.png        # F1-score curve
    ├── BoxPR_curve.png        # Precision-Recall curve
    └── weights/               # Model checkpoints
        ├── best.pt            # Best weights from Run 2 (Production Model)
        └── last.pt            # Last epoch weights
```

---

## 🎓 Model Training

### Training Configuration

The model underwent two separate training experiments to achieve optimal performance:

#### Training Run 1 Configuration
- **Model**: YOLOv12n (nano variant)
- **Input Size**: 640×640 pixels
- **Batch Size**: 8
- **Epochs**: 40
- **Optimizer**: AdamW (auto)
- **Initial Learning Rate**: 0.001
- **Device**: GPU (CUDA)

#### Training Run 2 Configuration ⭐
- **Model**: YOLOv12n (nano variant)
- **Input Size**: 640×640 pixels
- **Batch Size**: 8
- **Epochs**: 100
- **Optimizer**: AdamW (auto)
- **Initial Learning Rate**: 0.001
- **Device**: GPU (CUDA)

### Training Results

Comprehensive training metrics and visualizations are available in both training directories:

- **Precision-Recall Curves**: `BoxPR_curve.png`
- **F1-Score Curves**: `BoxF1_curve.png`
- **Confusion Matrices**: `confusion_matrix.png` and `confusion_matrix_normalized.png`
- **Training Metrics**: `results.png` and `results.csv`
- **Loss Curves**: Included in `results.png`

### Custom Training

To train on your own dataset:

1. **Prepare Dataset**: Organize your dataset in YOLO format
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. **Create Configuration File**: `data.yaml`
   ```yaml
   path: /path/to/dataset
   train: images/train
   val: images/val
   nc: 2
   names: ['fire', 'smoke']
   ```

3. **Run Training**:
   ```python
   from ultralytics import YOLO
   
   model = YOLO("yolo12n.pt")  # Start from pre-trained weights
   results = model.train(
       data="path/to/data.yaml",
       epochs=100,
       imgsz=640,
       batch=8,
       device=0,
       patience=20
   )
   ```

---

## ⚙️ Configuration

### Detection Thresholds

Adjust confidence and IoU thresholds to balance precision and recall:

```python
# Higher confidence = fewer false positives, more false negatives
results = model(frame, conf=0.25, iou=0.45)

# Lower confidence = more detections, potentially more false positives
results = model(frame, conf=0.15, iou=0.35)
```

### Alarm Settings

```python
# Alarm sound file path
alarm_sound = "minion_fire_alarm.mp3"

# Cooldown between Telegram notifications (seconds)
cooldown_seconds = 30  # Adjust based on your needs
```

### Video Source Configuration

```python
# Video file
cap = cv2.VideoCapture("path/to/video.mp4")

# Webcam (default camera)
cap = cv2.VideoCapture(0)

# IP Camera (RTSP stream)
cap = cv2.VideoCapture("rtsp://username:password@ip:port/stream")
```

---

## 📊 Performance Metrics

### Inference Performance

- **Inference Speed**: ~28ms per frame (384×640 resolution)
- **Preprocessing**: ~2-3ms
- **Postprocessing**: ~4-8ms
- **FPS**: ~35-40 frames per second
- **Model Size**: ~6MB (YOLOv12n nano)

### Detection Accuracy (Training Run 2)

- **mAP50**: 76.97%
- **mAP50-95**: 45.93%
- **Precision**: 74.51%
- **Recall**: 70.56%
- **F1-Score**: ~72.4%

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Dual-core 2.0 GHz | Quad-core 3.0+ GHz |
| **RAM** | 4 GB | 8 GB+ |
| **GPU** | Optional | NVIDIA GPU with CUDA |
| **Storage** | 2 GB | 5 GB+ |
| **OS** | Windows 10 / Linux / macOS | Latest version |

---

## 🤝 Contributing

We welcome contributions! This project thrives on community involvement. Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs**: Found an issue? Open a GitHub issue
- 💡 **Suggest Features**: Have an idea? Share it in discussions
- 📝 **Improve Documentation**: Help make the docs better
- 🔧 **Submit Pull Requests**: Fix bugs or add features
- ⭐ **Star the Repo**: Show your support!

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Add comments for complex logic
- Update documentation for new features
- Write clear commit messages

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License** allows you to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use
- ✅ Patent use

---

## 🙏 Acknowledgments

We extend our gratitude to:

- **[Ultralytics](https://ultralytics.com/)** for the exceptional YOLOv12 implementation and continuous improvements
- **The Open Source Community** for invaluable tools and libraries
- **Contributors and Testers** who helped improve this project
- **Researchers** whose work laid the foundation for modern object detection

### Special Thanks

- YOLO community for continuous innovation
- OpenCV team for robust computer vision tools
- All contributors who helped refine this project

---

## 📧 Contact & Support

- **Email**: youssef111ahmed111@gmail.com

For questions, suggestions, or support, please open an issue on GitHub. We aim to respond within 24-48 hours.

---

<div align="center">

### 🔥 Made with ❤️ for Safety and Security 🔥

**Protecting lives through intelligent detection**

⭐ **Star this repo if you find it helpful!** ⭐

[⬆ Back to Top](#-smoke--fire-detection-using-yolov12n)

</div>
