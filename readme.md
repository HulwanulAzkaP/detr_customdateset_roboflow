# 🔍 DETR Object Detection Project

A powerful PyTorch Lightning implementation of Detection Transformer (DETR) for object detection, with seamless Roboflow integration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🧠 Arsitektur Model DETR

Model DETR yang digunakan dalam proyek ini terdiri dari beberapa komponen utama:

1. **Backbone (ResNet-50)**: 
   - Digunakan untuk mengekstrak fitur dari gambar input.
   - Layer terakhir (`layer4`) dari ResNet dihapus untuk mempertahankan fitur spatial.
   
2. **Positional Encoding**:
   - Menambahkan informasi posisi pada fitur yang dihasilkan oleh backbone.
   - Menggunakan **sinusoidal positional encoding**.

3. **Transformer Encoder-Decoder**:
   - **Encoder**: Memproses fitur yang diberikan oleh backbone bersama dengan positional encoding.
   - **Decoder**: Menerima query embedding untuk memprediksi bounding box dan label kelas.

4. **Detection Heads**:
   - **Classification Head**: Untuk memprediksi label kelas.
   - **Bounding Box Regression Head**: Untuk memprediksi koordinat bounding box.

### ⚛️Struktur Model

```mermaid
flowchart TD
    A["Input Image + Annotations"] --> B[Data Augmentation]
    B --> C[ResNet-50 Backbone]
    C --> D["Conv Layer (2048 channels to 256 channels)"]
    D --> E[Flatten Features]
    E --> F[Positional Encoding]
    F --> G[Transformer Encoder]
    G --> H[Transformer Decoder]
    H --> I[Query Embedding]
    I --> J1[Classification Head]
    I --> J2[Bounding Box Regression Head]
    J1 --> K1[Calculate Classification Loss]
    J2 --> K2[Calculate Bounding Box Loss]
    K1 & K2 --> L[Total Loss]
    L --> M[Backpropagation]
    M --> N["Optimizer (e.g., Adam)"]
    N --> O[Save Checkpoint]
```

## 🌟 Highlights

- 🚀 Fast and efficient object detection
- 🎯 Real-time inference capabilities
- 📊 Comprehensive evaluation metrics
- 🔄 Custom dataset support via Roboflow
- 📈 TensorBoard visualization
- 🎮 CUDA-enabled GPU acceleration

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/HulwanulAzkaP/detr_customdateset_roboflow.git
cd detr_customdataset_roboflow
```

### 2️⃣ Install Dependencies
```bash
pip install torch torchvision pytorch-lightning transformers roboflow opencv-python supervision
```

## ⚙️ Configuration Steps

### 1. 🔑 Roboflow Setup

#### In `config.py`:
```python
ROBOFLOW_API_KEY = "your_api_key_here"  # Add your key here
```

#### In `src/config/default.py`:
```python
@dataclass
class DataConfig:
    roboflow_api_key: str = "your_api_key_here"  # Add your key here
    workspace: str = "your-workspace"     # Your workspace
    project: str = "your-project"         # Your project
    version: int = 1                      # Version number
```

### 2. 📁 File Paths

#### 📍 In `train.py`:
```python
MODEL_PATH = os.path.join(HOME, 'output_model')
```

#### 📍 In `inference/inference_detr.py`:
```python
model_path = "../output_model/model.safetensors" //change the folder from your project
dataset_path = "path/to/your/dataset"
```

#### 📍 In `evaluate.py`:
```python
dataset_path = "path_roboflow_project" //change the folder from your project
```

## 🚀 Usage Guide

### 📚 Training

1. Verify your dataset structure:
```
dataset/
├── train/
├── valid/
└── test/
```

2. Start training:
```bash
python train.py
```

### 📊 Evaluation

```bash
python evaluate.py
```

### 🎥 Inference

For webcam:
```bash
python inference/inference_detr.py //to use webcam un comment the webcam code
```

For video:
```bash
python inference/inference_detr.py //then input your filename include the extension(.mp4)
```

## 📂 Project Structure

```
DETR_AZKA/
├── 📁 data/
│   └── dataset.py
├── 📁 inference/
│   ├── inference_detr.py
│   └── Test.mp4
├── 📁 models/
│   └── detr.py
├── 📁 src/
│   ├── 📁 config/
│   │   └── default.py
│   ├── 📁 data_handling/
│   │   ├── dataloader.py
│   │   └── dataset.py
│   ├── 📁 model/
│   │   └── detr.py
│   └── 📁 utils/
│       ├── helpers.py
│       ├── logging_config.py
│       └── config.py
├── evaluate.py
└── train.py
```

## 📊 Training Monitoring

Launch TensorBoard:
```bash
tensorboard --logdir logs/ //optional
```

📈 View:
- Loss curves
- Validation metrics
- Learning rates
- Model predictions

## 📋 Evaluation Metrics

### 🎯 Performance Metrics
- Average Precision (AP)
- IoU thresholds
- Object size performance
- Recall rates

### 📸 Visual Output
- Bounding box visualization
- Ground truth comparison
- Confidence scoring

## ❗ Troubleshooting

### 🔑 Roboflow Issues
- Verify API key
- Check workspace/project names
- Confirm version number

### 💾 Storage Issues
- Check disk space
- Verify write permissions

### 🎮 CUDA Problems
- Reduce batch size
- Enable mixed precision


## 📫 Support

Having issues? Let's solve them:

1. 📧 Contact: hulwanulazkap@gmail.com

## 📄 License

This project is under the MIT License. See [LICENSE](LICENSE) for details.

---
⭐ If this project helped you, please consider giving it a star!
