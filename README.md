# 🎯 InsightFace Demo - Face Recognition System

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FAISS](https://img.shields.io/badge/FAISS-accelerated-green.svg)](https://github.com/facebookresearch/faiss)
[![InsightFace](https://img.shields.io/badge/InsightFace-powered-red.svg)](https://github.com/deepinsight/insightface)

**A production-ready face recognition system with real-time processing, similarity search, and modern GUI interface.**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-usage) • [🔧 API Reference](#-development) • [💡 Examples](#-examples)

</div>

---

## 🎨 Overview

**InsightFace Demo** is a comprehensive face recognition system built with state-of-the-art deep learning models. It combines the power of **InsightFace** for accurate face recognition, **FAISS** for lightning-fast vector similarity search, and **PySide6** for a modern desktop GUI experience.

## ✨ Features

### 🔍 **Core Functionality**
- **🆚 Face Verification**: Compare two faces with confidence scores and similarity metrics
- **🔎 Face Search**: Find similar faces in large image collections using vector similarity
- **📹 Real-time Recognition**: Live face detection and recognition from webcam streams
- **👥 Multi-face Detection**: Process multiple faces in a single image simultaneously

### ⚡ **Performance & Scalability**
- **🚀 FAISS Integration**: Ultra-fast approximate nearest neighbor search
- **🎮 GPU Acceleration**: CUDA support for high-throughput processing
- **💾 Memory Efficient**: Optimized memory usage for large-scale deployments
- **📊 Batch Processing**: Process multiple images efficiently

### 🖥️ **User Interface**
- **🎨 Modern GUI**: Clean, intuitive desktop interface built with PySide6
- **📱 Responsive Design**: Adaptive layouts for different screen sizes
- **📈 Real-time Metrics**: Live performance monitoring and statistics

---

## 🏗️ Architecture

### 📁 **Project Structure**

```
insightface-demo/
├── 📂 src/faceid/                 # Main application package
│   ├── 🎮 __main__.py            # Application entry point
│   ├── ⚙️ configs/               # Configuration files
│   │   └── default.yaml          # Default settings (models, paths, thresholds)
│   ├── 🧠 core/                  # Core engine and business logic
│   │   ├── __init__.py           # Package initialization
│   │   └── engine.py             # FaceEngine - main processing logic
│   ├── 🛠️ scripts/               # Utility scripts and tools
│   │   ├── build_facebank_faiss.py        # Build identity database index
│   │   ├── build_library_faiss.py         # Build image library index
│   ├── 🖥️ ui/                    # User interface components
│   │   ├── main_window.py        # Main application window
│   │   ├── verify_tab.py         # Face verification interface
│   │   ├── library_tab.py        # Image library search interface
│   │   └── realtime_tab.py       # Real-time webcam recognition
│   └── 🔧 utils/                 # Utility functions and helpers
│       ├── face_utils.py         # Face detection and processing
│       └── image_utils.py        # Image loading and manipulation
├── 📂 data/                      # Data storage directory
│   ├── facebank/                 # Known identity database
│   │   ├── person_1/             # Individual identity folders
│   │   │   ├── 01.jpg           # Reference images for each person
│   │   │   └── ...
│   │   └── person_n/
│   └── library/                  # General image collection
│       ├── image_001.jpg         # Searchable image database
│       └── ...
├── 📂 outputs/                   # Generated indices and metadata
│   ├── facebank.index           # FAISS index for identity database
│   ├── labels.json              # Identity labels mapping
│   ├── library.index            # FAISS index for image library
│   ├── library_meta.json        # Image metadata and paths
│   └── similarity_matrix_*.csv  # Precomputed similarity matrices
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
```

### 🔧 **Core Components**

#### **🧠 FaceEngine (`core/engine.py`)**
The central processing unit that orchestrates all face recognition operations:

- **Face Detection**: Utilizes InsightFace models for accurate face detection
- **Feature Extraction**: Generates 512-dimensional face embeddings
- **Similarity Computation**: Calculates cosine similarity between face vectors
- **FAISS Integration**: Manages vector indices for fast similarity search
- **Threshold Management**: Configurable similarity thresholds for matching decisions

#### **🎨 User Interface (`ui/`)**
Modern PySide6-based desktop application with three main modules:

- **`verify_tab.py`**: Two-image face verification with confidence scores
- **`library_tab.py`**: Search similar faces in large image collections
- **`realtime_tab.py`**: Live webcam face recognition and identification
- **`main_window.py`**: Main application container with tabbed interface

#### **🛠️ Processing Scripts (`scripts/`)**
Batch processing tools for database preparation:

- **`build_facebank_faiss.py`**: Creates searchable index from identity database
- **`build_library_faiss.py`**: Builds vector index for general image library

#### **🔧 Utility Modules (`utils/`)**
Reusable helper functions:

- **`face_utils.py`**: Face detection, embedding extraction, and normalization
- **`image_utils.py`**: Image I/O, preprocessing, and format conversions
---

## 🚀 Quick Start

### 📋 **Prerequisites**

- **Python 3.11+**: Ensure you have Python 3.11 or higher installed
- **Linux**: Ubuntu 20.04+ / CentOS 8+ / Fedora 35+ recommended
- **Git**: For cloning the repository
- **Webcam**: Optional, for real-time face recognition features

### 🛠️ **Installation Guide**

#### **Step 1: Clone Repository and Setup Virtual Environment**

```bash
# Clone the repository
git clone https://github.com/nguyenmaiductrong/insightface-demo.git
cd insightface-demo

# Create Python virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### **Step 2: Install Dependencies**

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

#### **Step 3: Prepare Data Structure**

Create the proper data directory structure and add your images:

```bash
# Create required directories
mkdir -p data/facebank data/library outputs

# Example: Add known identities to facebank
# Each person should have their own folder with reference images
mkdir -p data/facebank/john_doe
mkdir -p data/facebank/jane_smith

# Copy reference images (5-10 images per person recommended)
# data/facebank/john_doe/01.jpg
# data/facebank/john_doe/02.jpg
# data/facebank/jane_smith/01.jpg
# data/facebank/jane_smith/02.jpg

# Add general images to library for similarity search
# Copy your image collection to data/library/
# data/library/event_photo_001.jpg
# data/library/event_photo_002.jpg
```

**Data Structure Example:**
```
data/
├── facebank/              # Known identities database
│   ├── person_1/          # Individual person folder
│   │   ├── 01.jpg        # Reference image 1
│   │   ├── 02.jpg        # Reference image 2
│   │   └── ...
│   └── person_2/
│       ├── 01.jpg
│       └── ...
└── library/               # General image collection
    ├── photo_001.jpg
    ├── photo_002.jpg
    └── ...
```

#### **Step 4: Build Library FAISS Index**

Generate the searchable index for your image library:

```bash
# Build FAISS index for image library (similarity search)
python -m src.faceid.scripts.build_library_faiss

# Expected output:
# Applied providers: ['CPUExecutionProvider']...
# Building library embeddings: 100%|██████████| 150/150 [01:47<00:00,  1.40img/s]
# Built library index with 831 faces
# Index -> /path/to/outputs/library.index
# Meta  -> /path/to/outputs/library_meta.json
```

#### **Step 5: Build Facebank FAISS Index**

Create the identity database index for known persons:

```bash
# Build FAISS index for facebank (identity recognition)
python -m src.faceid.scripts.build_facebank_faiss

# Expected output:
# Applied providers: ['CPUExecutionProvider']...
# Processing facebank: 100%|██████████| 4/4 [00:15<00:00,  3.85s/person]
# Built facebank with 4 identities, 127 total faces
# Index -> /path/to/outputs/facebank.index
# Labels -> /path/to/outputs/labels.json
```

#### **Step 6: Launch Application**

Start the GUI application:

```bash
# Launch the main application
python -m src.faceid

# Alternative: Direct execution
python src/faceid/__main__.py
```

The application window will open with three tabs:
- **Face Verification**: Compare two face images
- **Library Search**: Find similar faces in your image collection  
- **Real-time Recognition**: Live webcam face recognition

### 📊 **Performance Tips**

- **Image Quality**: Use high-resolution images (>300px face size) for better accuracy
- **Lighting**: Ensure good lighting conditions in reference images
- **Multiple Angles**: Include 5-10 reference images per person from different angles
- **Batch Processing**: For large datasets, consider processing in smaller batches

---

## 📖 Usage

### 🔍 **Face Verification**
1. Open the **"Face Verification"** tab
2. Click **"Choose First Image"** and select your first image
3. Click **"Choose Second Image"** and select your second image  
4. Click **"Compare"** to get similarity score and match result

### 🎯 **Library Search**
1. Open the **"Library Search"** tab
2. Browse and select your library directory (default: `data/library`)
3. Click **"Refresh"** to load image thumbnails
4. Click **"Choose Query Image"** and select the image you want to search for
5. Click **"Search"** to find similar faces in your library
6. Results will show only images above the similarity threshold

### 📹 **Real-time Recognition**
1. Open the **"Real-time Recognition"** tab
2. Click **"Start Camera"** to begin webcam capture
3. The system will detect and identify faces in real-time
4. Known identities will be labeled with their names
5. Click **"Stop Camera"** to end the session

### ⚙️ **Threshold Calibration**

Optimize the similarity threshold for your specific facebank to improve recognition accuracy:

```bash
# Run threshold calibration with default settings
python3 -m src.faceid.scripts.calibrate_threshold

# Or with custom paths
python3 -m src.faceid.scripts.calibrate_threshold \
    --facebank-dir data/facebank \
    --config src/faceid/configs/default.yaml \
    --output-dir outputs
```

**What it does:**
- Analyzes all face images in your facebank
- Computes similarity scores between same/different person pairs
- Finds optimal threshold using F1-Score maximization
- Generates visualization chart and recommendations

**Output files:**
- `outputs/threshold_analysis.png` - F1-Score vs Threshold visualization
- `outputs/optimal_threshold.txt` - Recommended threshold configuration

**Example output:**
```
THRESHOLD OPTIMIZATION RESULTS
==================================================
Current threshold:     0.36
Optimal threshold:     0.2946
==================================================
PERFORMANCE METRICS:
Accuracy:              97.4% 
Precision:             95.2%
Recall:                98.1%
F1-Score:              96.6%
==================================================
RECOMMENDATION: Lower threshold from 0.36 to 0.2946
BENEFIT: Better recall (catch more true matches)
```

**How to apply the results:**
1. Update `src/faceid/configs/default.yaml`:
   ```yaml
   threshold: 0.2946  # Use the recommended value
   ```
2. Restart the application to use the new threshold

---
