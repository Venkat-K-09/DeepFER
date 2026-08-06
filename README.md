# DeepFER: Facial Emotion Recognition Engine 🎭

DeepFER is a full-stack, real-time **Facial Emotion Recognition (FER)** web application powered by **PyTorch**, **OpenCV**, and **Flask**. It uses OpenCV Haar Cascades for real-time face detection and custom Deep Convolutional Neural Networks (CNN) to classify facial expressions across 7 universal emotion categories.

---

## 🌟 Key Features

- 🎥 **Real-time Live Webcam Stream**: Low-latency emotion detection right inside your browser.
- 🖼️ **Image File Classifier**: Drag-and-drop support for static images (JPG, PNG, WEBP) with annotated bounding boxes and confidence score breakdowns.
- 🧠 **Dual Model Support**: Switch seamlessly between a **Custom Deep CNN** architecture and fine-tuned **ResNet-18 Transfer Learning**.
- 📊 **Model Benchmarks Dashboard**: Built-in confusion matrix visualizer and detailed precision/recall/F1 metrics evaluation.
- 🐳 **Docker & Production Ready**: Pre-configured multi-stage `Dockerfile`, `docker-compose.yml`, and `Procfile` for one-command deployment.

---

## 📂 Project Structure

```text
DeepFER/
├── app.py                   # Flask Web Server & API Endpoints
├── Dockerfile               # Multi-stage Docker container build configuration
├── docker-compose.yml       # Docker Compose setup for local container orchestration
├── Procfile                 # Cloud deployment startup descriptor (Render / Heroku)
├── requirements.txt         # Dependencies (PyTorch, OpenCV-Headless, Flask, Gunicorn)
├── test_system.py           # PyTorch & OpenCV pipeline sanity check
├── test_app.py              # Automated test suite for Flask endpoints
├── train.py                 # Neural network training script
├── evaluate.py              # Model evaluation and confusion matrix generation
├── src/                     # Core Machine Learning Modules
│   ├── dataset.py           # Data loading & PyTorch Dataset class
│   ├── detector.py          # Real-time face detection & emotion prediction engine
│   ├── inference.py         # Checkpoint loader utilities
│   ├── models.py            # Custom CNN & Transfer Learning PyTorch architectures
│   └── utils.py             # Evaluation metrics and plotting helper functions
├── models/                  # Saved Model Weights & Benchmark Artifacts
│   ├── best_custom_cnn.pth  # Trained Custom CNN weights
│   ├── evaluation_custom_cnn_eval.json
│   └── confusion_matrix_custom_cnn_eval.png
├── static/                  # Frontend UI Assets
│   ├── app.js               # Webcam stream controller & API client logic
│   └── style.css            # Dark mode glassmorphic UI stylesheet
└── templates/
    └── index.html           # Single Page Application template
```

---

## 🚀 Local Quickstart

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Setup Environment
```bash
# Clone repository
git clone https://github.com/<YOUR-USERNAME>/DeepFER.git
cd DeepFER

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify System Installation
```bash
python test_system.py
python -m unittest test_app.py
```

### 3. Start Web Application
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 🐳 Running with Docker

You can run the entire application using Docker without installing Python or C++ libraries manually:

```bash
# Build and launch container
docker compose up --build
```
Access the application at **`http://localhost:5000`**.

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main Web Interface |
| `/api/health` | `GET` | Health status probe for load balancers |
| `/api/config` | `GET` | Returns active model configuration |
| `/api/predict_image` | `POST` | Upload file (`file`) or base64 (`image_base64`) for face prediction |
| `/api/predict_frame` | `POST` | Process single webcam frame (`frame` base64 string) |
| `/api/switch_model` | `POST` | Switch active model (`{"model": "custom"}`) |
| `/api/metrics` | `GET` | Retrieve JSON benchmark metrics |

---

## 🌐 Web Deployment Guide (Render)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - DeepFER application"
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/DeepFER.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Log into [Render.com](https://render.com).
   - Click **New +** → **Web Service**.
   - Connect your `DeepFER` GitHub repository.
   - Select **Docker** environment (Render uses your [Dockerfile](file:///c:/Users/lenovo/OneDrive/Desktop/DeepFER/Dockerfile)).
   - Click **Deploy Web Service**.

Render will automatically build the container and provide your live application link!

---

## 📄 License

Distributed under the [MIT License](file:///c:/Users/lenovo/OneDrive/Desktop/DeepFER/LICENSE).
