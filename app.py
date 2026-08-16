import os
import io
import base64
import json
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.detector import EmotionDetector
from src.dataset import EMOTION_CLASSES

app = Flask(__name__)

# Check available models
MODEL_PATHS = {
    'resnet18': 'models/best_transfer_resnet18.pth',
    'custom': 'models/best_custom_cnn.pth'
}

active_model_key = 'resnet18' if os.path.exists('models/best_transfer_resnet18.pth') else 'custom'
detector_instance = None

def get_detector():
    global detector_instance, active_model_key
    if detector_instance is None:
        model_path = MODEL_PATHS.get(active_model_key)
        if not os.path.exists(model_path):
            # Fallback if specific file doesn't exist yet
            model_path = None
        
        m_type = 'transfer' if active_model_key == 'resnet18' else 'custom'
        b_name = 'resnet18'
        detector_instance = EmotionDetector(model_path=model_path, model_type=m_type, backbone_name=b_name)
    return detector_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models/<path:filename>')
def serve_model_asset(filename):
    return send_from_directory('models', filename)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'active_model': active_model_key,
        'available_models': [k for k, v in MODEL_PATHS.items() if os.path.exists(v)]
    })

@app.route('/api/config')
def get_config():
    return jsonify({
        'active_model': active_model_key,
        'available_models': {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    })

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    global detector_instance, active_model_key
    data = request.get_json() or {}
    model_key = data.get('model', 'custom')
    
    if model_key in MODEL_PATHS:
        model_path = MODEL_PATHS[model_key]
        if os.path.exists(model_path):
            active_model_key = model_key
            m_type = 'transfer' if active_model_key == 'resnet18' else 'custom'
            detector_instance = EmotionDetector(model_path=model_path, model_type=m_type, backbone_name='resnet18')
            return jsonify({'status': 'success', 'active_model': active_model_key})
        else:
            return jsonify({'status': 'error', 'message': f'Model file {model_path} not found.'}), 404
    return jsonify({'status': 'error', 'message': 'Invalid model requested'}), 400

@app.route('/api/predict_image', methods=['POST'])
def predict_image():
    detector = get_detector()
    
    if 'file' in request.files:
        file = request.files['file']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif request.json and 'image_base64' in request.json:
        b64_str = request.json['image_base64']
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return jsonify({'error': 'No image provided'}), 400

    if img_bgr is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    annotated_frame, faces = detector.detect_and_predict(img_bgr)

    # Encode annotated image to base64 for display
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'faces_detected': len(faces),
        'faces': faces,
        'annotated_image': f"data:image/jpeg;base64,{annotated_b64}"
    })

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    detector = get_detector()
    data = request.get_json() or {}
    b64_str = data.get('frame', '')

    if not b64_str:
        return jsonify({'error': 'Empty frame'}), 400

    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400

        annotated_frame, faces = detector.detect_and_predict(img_bgr)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'faces_count': len(faces),
            'faces': faces,
            'annotated_frame': f"data:image/jpeg;base64,{annotated_b64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_data = {}
    for key in ['custom_cnn', 'custom_cnn_eval', 'transfer_resnet18']:
        eval_path = f'models/evaluation_{key}.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                data = json.load(f)
                mapped_key = 'custom_cnn' if key == 'custom_cnn_eval' else key
                metrics_data[mapped_key] = data
    return jsonify(metrics_data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting DeepFER Flask Web Server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

