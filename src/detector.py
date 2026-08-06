import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import EMOTION_CLASSES, IDX_TO_CLASS
from src.models import build_model

# Color mapping for emotions (BGR format for OpenCV)
EMOTION_COLORS = {
    'angry': (0, 0, 220),       # Bright Red
    'disgust': (0, 140, 255),    # Orange
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 220, 0),       # Green
    'neutral': (200, 200, 200),  # Light Gray
    'sad': (220, 100, 0),       # Blue
    'surprise': (0, 215, 255)   # Yellow
}

class EmotionDetector:
    """
    Real-time face detection and facial emotion recognition engine using OpenCV and PyTorch.
    """
    def __init__(self, model_path=None, model_type='custom', backbone_name='resnet18', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(model_type=model_type, backbone_name=backbone_name).to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
            
        self.model.eval()

        # OpenCV Haar Cascade Face Detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Image Transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_and_predict(self, image_np):
        """
        Detects faces in OpenCV image (BGR uint8) and predicts emotion for each face.
        Returns:
            annotated_frame: numpy array (BGR)
            face_results: list of dicts with bounding boxes, predictions, and confidence
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        annotated_frame = image_np.copy()
        results = []

        for (x, y, w, h) in faces:
            # Crop face region
            face_roi = image_np[y:y+h, x:x+w]
            
            # Convert BGR to RGB PIL Image
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            # Preprocess tensor
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            top_idx = int(np.argmax(probs))
            top_emotion = IDX_TO_CLASS[top_idx]
            confidence = float(probs[top_idx])

            # Prepare full emotion probabilities dict
            prob_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}

            results.append({
                'box': [int(x), int(y), int(w), int(h)],
                'top_emotion': top_emotion,
                'confidence': confidence,
                'probabilities': prob_dict
            })

            # Draw visual bounding box and label
            color = EMOTION_COLORS.get(top_emotion, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            label_str = f"{top_emotion.capitalize()}: {confidence*100:.1f}%"
            
            # Background rectangle for label text
            (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x, y - text_h - 10),
                (x + text_w + 6, y),
                color,
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame, label_str, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

        return annotated_frame, results

if __name__ == '__main__':
    detector = EmotionDetector()
    sample_img = np.zeros((300, 300, 3), dtype=np.uint8)
    frame, results = detector.detect_and_predict(sample_img)
    print(f"Detector initialized successfully. Detected faces in blank image: {len(results)}")
