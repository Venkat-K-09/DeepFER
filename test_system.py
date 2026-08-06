import os
import cv2
import numpy as np
import torch
from src.detector import EmotionDetector
from src.dataset import EMOTION_CLASSES

def test_pipeline():
    print("--- System Verification Test ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # Create dummy face image (synthetic test)
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (200, 200), 100, (255, 255, 255), -1)

    print("Initializing EmotionDetector...")
    detector = EmotionDetector(model_type='custom')
    annotated, results = detector.detect_and_predict(dummy_img)
    print("Detector initialized successfully.")
    print("Verification complete!")

if __name__ == '__main__':
    test_pipeline()
