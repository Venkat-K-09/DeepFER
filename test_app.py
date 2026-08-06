import unittest
import json
import base64
import cv2
import numpy as np
from app import app

class TestDeepFERApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_health_check(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertIn('active_model', data)

    def test_config_endpoint(self):
        response = self.client.get('/api/config')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('active_model', data)
        self.assertIn('available_models', data)

    def test_predict_image_file(self):
        # Create a simple 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        import io
        response = self.client.post(
            '/api/predict_image',
            data={'file': (io.BytesIO(buffer.tobytes()), 'test.jpg', 'image/jpeg')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('annotated_image', data)

    def test_predict_frame_b64(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        b64_str = base64.b64encode(buffer.tobytes()).decode('utf-8')

        response = self.client.post(
            '/api/predict_frame',
            data=json.dumps({'frame': f"data:image/jpeg;base64,{b64_str}"}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('annotated_frame', data)
        self.assertIn('faces_count', data)

    def test_metrics_endpoint(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)

    def test_serve_model_asset(self):
        response = self.client.get('/models/confusion_matrix_custom_cnn_eval.png')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
