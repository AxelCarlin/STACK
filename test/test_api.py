import unittest
import asyncio
from fastapi.testclient import TestClient
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import create_app

class TestAPI(unittest.TestCase):
    """Test suite for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("status", data)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
    
    def test_prediction_without_model(self):
        """Test prediction when no model is loaded"""
        prediction_data = {
            "features": [[1, 2, 3, 4, 5] for _ in range(6)],
            "patient_id": "test_patient"
        }
        
        response = self.client.post("/api/v1/predict", json=prediction_data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Model not loaded", response.json()["detail"])
    
    def test_model_info_without_model(self):
        """Test model info when no model is loaded"""
        response = self.client.get("/api/v1/model_info")
        self.assertEqual(response.status_code, 400)
    
    def test_visualizations_endpoint(self):
        """Test visualizations listing"""
        response = self.client.get("/api/v1/visualizations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("visualizations", data)
    
    def test_training_status(self):
        """Test training status endpoint"""
        response = self.client.get("/api/v1/training_status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("progress", data)
    
    def test_cleanup_endpoint(self):
        """Test cleanup endpoint"""
        response = self.client.delete("/api/v1/cleanup")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
