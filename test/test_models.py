# sepsis_detection_system/tests/test_models.py
import unittest
import numpy as np
from models.xgboost_advanced import AdvancedXGBModel
from models.cnn_model import OptimizedCNNModel

class TestModels(unittest.TestCase):
    def test_xgboost_model(self):
        model = AdvancedXGBModel()
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        self.assertTrue(model.is_fitted)
        preds = model.predict_proba(X)
        self.assertEqual(len(preds), 100)

    def test_cnn_model(self):
        model = OptimizedCNNModel(num_features=10, time_window=6)
        X = np.random.rand(100, 6, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        self.assertTrue(model.is_fitted)
        preds = model.predict_proba(X)
        self.assertEqual(len(preds), 100)

if __name__ == '__main__':
    unittest.main()