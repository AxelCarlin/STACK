# ==============================================================================
# models/base_models.py
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all ML models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model - base implementation ignores validation data"""
        if self.model is None:
            raise ValueError(f"Model {self.name} is not initialized")
        
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        logger.info(f"Model {self.name} trained successfully")

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not trained")
        if self.model is None:
            raise ValueError(f"Model {self.name} is not initialized")
        return self.model.predict_proba(X)[:, 1]

class OptimizedLGBMModel(BaseModel):
    """Optimized LightGBM model for medical data"""
    
    def __init__(self):
        super().__init__("LightGBM_Medical")
        try:
            self.model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.02,
                max_depth=7,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                reg_alpha=0.1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            logger.info(f"Initialized {self.name}")
        except Exception as e:
            logger.error(f"Error initializing LGBM: {e}")
            raise

class OptimizedRFModel(BaseModel):
    """Optimized Random Forest model"""
    
    def __init__(self):
        super().__init__("RandomForest_Medical")
        try:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=42
            )
            logger.info(f"Initialized {self.name}")
        except Exception as e:
            logger.error(f"Error initializing RF: {e}")
            raise

class OptimizedLRModel(BaseModel):
    """Optimized Logistic Regression model"""
    
    def __init__(self):
        super().__init__("LogisticRegression_Medical")
        try:
            self.model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.1,
                penalty='elasticnet',
                l1_ratio=0.5,
                solver='saga',
                random_state=42
            )
            logger.info(f"Initialized {self.name}")
        except Exception as e:
            logger.error(f"Error initializing LR: {e}")
            raise