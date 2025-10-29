import logging
from xgboost import XGBClassifier
from typing import Optional, Dict
from config.settings import config  # Corrección aquí
from models.base_models import BaseModel

logger = logging.getLogger(__name__)

class AdvancedXGBModel(BaseModel):
    """Advanced XGBoost model with early stopping and validation"""
    
    def __init__(self, name: str = "AdvancedXGBoost", params: Optional[Dict] = None):
        super().__init__(name)
        self.params = params or config.XGBOOST_PARAMS.copy()  # Ahora usa config.XGBOOST_PARAMS correctamente
        self.training_history = {}
        # No inicializar self.model aquí porque XGBoost necesita parámetros dinámicos
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Train with optional validation set for early stopping"""
        fit_params = self.params.copy()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        else:
            fit_params.pop('early_stopping_rounds', None)
        
        self.model = XGBClassifier(**fit_params)
        
        if eval_set:
            self.model.fit(
                X, y, 
                eval_set=eval_set,
                verbose=verbose
            )
            self.training_history = self.model.evals_result_
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        logger.info(f"Model {self.name} trained successfully")