from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

@dataclass
class SepsisConfig:
    # Data configuration
    VITAL_SIGNS: List[str] = None
    LAB_VALUES: List[str] = None
    EXCLUDE_COLS: set = None
    CATEGORICAL_COLS: List[str] = None
    NORMAL_RANGES: Dict[str, Tuple[float, float]] = None
    TIME_WINDOW: int = 6
    
    # Model configuration
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # XGBoost parameters
    XGBOOST_PARAMS: Dict = None
    
    # Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    CHARTS_DIR: str = "charts"
    LOGS_DIR: str = "logs"
    
    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    def __post_init__(self):
        if self.VITAL_SIGNS is None:
            self.VITAL_SIGNS = ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'O2Sat']
        
        if self.LAB_VALUES is None:
            self.LAB_VALUES = ['WBC', 'Lactate', 'pH', 'HCO3', 'BaseExcess', 'PaCO2', 
                              'Creatinine', 'BUN', 'Bilirubin_total', 'Platelets']
        
        if self.EXCLUDE_COLS is None:
            self.EXCLUDE_COLS = {'SepsisLabel', 'Patient_ID', 'Unit1', 'Unit2', 'Unnamed: 0', 'Hour'}
        
        if self.CATEGORICAL_COLS is None:
            self.CATEGORICAL_COLS = ['Gender']
        
        if self.NORMAL_RANGES is None:
            self.NORMAL_RANGES = {
                'HR': (60, 100), 'Temp': (36.0, 37.5), 'SBP': (90, 140),
                'DBP': (60, 90), 'Resp': (12, 20), 'O2Sat': (95, 100),
                'pH': (7.35, 7.45), 'Lactate': (0.5, 2.2), 'WBC': (4.0, 11.0)
            }
        
        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 8,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_lambda': 3.0,
                'reg_alpha': 0.2,
                'scale_pos_weight': 4,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'n_jobs': -1,
                'random_state': self.RANDOM_STATE,
                'early_stopping_rounds': 50,
                'verbosity': 0
            }
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.CHARTS_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# Global config instance
config = SepsisConfig()