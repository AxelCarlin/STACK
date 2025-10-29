from pydantic import BaseModel
from typing import List, Optional, Dict
        
class PredictionRequest(BaseModel):
    """Request model for individual predictions"""
    features: List[List[float]]  # Shape: [time_window, num_features]
    patient_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    sepsis_probability: float
    risk_level: str
    confidence: float
    explanation: Optional[Dict] = None

class TrainingStatus(BaseModel):
    """Training status model"""
    status: str
    progress: float
    message: str
    metrics: Optional[Dict] = None

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    base_models: List[str]
    meta_model: str
    feature_count: int
    training_metrics: Optional[Dict] = None
