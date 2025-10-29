import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate data integrity and format"""
    
    @staticmethod
    def validate_sepsis_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate sepsis dataset"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "info": {}
        }
        
        try:
            # Basic structure checks
            required_cols = ['Patient_ID', 'Hour']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                validation_result["errors"].append(f"Missing required columns: {missing_required}")
                validation_result["is_valid"] = False
            
            # Data type checks
            if 'Patient_ID' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Patient_ID']):
                    validation_result["warnings"].append("Patient_ID is not numeric")
            
            if 'Hour' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Hour']):
                    validation_result["warnings"].append("Hour is not numeric")
            
            # Missing data analysis
            missing_percentages = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percentages[missing_percentages > 50]
            
            if not high_missing.empty:
                validation_result["warnings"].append(f"Columns with >50% missing data: {high_missing.to_dict()}")
            
            # Summary info
            validation_result["info"] = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "unique_patients": df['Patient_ID'].nunique() if 'Patient_ID' in df.columns else 0,
                "date_range": {
                    "min_hour": df['Hour'].min() if 'Hour' in df.columns else None,
                    "max_hour": df['Hour'].max() if 'Hour' in df.columns else None
                },
                "missing_data_summary": missing_percentages.describe().to_dict()
            }
            
            # Sepsis label analysis
            if 'SepsisLabel' in df.columns:
                sepsis_counts = df['SepsisLabel'].value_counts()
                validation_result["info"]["sepsis_distribution"] = sepsis_counts.to_dict()
                
                # Check for class imbalance
                if len(sepsis_counts) == 2:
                    imbalance_ratio = sepsis_counts.min() / sepsis_counts.max()
                    if imbalance_ratio < 0.1:
                        validation_result["warnings"].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f})")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    @staticmethod
    def validate_prediction_input(features: List[List[float]], expected_shape: Optional[Tuple] = None) -> Dict[str, Any]:
        """Validate prediction input format"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "info": {}
        }
        
        try:
            # Convert to numpy for analysis
            features_array = np.array(features)
            
            # Shape validation
            if expected_shape and features_array.shape != expected_shape:
                validation_result["errors"].append(
                    f"Shape mismatch: expected {expected_shape}, got {features_array.shape}"
                )
                validation_result["is_valid"] = False
            
            # Check for invalid values
            if np.any(np.isnan(features_array)):
                validation_result["errors"].append("NaN values detected in features")
                validation_result["is_valid"] = False
            
            if np.any(np.isinf(features_array)):
                validation_result["errors"].append("Infinite values detected in features")
                validation_result["is_valid"] = False
            
            # Summary info
            validation_result["info"] = {
                "shape": features_array.shape,
                "data_type": str(features_array.dtype),
                "value_range": {
                    "min": float(features_array.min()),
                    "max": float(features_array.max()),
                    "mean": float(features_array.mean()),
                    "std": float(features_array.std())
                }
            }
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result

class ModelPersistence:
    """Handle model saving and loading"""
    
    @staticmethod
    def save_model_artifacts(model, scaler, metadata: Dict, base_path: str):
        """Save all model artifacts"""
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        try:
            # Save model
            model_path = base_path / "ensemble_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
            
            # Save scaler
            if scaler is not None:
                scaler_path = base_path / "scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Scaler saved to {scaler_path}")
            
            # Save metadata
            metadata_path = base_path / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            
            # Create model info file
            info = {
                "created_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "files": ["ensemble_model.pkl", "scaler.pkl", "model_metadata.json"]
            }
            
            info_path = base_path / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            return str(base_path)
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    
    @staticmethod
    def load_model_artifacts(base_path: str) -> Tuple[Any, Any, Dict]:
        """Load all model artifacts"""
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Model path {base_path} does not exist")
        
        try:
            # Load model
            model_path = base_path / "ensemble_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler
            scaler = None
            scaler_path = base_path / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Load metadata
            metadata = {}
            metadata_path = base_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model artifacts loaded from {base_path}")
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise

class PerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, log_file: str = "performance_log.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log_prediction(self, prediction: float, confidence: float, 
                      processing_time: float, metadata: Dict = None):
        """Log individual prediction performance"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": float(prediction),
            "confidence": float(confidence),
            "processing_time_ms": float(processing_time * 1000),
            "metadata": metadata or {}
        }
        
        # Append to log file
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep only recent entries (last 10000)
            if len(logs) > 10000:
                logs = logs[-10000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for recent period"""
        try:
            if not self.log_file.exists():
                return {"error": "No performance logs found"}
            
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {"error": "No log entries found"}
            
            # Filter recent logs
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            recent_logs = [
                log for log in logs 
                if datetime.fromisoformat(log['timestamp']) >= cutoff_date
            ]
            
            if not recent_logs:
                return {"error": f"No logs found in the last {days} days"}
            
            # Calculate statistics
            predictions = [log['prediction'] for log in recent_logs]
            confidences = [log['confidence'] for log in recent_logs]
            processing_times = [log['processing_time_ms'] for log in recent_logs]
            
            summary = {
                "period_days": days,
                "total_predictions": len(recent_logs),
                "prediction_stats": {
                    "mean": np.mean(predictions),
                    "std": np.std(predictions),
                    "min": np.min(predictions),
                    "max": np.max(predictions)
                },
                "confidence_stats": {
                    "mean": np.mean(confidences),
                    "std": np.std(confidences),
                    "min": np.min(confidences),
                    "max": np.max(confidences)
                },
                "performance_stats": {
                    "mean_processing_time_ms": np.mean(processing_times),
                    "max_processing_time_ms": np.max(processing_times),
                    "min_processing_time_ms": np.min(processing_times)
                },
                "risk_level_distribution": {
                    "high": sum(1 for p in predictions if p >= 0.8),
                    "medium": sum(1 for p in predictions if 0.5 <= p < 0.8),
                    "low": sum(1 for p in predictions if p < 0.5)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
