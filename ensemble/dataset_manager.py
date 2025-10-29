import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnsembleDatasetManager:
    """Manages datasets for ensemble models"""
    
    def __init__(self, output_dir: str = "ensemble_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.base_models_predictions = {}
        self.meta_model_features = None
        self.feature_metadata = {}
        
    def save_base_model_predictions(self, model_name: str, fold_predictions: np.ndarray, 
                                  test_predictions: np.ndarray, feature_importance: np.ndarray = None):
        """Save base model predictions and metadata"""
        model_data = {
            'fold_predictions': fold_predictions,
            'test_predictions': test_predictions,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_models_predictions[model_name] = model_data
        
        # Save to disk
        save_path = self.output_dir / f"{model_name}_predictions.npz"
        save_data = {k: v for k, v in model_data.items() if v is not None}
        np.savez(save_path, **save_data)
        logger.info(f"Predictions for {model_name} saved to {save_path}")
    
    def create_meta_model_dataset(self, target: np.ndarray, include_interactions: bool = True) -> pd.DataFrame:
        """Create advanced meta-model dataset"""
        if not self.base_models_predictions:
            raise ValueError("No base model predictions available")
        
        # Extract fold predictions
        base_predictions = []
        model_names = []
        
        for model_name, data in self.base_models_predictions.items():
            base_predictions.append(data['fold_predictions'])
            model_names.append(model_name)
        
        # Verificar número de modelos
        if len(model_names) != 6:
            logger.error(f"Expected 6 base models, got {len(model_names)}: {model_names}")
            raise ValueError(f"Expected 6 base models, got {len(model_names)}")
        
        base_predictions = np.column_stack(base_predictions)
        
        # Create base DataFrame
        meta_df = pd.DataFrame(base_predictions, columns=[f'{name}_pred' for name in model_names])
        
        # Statistical features
        meta_df['ensemble_mean'] = base_predictions.mean(axis=1)
        meta_df['ensemble_std'] = base_predictions.std(axis=1)
        meta_df['ensemble_min'] = base_predictions.min(axis=1)
        meta_df['ensemble_max'] = base_predictions.max(axis=1)
        meta_df['ensemble_range'] = meta_df['ensemble_max'] - meta_df['ensemble_min']
        meta_df['ensemble_median'] = np.median(base_predictions, axis=1)
        
        # Consensus features
        meta_df['high_confidence'] = (base_predictions > 0.7).sum(axis=1) / len(model_names)
        meta_df['low_confidence'] = (base_predictions < 0.3).sum(axis=1) / len(model_names)
        meta_df['disagreement'] = np.abs(base_predictions - base_predictions.mean(axis=1)[:, np.newaxis]).mean(axis=1)
        
        # Ranking features
        for i, name in enumerate(model_names):
            meta_df[f'{name}_rank'] = base_predictions[:, i].argsort().argsort() / len(base_predictions)
        
        if include_interactions:
            # Model interactions
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    meta_df[f'{name1}_{name2}_diff'] = base_predictions[:, i] - base_predictions[:, j]
                    meta_df[f'{name1}_{name2}_product'] = base_predictions[:, i] * base_predictions[:, j]
        
        # Verificar número de características
        expected_features = 51  # 6 (pred) + 6 (stats) + 3 (consensus) + 6 (rank) + 30 (interactions)
        if meta_df.shape[1] != expected_features:
            logger.error(f"Feature shape mismatch, expected: {expected_features}, got {meta_df.shape[1]}")
            raise ValueError(f"Feature shape mismatch, expected: {expected_features}, got {meta_df.shape[1]}")
        
        # Save metadata
        self.feature_metadata = {
            'base_models': model_names,
            'meta_features': list(meta_df.columns),
            'target_distribution': {
                'positive_class': int(target.sum()),
                'negative_class': int(len(target) - target.sum()),
                'class_ratio': float(target.mean())
            },
            'created_at': datetime.now().isoformat()
        }
        
        self.meta_model_features = meta_df
        
        # Save datasets
        dataset_path = self.output_dir / "meta_model_dataset.csv"
        meta_df.to_csv(dataset_path, index=False)
        
        metadata_path = self.output_dir / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
        
        logger.info(f"Meta-model dataset created with {len(meta_df.columns)} features")
        logger.info(f"Saved to {dataset_path}")
        
        return meta_df
    
    def analyze_meta_features(self, target: np.ndarray) -> Dict:
        """Analyze meta-model features"""
        if self.meta_model_features is None:
            raise ValueError("No meta-model dataset available")
        
        analysis = {}
        
        # Feature correlations with target
        correlations = {}
        for col in self.meta_model_features.columns:
            corr = np.corrcoef(self.meta_model_features[col], target)[0, 1]
            correlations[col] = corr
        
        analysis['feature_correlations'] = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        analysis['feature_stats'] = self.meta_model_features.describe().to_dict()
        
        # Mutual information (if sklearn available)
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(self.meta_model_features, target, random_state=42)
            analysis['mutual_information'] = dict(zip(self.meta_model_features.columns, mi_scores))
        except ImportError:
            logger.warning("sklearn not available for mutual information calculation")
        
        return analysis