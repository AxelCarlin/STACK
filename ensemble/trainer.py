# ==============================================================================
# ensemble/trainer.py (CORREGIDO)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import logging
from typing import Dict, Tuple, List
from pathlib import Path
import gc

from data.loader import MedicalDataLoader
from data.preprocessor import SepsisDataPreprocessor
from models.base_models import OptimizedLGBMModel, OptimizedRFModel, OptimizedLRModel
from models.cnn_model import OptimizedCNNModel
from models.xgboost_advanced import AdvancedXGBModel
from ensemble.dataset_manager import EnsembleDatasetManager
from visualization.charts import SepsisVisualizer
from config.settings import config

logger = logging.getLogger(__name__)

class AdvancedSepsisEnsemble:
    def __init__(self, cv_folds: int = None, use_patient_groups: bool = True):
        self.cv_folds = cv_folds or config.CV_FOLDS
        self.use_patient_groups = use_patient_groups
        self.base_models = []
        self.meta_model = None
        self.dataset_manager = EnsembleDatasetManager()
        self.visualizer = SepsisVisualizer()
        self.oof_predictions = None
        self.base_results = {}
        
    def _initialize_models(self, num_features: int, time_window: int):
        self.base_models = [
            AdvancedXGBModel("XGBoost_Primary"),
            AdvancedXGBModel("XGBoost_Secondary", {
                **config.XGBOOST_PARAMS,
                'max_depth': 6,
                'learning_rate': 0.02,
                'subsample': 0.9
            }),
            OptimizedLGBMModel(),
            OptimizedRFModel(),
            OptimizedLRModel(),
            OptimizedCNNModel(num_features, time_window)
        ]
        
        self.meta_model = AdvancedXGBModel("MetaModel_XGBoost", {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'objective': 'binary:logistic',
            'random_state': config.RANDOM_STATE
        })
        
        logger.info(f"Initialized {len(self.base_models)} base models and meta-model")
        
    def train(self, X: np.ndarray, y: np.ndarray, patient_ids: pd.Series = None) -> Dict:
        logger.info("Starting advanced ensemble training...")
        logger.info(f"Input data shape: {X.shape}")
        
        if len(X.shape) == 3:
            num_features, time_window = X.shape[2], X.shape[1]
        else:
            raise ValueError(f"Expected 3D data but got {X.shape}")
        
        self._initialize_models(num_features, time_window)
        
        if self.use_patient_groups and patient_ids is not None:
            cv_splitter = GroupKFold(n_splits=self.cv_folds)
            cv_splits = list(cv_splitter.split(X, y, groups=patient_ids))
            logger.info(f"Using GroupKFold with {self.cv_folds} folds")
        else:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
            cv_splits = list(skf.split(X, y))
            logger.info(f"Using StratifiedKFold with {self.cv_folds} folds")
        
        self.oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training {base_model.name}...")
            fold_aucs = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if base_model.name == "CNN_Medical":
                    X_train_model, X_val_model = X_train, X_val
                else:
                    X_train_model = X_train.reshape(X_train.shape[0], -1)
                    X_val_model = X_val.reshape(X_val.shape[0], -1)
                
                if base_model.name == "CNN_Medical":
                    temp_model = OptimizedCNNModel(num_features, time_window)
                elif "XGBoost" in base_model.name:
                    temp_model = AdvancedXGBModel(base_model.name, base_model.params)
                else:
                    temp_model = type(base_model)()
                
                if "XGBoost" in base_model.name:
                    temp_model.fit(X_train_model, y_train, X_val_model, y_val)
                else:
                    temp_model.fit(X_train_model, y_train)
                
                val_preds = temp_model.predict_proba(X_val_model)
                val_preds = np.nan_to_num(val_preds, nan=0.5)
                self.oof_predictions[val_idx, i] = val_preds
                
                fold_auc = roc_auc_score(y_val, val_preds)
                fold_auprc = average_precision_score(y_val, val_preds)
                fold_aucs.append(fold_auc)
                
                logger.info(f"  Fold {fold+1} - AUC: {fold_auc:.4f}, AUPRC: {fold_auprc:.4f}")
            
            if base_model.name == "CNN_Medical":
                X_full = X
            else:
                X_full = X.reshape(X.shape[0], -1)
            
            base_model.fit(X_full, y)
            
            avg_auc = np.mean(fold_aucs)
            self.base_results[base_model.name] = {
                'cv_auc': avg_auc,
                'cv_std': np.std(fold_aucs),
                'predictions': self.oof_predictions[:, i]
            }
            
            self.dataset_manager.save_base_model_predictions(
                base_model.name,
                self.oof_predictions[:, i],
                self.oof_predictions[:, i],
                getattr(base_model, 'feature_importance_', None)
            )
            
            logger.info(f"  {base_model.name} CV-AUC: {avg_auc:.4f} ± {np.std(fold_aucs):.4f}")
            gc.collect()
        
        logger.info("Creating meta-model dataset...")
        meta_dataset = self.dataset_manager.create_meta_model_dataset(y, include_interactions=True)
        
        logger.info("Training meta-model...")
        self.meta_model.fit(meta_dataset.values, y)
        
        ensemble_preds = self.meta_model.predict_proba(meta_dataset.values)
        ensemble_auc = roc_auc_score(y, ensemble_preds)
        ensemble_auprc = average_precision_score(y, ensemble_preds)
        
        logger.info("=== FINAL ENSEMBLE RESULTS ===")
        logger.info(f"AUROC: {ensemble_auc:.4f}")
        logger.info(f"AUPRC: {ensemble_auprc:.4f}")
        
        try:
            created_files = self.visualizer.generate_comprehensive_report(
                models_results=self.base_results,
                y_true=y,
                meta_dataset=meta_dataset,
                training_history=None,
                patient_ids=patient_ids
            )
            logger.info(f"Visualizations generated: {created_files}")
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
        
        return {
            'ensemble_auc': ensemble_auc,
            'ensemble_auprc': ensemble_auprc,
            'base_results': self.base_results
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        base_predictions = []
        for model in self.base_models:
            if model.name == "CNN_Medical":
                pred = model.predict_proba(X)
            else:
                X_2d = X.reshape(X.shape[0], -1)
                pred = model.predict_proba(X_2d)
            base_predictions.append(pred)
        
        base_predictions = np.column_stack(base_predictions)
        meta_features = self._create_meta_features_for_prediction(base_predictions)
        return self.meta_model.predict_proba(meta_features)
    
    def _create_meta_features_for_prediction(self, base_predictions: np.ndarray) -> np.ndarray:
        meta_features = []
        meta_features.append(base_predictions)
        means = base_predictions.mean(axis=1).reshape(-1, 1)
        stds = base_predictions.std(axis=1).reshape(-1, 1)
        mins = base_predictions.min(axis=1).reshape(-1, 1)
        maxs = base_predictions.max(axis=1).reshape(-1, 1)
        ranges = (maxs - mins)
        medians = np.median(base_predictions, axis=1).reshape(-1, 1)
        high_conf = (base_predictions > 0.7).sum(axis=1).reshape(-1, 1) / base_predictions.shape[1]
        low_conf = (base_predictions < 0.3).sum(axis=1).reshape(-1, 1) / base_predictions.shape[1]
        all_features = [base_predictions, means, stds, mins, maxs, ranges, medians, high_conf, low_conf]
        return np.column_stack(all_features)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        ensemble_preds = self.predict(X)
        ensemble_preds = np.nan_to_num(ensemble_preds, nan=0.5)
        
        auc = roc_auc_score(y, ensemble_preds)
        auprc = average_precision_score(y, ensemble_preds)
        
        thresholds = [0.1, 0.2, 0.3, 0.5]
        metrics = {'Ensemble_AUROC': auc, 'Ensemble_AUPRC': auprc}
        
        for thresh in thresholds:
            pred_binary = (ensemble_preds >= thresh).astype(int)
            tp = np.sum((pred_binary == 1) & (y == 1))
            tn = np.sum((pred_binary == 0) & (y == 0))
            fp = np.sum((pred_binary == 1) & (y == 0))
            fn = np.sum((pred_binary == 0) & (y == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f'Sens_@{thresh}'] = sensitivity
            metrics[f'Spec_@{thresh}'] = specificity
        
        return metrics

def train_pipeline(filepath: str, nrows: int = None) -> Tuple[AdvancedSepsisEnsemble, Dict]:
    logger.info("=== SEPSIS DETECTION TRAINING PIPELINE ===")
    
    try:
        data_loader = MedicalDataLoader(filepath)
        data = data_loader.load_data(nrows=nrows)
        
        preprocessor = SepsisDataPreprocessor(data_loader)
        
        # 1. PREPROCESAR DATOS Y AJUSTAR ESCALADOR (LA CORRECCIÓN)
        # Llamamos a prepare_cnn_data UNA SOLA VEZ sobre todos los datos.
        # is_training=True asegura que el escalador se AJUSTE (fit)
        # y que las feature_columns se generen y guarden en el data_loader.
        logger.info("Preparing full dataset and fitting scaler...")
        X_full, y_full, patient_ids_full = preprocessor.prepare_cnn_data(data, is_training=True)

        logger.info(f"Full data prepared. Shape: {X_full.shape}")
        
        # 2. DIVIDIR PACIENTES
        unique_patients = patient_ids_full.unique()
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        train_mask = patient_ids_full.isin(train_patients)
        test_mask = patient_ids_full.isin(test_patients)
        
        # 3. CREAR CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
        # X_train y X_test son ahora slices del X_full ya preprocesado y escalado.
        # Ambos tendrán la misma forma (excepto en la primera dimensión).
        X_train, X_test = X_full[train_mask], X_full[test_mask]
        y_train, y_test = y_full[train_mask], y_full[test_mask]
        patient_train = patient_ids_full[train_mask]
        
        # Verificar que la división no resultó en un conjunto vacío
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
             raise ValueError(f"Data split resulted in empty array. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
             
        logger.info(f"Training split: {X_train.shape[0]} samples, {len(train_patients)} patients")
        logger.info(f"Test split: {X_test.shape[0]} samples, {len(test_patients)} patients")
        
        # 4. ENTRENAR EL ENSEMBLE
        # El ensemble se entrena con X_train (que tiene 51 características aplanadas)
        ensemble = AdvancedSepsisEnsemble(cv_folds=3, use_patient_groups=True)
        training_results = ensemble.train(X_train, y_train, patient_train)
        
        # 5. EVALUAR EN PRUEBA
        # ensemble.evaluate recibe X_test (que tiene 51 características aplanadas)
        # Ya no hay desajuste, porque X_train y X_test vienen de la misma fuente X_full.
        logger.info("Evaluating on test set...")
        test_results = ensemble.evaluate(X_test, y_test)
        
        logger.info("=== TEST SET RESULTS ===")
        for metric, score in test_results.items():
            logger.info(f"{metric}: {score:.4f}")
        
        # 6. GUARDAR MODELOS
        models_dir = Path(config.MODELS_DIR)
        models_dir.mkdir(exist_ok=True)
        
        ensemble_path = models_dir / "advanced_sepsis_model.pkl"
        scaler_path = models_dir / "advanced_scaler.pkl"
        
        # Guardamos el ensemble
        joblib.dump(ensemble, ensemble_path)
        
        # Guardamos el data_loader, que ahora contiene el escalador AJUSTADO
        # y la LISTA DE CARACTERÍSTICAS COMPLETA (51)
        joblib.dump(data_loader, scaler_path)
        
        logger.info(f"Models saved to {models_dir}")
        
        return ensemble, {**training_results, **test_results}
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise
    
def train_pipeline_background(filepath: str):
    try:
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        
        from api import routes
        
        routes.training_status.update({
            "status": "training",
            "progress": 0.1,
            "message": "Loading data..."
        })
        
        ensemble, results = train_pipeline(filepath, nrows=5000)
        
        routes.ensemble_model = ensemble
        
        routes.training_status.update({
            "status": "completed",
            "progress": 1.0,
            "message": "Training completed successfully",
            "metrics": results
        })
        
        logger.info("Background training completed successfully")
        
        try:
            Path(filepath).unlink()
        except Exception as e:
            logger.warning(f"Could not remove temp file {filepath}: {e}")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")
        try:
            from api import routes
            routes.training_status.update({
                "status": "error",
                "progress": 0.0,
                "message": f"Training failed: {str(e)}"
            })
        except ImportError:
            logger.error("Could not update training status due to import error")

class EnsembleModelRegistry:
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = {}
        self._load_registry()
    
    def _load_registry(self):
        if self.registry_path.exists():
            try:
                import json
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                logger.info(f"Loaded model registry with {len(registry_data)} entries")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def register_model(self, model_name: str, model_path: str, metadata: Dict):
        self.models[model_name] = {
            "path": model_path,
            "metadata": metadata,
            "registered_at": pd.Timestamp.now().isoformat()
        }
        self._save_registry()
        logger.info(f"Registered model: {model_name}")
    
    def _save_registry(self):
        try:
            import json
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def get_model_info(self, model_name: str) -> Dict:
        return self.models.get(model_name, {})
    
    def list_models(self) -> List[str]:
        return list(self.models.keys())

class TrainingProgressCallback:
    def __init__(self, total_steps: int, update_interval: int = 10):
        self.total_steps = total_steps
        self.current_step = 0
        self.update_interval = update_interval
        self.start_time = pd.Timestamp.now()
    
    def update(self, step: int, message: str = ""):
        self.current_step = step
        progress = step / self.total_steps
        
        if step % self.update_interval == 0 or step == self.total_steps:
            elapsed = pd.Timestamp.now() - self.start_time
            eta = elapsed * (self.total_steps - step) / step if step > 0 else pd.Timedelta(0)
            
            logger.info(f"Progress: {progress:.1%} ({step}/{self.total_steps}) - {message}")
            logger.info(f"Elapsed: {elapsed}, ETA: {eta}")
    
    def get_progress(self) -> Dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.current_step / self.total_steps,
            "elapsed_time": (pd.Timestamp.now() - self.start_time).total_seconds()
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        ensemble, results = train_pipeline(
            r"C:\Users\axelc\Downloads\STACK\Dataset.csv",
            nrows=1000
        )
        print("Training completed successfully!")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Training failed: {e}")