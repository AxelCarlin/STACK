# sepsis_detection_system/visualization/dashboard.py
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import logging

logger = logging.getLogger(__name__)

class AdvancedSepsisVisualizer:
    def __init__(self, output_dir: str = "advanced_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar tema
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_ensemble_performance_dashboard(self, base_results: dict, ensemble_result: dict, 
                                           oof_predictions: np.ndarray, y_true: np.ndarray) -> str:
        """Crea un dashboard interactivo de performance del ensemble"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 'Feature Importance',
                          'Model Contributions', 'Prediction Distribution', 'Calibration Plot'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # ROC Curves
        for i, (model_name, results) in enumerate(base_results.items()):
            if 'predictions' in results:
                fpr, tpr, _ = roc_curve(y_true, results['predictions'])
                auc = roc_auc_score(y_true, results['predictions'])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC={auc:.3f})', 
                                       mode='lines'), row=1, col=1)
        
        # Ensemble ROC
        ensemble_preds = oof_predictions.mean(axis=1)
        fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_preds)
        auc_ens = roc_auc_score(y_true, ensemble_preds)
        fig.add_trace(go.Scatter(x=fpr_ens, y=tpr_ens, name=f'Ensemble (AUC={auc_ens:.3f})', 
                               mode='lines', line=dict(width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                               line=dict(dash='dash', color='gray'), name='Random'), row=1, col=1)
        
        # Precision-Recall Curves
        for i, (model_name, results) in enumerate(base_results.items()):
            if 'predictions' in results:
                precision, recall, _ = precision_recall_curve(y_true, results['predictions'])
                auprc = average_precision_score(y_true, results['predictions'])
                fig.add_trace(go.Scatter(x=recall, y=precision, name=f'{model_name} (AUPRC={auprc:.3f})', 
                                       mode='lines'), row=1, col=2)
        
        # Ensemble PR
        precision_ens, recall_ens, _ = precision_recall_curve(y_true, ensemble_preds)
        auprc_ens = average_precision_score(y_true, ensemble_preds)
        fig.add_trace(go.Scatter(x=recall_ens, y=precision_ens, name=f'Ensemble (AUPRC={auprc_ens:.3f})', 
                               mode='lines', line=dict(width=3)), row=1, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Ensemble Performance Dashboard")
        
        output_path = self.output_dir / "ensemble_dashboard.html"
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_shap_waterfall(self, shap_values: np.ndarray, feature_names: list, 
                             sample_idx: int = 0) -> str:
        """Crea gráfico waterfall de SHAP para explicabilidad"""
        import plotly.graph_objects as go
        
        if len(shap_values.shape) > 1:
            values = shap_values[sample_idx]
        else:
            values = shap_values
        
        # Ordenar por valor absoluto
        sorted_indices = np.argsort(np.abs(values))[::-1][:15]  # Top 15
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        colors = ['red' if v < 0 else 'green' for v in sorted_values]
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(sorted_values),
            x=sorted_features,
            textposition="outside",
            text=[f"{v:.3f}" for v in sorted_values],
            y=sorted_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}}
        ))
        
        fig.update_layout(
            title=f"SHAP Feature Contributions (Sample {sample_idx})",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=600
        )
        
        output_path = self.output_dir / f"shap_waterfall_sample_{sample_idx}.html"
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_meta_model_analysis(self, meta_features: pd.DataFrame, target: np.ndarray) -> str:
        """Análisis visual del meta-modelo"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Correlations', 'Base Model Agreement', 
                          'Confidence Distribution', 'Error Analysis'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Matriz de correlación
        corr_matrix = meta_features.corr()
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ), row=1, col=1)
        
        # Distribución de confianza
        if 'ensemble_std' in meta_features.columns:
            fig.add_trace(go.Histogram(
                x=meta_features['ensemble_std'],
                nbinsx=50,
                name='Std Distribution'
            ), row=2, col=1)
        
        fig.update_layout(height=800, title_text="Meta-Model Analysis Dashboard")
        
        output_path = self.output_dir / "meta_model_analysis.html"
        fig.write_html(str(output_path))
        return str(output_path)