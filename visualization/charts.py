import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from config.settings import config

logger = logging.getLogger(__name__)

class SepsisVisualizer:
    """Traditional matplotlib/seaborn visualizations with enhanced charts"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or config.CHARTS_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def save_plot(self, filename: str, dpi: int = 300) -> str:
        """Save current plot"""
        filepath = self.output_dir / filename
        plt.savefig(filepath, bbox_inches='tight', dpi=dpi, facecolor='white')
        plt.close()
        logger.info(f"Plot saved: {filepath}")
        return str(filepath)
    
    def plot_model_precision_comparison(self, models_results: Dict, y_true: np.ndarray, 
                                       thresholds: List[float] = [0.3, 0.5, 0.7]) -> str:
        """Plot precision, recall, and F1 score comparison across models and thresholds"""
        try:
            metrics_data = []
            
            for model_name, results in models_results.items():
                if 'predictions' in results:
                    y_pred_probs = results['predictions']
                    
                    for threshold in thresholds:
                        y_pred_binary = (y_pred_probs >= threshold).astype(int)
                        
                        precision = precision_score(y_true, y_pred_binary, zero_division=0)
                        recall = recall_score(y_true, y_pred_binary, zero_division=0)
                        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                        accuracy = accuracy_score(y_true, y_pred_binary)
                        
                        metrics_data.extend([
                            {'Model': model_name, 'Threshold': threshold, 'Metric': 'Precision', 'Score': precision},
                            {'Model': model_name, 'Threshold': threshold, 'Metric': 'Recall', 'Score': recall},
                            {'Model': model_name, 'Threshold': threshold, 'Metric': 'F1-Score', 'Score': f1},
                            {'Model': model_name, 'Threshold': threshold, 'Metric': 'Accuracy', 'Score': accuracy}
                        ])
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create subplots for each metric
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Comparison Across Thresholds', fontsize=16, y=0.98)
            
            metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            for idx, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[idx // 2, idx % 2]
                
                metric_data = df_metrics[df_metrics['Metric'] == metric]
                pivot_data = metric_data.pivot(index='Model', columns='Threshold', values='Score')
                
                pivot_data.plot(kind='bar', ax=ax, color=[color] * len(thresholds), alpha=0.8)
                ax.set_title(f'{metric} by Model and Threshold')
                ax.set_ylabel(f'{metric} Score')
                ax.set_xlabel('Model')
                ax.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', rotation=90, fontsize=8)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            return self.save_plot('model_precision_comparison.png')
            
        except Exception as e:
            logger.error(f"Error creating precision comparison: {e}")
            return ""
    
    def plot_ensemble_contribution_analysis(self, meta_dataset: pd.DataFrame, 
                                          base_model_names: List[str]) -> str:
        """Analyze how much each base model contributes to ensemble decisions"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Ensemble Base Model Contribution Analysis', fontsize=16)
            
            # 1. Base model predictions distribution
            ax1 = axes[0, 0]
            base_pred_cols = [col for col in meta_dataset.columns if col.endswith('_pred')]
            for i, col in enumerate(base_pred_cols[:6]):  # Limit to 6 models for readability
                ax1.hist(meta_dataset[col], bins=30, alpha=0.6, 
                        label=col.replace('_pred', ''), density=True)
            ax1.set_xlabel('Prediction Probability')
            ax1.set_ylabel('Density')
            ax1.set_title('Base Model Prediction Distributions')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Model agreement analysis
            ax2 = axes[0, 1]
            if 'disagreement' in meta_dataset.columns:
                ax2.hist(meta_dataset['disagreement'], bins=30, color='orange', alpha=0.7)
                ax2.set_xlabel('Model Disagreement')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Model Agreement Distribution')
                ax2.axvline(meta_dataset['disagreement'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {meta_dataset["disagreement"].mean():.3f}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. Confidence levels
            ax3 = axes[1, 0]
            if 'high_confidence' in meta_dataset.columns and 'low_confidence' in meta_dataset.columns:
                confidence_data = pd.DataFrame({
                    'High Confidence': meta_dataset['high_confidence'],
                    'Low Confidence': meta_dataset['low_confidence']
                })
                confidence_data.plot(kind='hist', bins=30, alpha=0.7, ax=ax3)
                ax3.set_xlabel('Confidence Level')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Ensemble Confidence Distribution')
                ax3.grid(True, alpha=0.3)
            
            # 4. Feature importance from ensemble statistics
            ax4 = axes[1, 1]
            ensemble_stats = ['ensemble_mean', 'ensemble_std', 'ensemble_range']
            available_stats = [stat for stat in ensemble_stats if stat in meta_dataset.columns]
            
            if available_stats:
                correlations = []
                for stat in available_stats:
                    # Calculate correlation with ensemble mean as proxy for importance
                    if 'ensemble_mean' in meta_dataset.columns:
                        corr = meta_dataset[stat].corr(meta_dataset['ensemble_mean'])
                        correlations.append(abs(corr))
                    else:
                        correlations.append(0)
                
                bars = ax4.bar(available_stats, correlations, color='lightblue', alpha=0.8)
                ax4.set_ylabel('Absolute Correlation')
                ax4.set_title('Ensemble Statistics Importance')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, corr in zip(bars, correlations):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{corr:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            return self.save_plot('ensemble_contribution_analysis.png')
            
        except Exception as e:
            logger.error(f"Error creating ensemble analysis: {e}")
            return ""
    
    def plot_calibration_curves(self, models_results: Dict, y_true: np.ndarray, n_bins: int = 10) -> str:
        """Plot calibration curves to assess prediction reliability"""
        try:
            from sklearn.calibration import calibration_curve
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Calibration plot
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
            
            for i, (model_name, results) in enumerate(models_results.items()):
                if 'predictions' in results:
                    y_prob = results['predictions']
                    
                    # Calculate calibration curve
                    fraction_pos, mean_pred_value = calibration_curve(
                        y_true, y_prob, n_bins=n_bins, normalize=False
                    )
                    
                    ax1.plot(mean_pred_value, fraction_pos, marker='o', 
                            label=model_name, color=colors[i])
                    
                    # Calculate Brier score (lower is better)
                    brier_score = np.mean((y_prob - y_true) ** 2)
                    
                    # Reliability diagram (histogram)
                    ax2.hist(y_prob, bins=20, alpha=0.6, density=True, 
                            label=f'{model_name} (Brier: {brier_score:.3f})', 
                            color=colors[i])
            
            ax1.set_xlabel('Mean Predicted Probability')
            ax1.set_ylabel('Fraction of Positives')
            ax1.set_title('Calibration Plot (Reliability Curve)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('Predicted Probability')
            ax2.set_ylabel('Density')
            ax2.set_title('Prediction Probability Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self.save_plot('calibration_curves.png')
            
        except Exception as e:
            logger.error(f"Error creating calibration curves: {e}")
            return ""
    
    def plot_learning_curves_comparison(self, training_history: Dict) -> str:
        """Plot training curves if available from models"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Model Training Progress Comparison', fontsize=16)
            
            models_with_history = {k: v for k, v in training_history.items() 
                                 if v and isinstance(v, dict)}
            
            if not models_with_history:
                # Create a placeholder plot
                axes[0, 0].text(0.5, 0.5, 'No training history available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                return self.save_plot('learning_curves_placeholder.png')
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_history)))
            
            # Plot training loss
            ax1 = axes[0, 0]
            for i, (model_name, history) in enumerate(models_with_history.items()):
                if 'train_logloss' in history and history['train_logloss']:
                    epochs = range(1, len(history['train_logloss']) + 1)
                    ax1.plot(epochs, history['train_logloss'], 
                            label=f'{model_name}', color=colors[i])
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Log Loss')
            ax1.set_title('Training Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot validation loss
            ax2 = axes[0, 1]
            for i, (model_name, history) in enumerate(models_with_history.items()):
                if 'val_logloss' in history and history['val_logloss']:
                    epochs = range(1, len(history['val_logloss']) + 1)
                    ax2.plot(epochs, history['val_logloss'], 
                            label=f'{model_name}', color=colors[i])
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Validation Log Loss')
            ax2.set_title('Validation Loss Curves')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Early stopping visualization
            ax3 = axes[1, 0]
            early_stop_info = []
            for model_name, history in models_with_history.items():
                if 'best_iteration' in history and history['best_iteration']:
                    early_stop_info.append({
                        'Model': model_name,
                        'Best Iteration': history['best_iteration']
                    })
            
            if early_stop_info:
                df_early_stop = pd.DataFrame(early_stop_info)
                bars = ax3.bar(df_early_stop['Model'], df_early_stop['Best Iteration'], 
                              color='lightgreen', alpha=0.8)
                ax3.set_ylabel('Best Iteration')
                ax3.set_title('Early Stopping Points')
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, iteration in zip(bars, df_early_stop['Best Iteration']):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{iteration}', ha='center', va='bottom')
            
            # Training efficiency comparison
            ax4 = axes[1, 1]
            ax4.text(0.5, 0.5, 'Training Efficiency\n(Implementation Dependent)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            
            plt.tight_layout()
            return self.save_plot('learning_curves_comparison.png')
            
        except Exception as e:
            logger.error(f"Error creating learning curves: {e}")
            return ""
    
    def plot_error_analysis(self, y_true: np.ndarray, ensemble_predictions: np.ndarray,
                           base_predictions: Dict, patient_ids: pd.Series = None) -> str:
        """Analyze prediction errors in detail"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Comprehensive Error Analysis', fontsize=16)
            
            # 1. Error distribution by prediction confidence
            ax1 = axes[0, 0]
            errors = np.abs(ensemble_predictions - y_true)
            
            # Bin by prediction confidence
            low_conf = ensemble_predictions < 0.3
            mid_conf = (ensemble_predictions >= 0.3) & (ensemble_predictions <= 0.7)
            high_conf = ensemble_predictions > 0.7
            
            conf_errors = [errors[low_conf], errors[mid_conf], errors[high_conf]]
            conf_labels = ['Low Conf\n(<0.3)', 'Mid Conf\n(0.3-0.7)', 'High Conf\n(>0.7)']
            
            bp = ax1.boxplot(conf_errors, labels=conf_labels, patch_artist=True)
            colors = ['lightcoral', 'khaki', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_ylabel('Absolute Error')
            ax1.set_title('Error by Prediction Confidence')
            ax1.grid(True, alpha=0.3)
            
            # 2. False positive vs False negative analysis
            ax2 = axes[0, 1]
            threshold = 0.5
            pred_binary = (ensemble_predictions >= threshold).astype(int)
            
            fp_mask = (pred_binary == 1) & (y_true == 0)
            fn_mask = (pred_binary == 0) & (y_true == 1)
            tp_mask = (pred_binary == 1) & (y_true == 1)
            tn_mask = (pred_binary == 0) & (y_true == 0)
            
            error_types = ['True Pos', 'True Neg', 'False Pos', 'False Neg']
            error_counts = [tp_mask.sum(), tn_mask.sum(), fp_mask.sum(), fn_mask.sum()]
            colors = ['green', 'lightgreen', 'orange', 'red']
            
            wedges, texts, autotexts = ax2.pie(error_counts, labels=error_types, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Prediction Outcome Distribution')
            
            # 3. Model agreement on errors
            ax3 = axes[0, 2]
            if base_predictions:
                # Calculate how many models agree on misclassified cases
                base_pred_array = np.array([results['predictions'] for results in base_predictions.values()]).T
                base_binary = (base_pred_array >= threshold).astype(int)
                
                # Focus on misclassified cases
                misclassified = pred_binary != y_true
                if misclassified.sum() > 0:
                    agreement_on_errors = np.sum(base_binary[misclassified], axis=1)
                    ax3.hist(agreement_on_errors, bins=range(len(base_predictions) + 2), 
                            alpha=0.7, color='orange', edgecolor='black')
                    ax3.set_xlabel('Number of Models in Agreement')
                    ax3.set_ylabel('Count of Misclassified Cases')
                    ax3.set_title('Model Agreement on Errors')
                    ax3.grid(True, alpha=0.3)
            
            # 4. Error by actual class
            ax4 = axes[1, 0]
            pos_errors = errors[y_true == 1]
            neg_errors = errors[y_true == 0]
            
            ax4.hist([neg_errors, pos_errors], bins=30, alpha=0.7, 
                    label=['No Sepsis', 'Sepsis'], color=['blue', 'red'])
            ax4.set_xlabel('Absolute Error')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Error Distribution by True Class')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Prediction vs True Value Scatter
            ax5 = axes[1, 1]
            scatter = ax5.scatter(y_true, ensemble_predictions, 
                                 c=errors, cmap='Reds', alpha=0.6)
            ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax5.set_xlabel('True Value')
            ax5.set_ylabel('Predicted Probability')
            ax5.set_title('Predictions vs Truth (Color = Error)')
            plt.colorbar(scatter, ax=ax5, label='Absolute Error')
            ax5.grid(True, alpha=0.3)
            
            # 6. High-error cases analysis
            ax6 = axes[1, 2]
            high_error_threshold = np.percentile(errors, 90)  # Top 10% errors
            high_error_cases = errors >= high_error_threshold
            
            if high_error_cases.sum() > 0:
                high_error_preds = ensemble_predictions[high_error_cases]
                ax6.hist(high_error_preds, bins=20, alpha=0.7, color='red', edgecolor='black')
                ax6.set_xlabel('Predicted Probability')
                ax6.set_ylabel('Count')
                ax6.set_title(f'High-Error Cases Distribution\n(Top 10%, n={high_error_cases.sum()})')
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self.save_plot('comprehensive_error_analysis.png')
            
        except Exception as e:
            logger.error(f"Error creating error analysis: {e}")
            return ""

    # NUEVAS GRÁFICAS INDIVIDUALES AGREGADAS
    def plot_individual_model_scatter(self, model_name: str, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> str:
        """Scatter plot individual para un modelo específico"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Scatter plot con colores por clase real
            colors = ['blue' if label == 0 else 'red' for label in y_true]
            sizes = [20 if label == 0 else 40 for label in y_true]  # Sepsis cases larger
            
            plt.scatter(range(len(y_true)), y_pred, c=colors, alpha=0.6, s=sizes)
            
            # Líneas de umbral
            plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Threshold 0.5')
            plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Threshold 0.3')
            plt.axhline(y=0.7, color='purple', linestyle='--', alpha=0.7, label='Threshold 0.7')
            
            # Áreas de riesgo
            plt.fill_between(range(len(y_true)), 0, 0.3, alpha=0.1, color='green', label='Low Risk')
            plt.fill_between(range(len(y_true)), 0.3, 0.7, alpha=0.1, color='yellow', label='Medium Risk')
            plt.fill_between(range(len(y_true)), 0.7, 1, alpha=0.1, color='red', label='High Risk')
            
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Probability')
            plt.title(f'{model_name} - Individual Predictions Scatter Plot')
            
            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.6, label='No Sepsis'),
                Patch(facecolor='red', alpha=0.6, label='Sepsis'),
                plt.Line2D([0], [0], color='green', linestyle='--', label='Threshold 0.5'),
                plt.Line2D([0], [0], color='orange', linestyle='--', label='Threshold 0.3'),
                plt.Line2D([0], [0], color='purple', linestyle='--', label='Threshold 0.7')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.grid(True, alpha=0.3)
            
            return self.save_plot(f'{model_name.replace(" ", "_")}_individual_scatter.png')
            
        except Exception as e:
            logger.error(f"Error creating individual scatter for {model_name}: {e}")
            return ""

    def plot_model_confidence_distribution(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Distribución de confianza por modelo"""
        try:
            n_models = len(models_results)
            cols = 3
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
            fig.suptitle('Model Confidence Distributions by True Class', fontsize=16)
            
            if rows == 1:
                axes = axes.reshape(1, -1) if n_models > 1 else [axes]
            
            model_names = list(models_results.keys())
            
            for idx, (model_name, results) in enumerate(models_results.items()):
                row, col = idx // cols, idx % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                if 'predictions' in results:
                    predictions = results['predictions']
                    
                    # Separar por clase real
                    pos_preds = predictions[y_true == 1]
                    neg_preds = predictions[y_true == 0]
                    
                    # Histograma con curvas de densidad
                    ax.hist(neg_preds, bins=25, alpha=0.5, color='blue', 
                           label=f'No Sepsis (n={len(neg_preds)})', density=True, edgecolor='darkblue')
                    ax.hist(pos_preds, bins=25, alpha=0.5, color='red', 
                           label=f'Sepsis (n={len(pos_preds)})', density=True, edgecolor='darkred')
                    
                    # Estadísticas en el gráfico
                    pos_mean = pos_preds.mean() if len(pos_preds) > 0 else 0
                    neg_mean = neg_preds.mean() if len(neg_preds) > 0 else 0
                    
                    ax.axvline(pos_mean, color='red', linestyle='-', alpha=0.8, 
                              label=f'Sepsis Mean: {pos_mean:.3f}')
                    ax.axvline(neg_mean, color='blue', linestyle='-', alpha=0.8, 
                              label=f'No Sepsis Mean: {neg_mean:.3f}')
                    
                    ax.set_xlabel('Predicted Probability')
                    ax.set_ylabel('Density')
                    ax.set_title(f'{model_name}')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
            
            # Ocultar subplots vacíos
            for idx in range(n_models, rows * cols):
                if rows > 1:
                    row, col = idx // cols, idx % cols
                    axes[row, col].set_visible(False)
                elif cols > 1 and idx < len(axes):
                    axes[idx].set_visible(False)
            
            plt.tight_layout()
            return self.save_plot('model_confidence_distributions.png')
            
        except Exception as e:
            logger.error(f"Error creating confidence distributions: {e}")
            return ""

    def plot_threshold_sensitivity_analysis(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Análisis de sensibilidad por umbral para todos los modelos"""
        try:
            thresholds = np.arange(0.1, 0.95, 0.05)
            
            plt.figure(figsize=(16, 12))
            
            # Subplot para cada métrica
            metrics = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score']
            colors = plt.cm.tab10(np.linspace(0, 1, len(models_results)))
            
            for metric_idx, metric in enumerate(metrics):
                plt.subplot(2, 2, metric_idx + 1)
                
                for i, (model_name, results) in enumerate(models_results.items()):
                    if 'predictions' in results:
                        y_pred = results['predictions']
                        metric_scores = []
                        
                        for thresh in thresholds:
                            y_binary = (y_pred >= thresh).astype(int)
                            
                            if metric == 'Sensitivity':
                                score = recall_score(y_true, y_binary, zero_division=0)
                            elif metric == 'Specificity':
                                tn = np.sum((y_binary == 0) & (y_true == 0))
                                fp = np.sum((y_binary == 1) & (y_true == 0))
                                score = tn / (tn + fp) if (tn + fp) > 0 else 0
                            elif metric == 'Precision':
                                score = precision_score(y_true, y_binary, zero_division=0)
                            else:  # F1-Score
                                score = f1_score(y_true, y_binary, zero_division=0)
                            
                            metric_scores.append(score)
                        
                        plt.plot(thresholds, metric_scores, marker='o', label=model_name, 
                                color=colors[i], linewidth=2, markersize=4)
                
                plt.xlabel('Threshold')
                plt.ylabel(f'{metric} Score')
                plt.title(f'{metric} vs Threshold')
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlim(0.1, 0.9)
                plt.ylim(0, 1.05)
            
            plt.tight_layout()
            return self.save_plot('threshold_sensitivity_analysis.png')
            
        except Exception as e:
            logger.error(f"Error creating threshold analysis: {e}")
            return ""

    def plot_prediction_uncertainty_analysis(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Análisis de incertidumbre en las predicciones"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Prediction Uncertainty Analysis', fontsize=16)
            
            # Calcular estadísticas de ensemble si hay múltiples modelos
            if len(models_results) > 1:
                all_preds = np.column_stack([results['predictions'] for results in models_results.values() 
                                           if 'predictions' in results])
                pred_mean = np.mean(all_preds, axis=1)
                pred_std = np.std(all_preds, axis=1)
                pred_min = np.min(all_preds, axis=1)
                pred_max = np.max(all_preds, axis=1)
                
                # 1. Incertidumbre vs Precisión
                ax1 = axes[0, 0]
                errors = np.abs(pred_mean - y_true)
                scatter = ax1.scatter(pred_std, errors, c=y_true, cmap='RdYlBu', alpha=0.6)
                ax1.set_xlabel('Prediction Standard Deviation')
                ax1.set_ylabel('Absolute Error')
                ax1.set_title('Uncertainty vs Error')
                plt.colorbar(scatter, ax=ax1, label='True Class')
                ax1.grid(True, alpha=0.3)
                
                # 2. Distribución de incertidumbre
                ax2 = axes[0, 1]
                ax2.hist(pred_std, bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax2.axvline(pred_std.mean(), color='red', linestyle='--', 
                           label=f'Mean: {pred_std.mean():.3f}')
                ax2.set_xlabel('Prediction Standard Deviation')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Uncertainty Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. Rango de predicciones
                ax3 = axes[1, 0]
                pred_range = pred_max - pred_min
                ax3.hist(pred_range, bins=30, alpha=0.7, color='green', edgecolor='black')
                ax3.axvline(pred_range.mean(), color='red', linestyle='--', 
                           label=f'Mean Range: {pred_range.mean():.3f}')
                ax3.set_xlabel('Prediction Range (Max - Min)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Prediction Range Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # 4. Casos de alta incertidumbre
                ax4 = axes[1, 1]
                high_uncertainty = pred_std > np.percentile(pred_std, 75)
                uncertain_preds = pred_mean[high_uncertainty]
                uncertain_true = y_true[high_uncertainty]
                
                colors = ['red' if t == 1 else 'blue' for t in uncertain_true]
                ax4.scatter(range(len(uncertain_preds)), uncertain_preds, c=colors, alpha=0.7)
                ax4.set_xlabel('High Uncertainty Cases')
                ax4.set_ylabel('Mean Predicted Probability')
                ax4.set_title(f'High Uncertainty Cases (Top 25%, n={len(uncertain_preds)})')
                ax4.grid(True, alpha=0.3)
            else:
                # Si solo hay un modelo, mostrar análisis básico
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'Uncertainty analysis requires\nmultiple models', 
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            return self.save_plot('prediction_uncertainty_analysis.png')
            
        except Exception as e:
            logger.error(f"Error creating uncertainty analysis: {e}")
            return ""

    def plot_model_decision_boundaries(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Visualizar fronteras de decisión de los modelos"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Model Decision Boundary Analysis', fontsize=16)
            
            thresholds = [0.3, 0.5, 0.7]
            
            for i, (model_name, results) in enumerate(list(models_results.items())[:6]):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                if 'predictions' in results:
                    predictions = results['predictions']
                    
                    # Crear bins de predicción
                    bins = np.linspace(0, 1, 21)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    # Calcular precisión por bin
                    precisions = []
                    counts = []
                    
                    for j in range(len(bins)-1):
                        mask = (predictions >= bins[j]) & (predictions < bins[j+1])
                        if mask.sum() > 0:
                            precision = y_true[mask].mean()
                            count = mask.sum()
                        else:
                            precision = 0
                            count = 0
                        precisions.append(precision)
                        counts.append(count)
                    
                    # Gráfico de barras con precisión por bin
                    bars = ax.bar(bin_centers, precisions, width=0.04, alpha=0.7, 
                                 color='skyblue', edgecolor='navy')
                    
                    # Colorear barras por count
                    for bar, count in zip(bars, counts):
                        if count > 0:
                            normalized_count = count / max(counts)
                            bar.set_color(plt.cm.viridis(normalized_count))
                    
                    # Líneas de umbral
                    for thresh, color in zip(thresholds, ['orange', 'green', 'purple']):
                        ax.axvline(thresh, color=color, linestyle='--', alpha=0.7,
                                  label=f'Threshold {thresh}')
                    
                    ax.set_xlabel('Predicted Probability')
                    ax.set_ylabel('Actual Positive Rate')
                    ax.set_title(f'{model_name}')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.1)
                    
                    if i == 0:  # Solo mostrar leyenda en el primer gráfico
                        ax.legend()
            
            # Ocultar subplots vacíos
            for i in range(len(models_results), 6):
                row, col = i // 3, i % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            return self.save_plot('model_decision_boundaries.png')
            
        except Exception as e:
            logger.error(f"Error creating decision boundaries: {e}")
            return ""

    # Mantener todos los métodos originales
    def plot_correlation_heatmap(self, df: pd.DataFrame, columns: List[str] = None) -> str:
        """Create correlation heatmap"""
        try:
            if columns is None:
                from config.settings import config
                columns = config.VITAL_SIGNS + config.LAB_VALUES
            
            columns = [col for col in columns if col in df.columns]
            if not columns:
                raise ValueError("No valid columns for heatmap")
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create plot
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f',
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            
            plt.title('Medical Features Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            
            return self.save_plot('correlation_heatmap.png')
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return ""
    
    def plot_roc_curves(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Plot ROC curves for multiple models"""
        try:
            plt.figure(figsize=(10, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            
            for i, (model_name, results) in enumerate(models_results.items()):
                if 'predictions' in results:
                    y_pred = results['predictions']
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    auc = results.get('cv_auc', 0.0)
                    
                    plt.plot(
                        fpr, tpr, 
                        color=colors[i % len(colors)],
                        label=f'{model_name} (AUC = {auc:.3f})',
                        linewidth=2
                    )
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves - Model Comparison', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            return self.save_plot('roc_curves_comparison.png')
            
        except Exception as e:
            logger.error(f"Error creating ROC curves: {e}")
            return ""
    
    def plot_precision_recall_curves(self, models_results: Dict, y_true: np.ndarray) -> str:
        """Plot precision-recall curves"""
        try:
            plt.figure(figsize=(10, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            
            for i, (model_name, results) in enumerate(models_results.items()):
                if 'predictions' in results:
                    y_pred = results['predictions']
                    precision, recall, _ = precision_recall_curve(y_true, y_pred)
                    
                    plt.plot(
                        recall, precision,
                        color=colors[i % len(colors)],
                        label=f'{model_name}',
                        linewidth=2
                    )
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves - Model Comparison', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return self.save_plot('precision_recall_curves.png')
            
        except Exception as e:
            logger.error(f"Error creating PR curves: {e}")
            return ""
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], top_n: int = 20) -> str:
        """Plot feature importance"""
        try:
            if not importance_dict:
                logger.warning("No feature importance data available")
                return ""
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importances = zip(*sorted_features)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(features)), importances)
            
            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Most Important Features', fontsize=14)
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Add value labels on bars
            for i, v in enumerate(importances):
                plt.text(v + max(importances) * 0.01, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            
            return self.save_plot('feature_importance.png')
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return ""
    
    def generate_comprehensive_report(self, models_results: Dict, y_true: np.ndarray, 
                                    meta_dataset: pd.DataFrame = None,
                                    training_history: Dict = None,
                                    patient_ids: pd.Series = None) -> List[str]:
        """Generate all visualizations and return list of created files"""
        created_files = []
        
        try:
            # Original plots
            created_files.append(self.plot_roc_curves(models_results, y_true))
            created_files.append(self.plot_precision_recall_curves(models_results, y_true))
            
            # Enhanced plots
            created_files.append(self.plot_model_precision_comparison(models_results, y_true))
            created_files.append(self.plot_calibration_curves(models_results, y_true))
            
            # Individual model plots
            for model_name, results in models_results.items():
                if 'predictions' in results:
                    created_files.append(self.plot_individual_model_scatter(model_name, y_true, results['predictions']))
            
            created_files.append(self.plot_model_confidence_distribution(models_results, y_true))
            created_files.append(self.plot_threshold_sensitivity_analysis(models_results, y_true))
            created_files.append(self.plot_prediction_uncertainty_analysis(models_results, y_true))
            created_files.append(self.plot_model_decision_boundaries(models_results, y_true))
            
            if meta_dataset is not None:
                base_model_names = [col.replace('_pred', '') for col in meta_dataset.columns if col.endswith('_pred')]
                created_files.append(self.plot_ensemble_contribution_analysis(meta_dataset, base_model_names))
            
            if training_history:
                created_files.append(self.plot_learning_curves_comparison(training_history))
            
            # Error analysis with ensemble predictions
            if 'Ensemble' in models_results:
                ensemble_preds = models_results['Ensemble']['predictions']
                base_preds = {k: v for k, v in models_results.items() if k != 'Ensemble'}
                created_files.append(self.plot_error_analysis(y_true, ensemble_preds, base_preds, patient_ids))
            
            # Filter out empty strings (failed plots)
            created_files = [f for f in created_files if f]
            
            logger.info(f"Generated {len(created_files)} visualization files")
            return created_files
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return created_files