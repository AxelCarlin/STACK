import pandas as pd
import numpy as np
from typing import Tuple
import logging

from config.settings import config
from data.loader import MedicalDataLoader

logger = logging.getLogger(__name__)

class SepsisDataPreprocessor:
    """Advanced preprocessing for sepsis data"""
    
    def __init__(self, data_loader: MedicalDataLoader):
        self.data_loader = data_loader
        
    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers based on normal medical ranges"""
        for col, (min_val, max_val) in config.NORMAL_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(min_val, max_val)
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced temporal features"""
        logger.info("Creating temporal features...")
        
        df_sorted = df.sort_values(['Patient_ID', 'Hour']).reset_index(drop=True)
        
        for vital in config.VITAL_SIGNS + config.LAB_VALUES:
            if vital in df_sorted.columns:
                try:
                    # Differences
                    df_sorted[f'{vital}_delta'] = df_sorted.groupby('Patient_ID')[vital].diff()
                    
                    # Rolling statistics
                    rolling_max = df_sorted.groupby('Patient_ID')[vital].expanding().max().reset_index(level=0, drop=True)
                    rolling_min = df_sorted.groupby('Patient_ID')[vital].expanding().min().reset_index(level=0, drop=True)
                    df_sorted[f'{vital}_rolling_max'] = rolling_max
                    df_sorted[f'{vital}_rolling_min'] = rolling_min
                    
                    # Trend calculation
                    def compute_trend(x):
                        if len(x) >= 2:
                            return np.polyfit(range(len(x)), x, 1)[0]
                        return 0
                    
                    trend = df_sorted.groupby('Patient_ID')[vital].rolling(
                        window=3, min_periods=2
                    ).apply(compute_trend).reset_index(level=0, drop=True)
                    df_sorted[f'{vital}_trend'] = trend
                    
                except Exception as e:
                    logger.error(f"Error creating features for {vital}: {e}")
                    continue
        
        # Create deterioration score
        self._create_deterioration_score(df_sorted)
        
        logger.info(f"Temporal features created. New columns: {df_sorted.shape[1] - df.shape[1]}")
        return df_sorted
    
    def _create_deterioration_score(self, df: pd.DataFrame):
        """Create vital signs deterioration score"""
        df['vital_deterioration_score'] = 0
        
        if 'HR' in df.columns:
            df['vital_deterioration_score'] += ((df['HR'] > 90) | (df['HR'] < 60)).astype(int)
        if 'Temp' in df.columns:
            df['vital_deterioration_score'] += ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int)
        if 'Resp' in df.columns:
            df['vital_deterioration_score'] += (df['Resp'] > 20).astype(int)
        if 'WBC' in df.columns:
            df['vital_deterioration_score'] += ((df['WBC'] > 12) | (df['WBC'] < 4)).astype(int)
    
    def advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced imputation strategy"""
        logger.info("Performing advanced imputation...")
        
        # Forward/backward fill for vital signs
        vital_cols = [col for col in config.VITAL_SIGNS if col in df.columns]
        for col in vital_cols:
            df[col] = df.groupby('Patient_ID')[col].fillna(method='ffill')
            df[col] = df.groupby('Patient_ID')[col].fillna(method='bfill')
            df[col] = df[col].fillna(df[col].median())
        
        # Default values for lab results
        lab_defaults = {
            'WBC': 7.0, 'Lactate': 1.5, 'pH': 7.4, 'HCO3': 24.0,
            'Creatinine': 1.0, 'BUN': 15.0, 'Platelets': 250.0
        }
        
        lab_cols = [col for col in config.LAB_VALUES if col in df.columns]
        for col in lab_cols:
            default_val = lab_defaults.get(col, df[col].median())
            df[col] = df[col].fillna(default_val)
        
        # Final cleanup
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in config.EXCLUDE_COLS:
                df[col] = df[col].fillna(df[col].median())
        
        # Check for remaining NaNs
        nan_counts = df[numeric_cols].isna().sum()
        if nan_counts.any():
            logger.warning(f"Remaining NaNs: {nan_counts[nan_counts > 0]}")
            for col in nan_counts[nan_counts > 0].index:
                df[col] = df[col].fillna(0)
        
        logger.info("Advanced imputation completed")
        return df
    
    def prepare_cnn_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Prepare data in CNN format"""
        try:
            df = self.clip_outliers(df)
            df = self.create_temporal_features(df)
            df = self.advanced_imputation(df)
            
            feature_columns = self.data_loader.get_feature_columns()
            grouped = df.groupby('Patient_ID')
            
            X_cnn, y_cnn, patient_ids = [], [], []
            
            for patient_id, group in grouped:
                group = group.sort_values('Hour')
                features = group[feature_columns].values
                label = group['SepsisLabel'].iloc[-1] if 'SepsisLabel' in group.columns else 0
                
                # Create time windows
                if len(features) >= config.TIME_WINDOW:
                    features = features[-config.TIME_WINDOW:]
                else:
                    pad_width = config.TIME_WINDOW - len(features)
                    features = np.pad(features, ((pad_width, 0), (0, 0)), mode='constant')
                
                X_cnn.append(features)
                y_cnn.append(label)
                patient_ids.append(patient_id)
            
            X_cnn = np.array(X_cnn)
            y_cnn = np.array(y_cnn)
            
            # Scale data
            original_shape = X_cnn.shape
            X_reshaped = X_cnn.reshape(-1, X_cnn.shape[-1])
            X_scaled = self.data_loader.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(original_shape)
            
            logger.info(f"CNN data prepared: {X_scaled.shape}")
            return X_scaled, y_cnn, pd.Series(patient_ids)
            
        except Exception as e:
            logger.error(f"Error preparing CNN data: {e}")
            raise
