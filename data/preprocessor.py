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
        
        # Lista para rastrear columnas creadas
        created_cols = []
        
        for vital in config.VITAL_SIGNS + config.LAB_VALUES:
            if vital in df_sorted.columns:
                try:
                    # Differences
                    delta_col = f'{vital}_delta'
                    df_sorted[delta_col] = df_sorted.groupby('Patient_ID')[vital].diff()
                    created_cols.append(delta_col)
                    
                    # Rolling statistics
                    rolling_max_col = f'{vital}_rolling_max'
                    rolling_min_col = f'{vital}_rolling_min'
                    rolling_max = df_sorted.groupby('Patient_ID')[vital].expanding().max().reset_index(level=0, drop=True)
                    rolling_min = df_sorted.groupby('Patient_ID')[vital].expanding().min().reset_index(level=0, drop=True)
                    df_sorted[rolling_max_col] = rolling_max
                    df_sorted[rolling_min_col] = rolling_min
                    created_cols.extend([rolling_max_col, rolling_min_col])
                    
                    # Trend calculation
                    def compute_trend(x):
                        if len(x) >= 2:
                            return np.polyfit(range(len(x)), x, 1)[0]
                        return 0
                    
                    trend_col = f'{vital}_trend'
                    trend = df_sorted.groupby('Patient_ID')[vital].rolling(
                        window=3, min_periods=2
                    ).apply(compute_trend).reset_index(level=0, drop=True)
                    df_sorted[trend_col] = trend
                    created_cols.append(trend_col)
                    
                except Exception as e:
                    logger.error(f"Error creating features for {vital}: {e}")
                    continue
        
        # Create deterioration score
        self._create_deterioration_score(df_sorted)
        created_cols.append('vital_deterioration_score')
        
        # Almacenar las columnas de características en el data_loader para consistencia
        # Esto es clave: ahora el data_loader conoce las nuevas columnas
        original_features = self.data_loader.get_feature_columns()
        self.data_loader.feature_columns = original_features + created_cols
        
        logger.info(f"Temporal features created. Total features: {len(self.data_loader.feature_columns)}")
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
        
        # Obtener todas las columnas (originales + temporales)
        all_feature_cols = self.data_loader.get_feature_columns()
        
        # Imputar columnas temporales (deltas, etc.) con 0
        for col in all_feature_cols:
             if '_delta' in col or '_trend' in col:
                df[col] = df[col].fillna(0)

        # Forward/backward fill para vitales (originales)
        vital_cols = [col for col in config.VITAL_SIGNS if col in df.columns]
        for col in vital_cols:
            df[col] = df.groupby('Patient_ID')[col].fillna(method='ffill')
            df[col] = df.groupby('Patient_ID')[col].fillna(method='bfill')
            df[col] = df[col].fillna(df[col].median())
        
        # Default values para labs (originales)
        lab_defaults = {
            'WBC': 7.0, 'Lactate': 1.5, 'pH': 7.4, 'HCO3': 24.0,
            'Creatinine': 1.0, 'BUN': 15.0, 'Platelets': 250.0
        }
        lab_cols = [col for col in config.LAB_VALUES if col in df.columns]
        for col in lab_cols:
            default_val = lab_defaults.get(col, df[col].median())
            df[col] = df[col].fillna(default_val)
        
        # Imputar cualquier resto en las columnas de características
        for col in all_feature_cols:
            if col in df.columns:
                # Llenar max/min rodantes que aún sean NaN (inicio del historial del paciente)
                if '_rolling_max' in col or '_rolling_min' in col:
                     # Llenar con el valor de la columna base (ej. 'HR_rolling_max' se llena con 'HR')
                    base_col = col.replace('_rolling_max', '').replace('_rolling_min', '')
                    if base_col in df.columns:
                         df[col] = df[col].fillna(df[base_col])
                
                # Llenar cualquier resto con 0 (seguro para columnas numéricas)
                df[col] = df[col].fillna(0) 

        logger.info("Advanced imputation completed")
        return df
    
    def prepare_cnn_data(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        Prepara datos en formato CNN, manejando el escalador y la selección de características.
        
        :param df: DataFrame con datos brutos.
        :param is_training: Si es True, ajusta (fit) el escalador.
        """
        try:
            df = self.clip_outliers(df)
            
            # 1. INGENIERÍA DE CARACTERÍSTICAS
            # Esta función ahora actualiza self.data_loader.feature_columns
            df = self.create_temporal_features(df)
            
            # 2. IMPUTACIÓN
            # Usa la lista de características actualizada para imputar correctamente
            df = self.advanced_imputation(df)
            
            # 3. SELECCIÓN DE CARACTERÍSTICAS (LA CORRECCIÓN)
            # Obtiene la lista COMPLETA de características (51) desde el data_loader
            feature_columns = self.data_loader.get_feature_columns()
            
            # Asegurarse de que todas las columnas existan en el df (por si acaso)
            for col in feature_columns:
                if col not in df.columns:
                    logger.warning(f"Columna de característica '{col}' no encontrada. Rellenando con 0.")
                    df[col] = 0.0
            
            grouped = df.groupby('Patient_ID')
            
            X_cnn, y_cnn, patient_ids = [], [], []
            
            for patient_id, group in grouped:
                group = group.sort_values('Hour')
                
                # Selecciona el conjunto completo de características (51)
                features = group[feature_columns].values 
                label = group['SepsisLabel'].iloc[-1] if 'SepsisLabel' in group.columns else 0
                
                # Crear ventanas de tiempo
                if len(features) >= config.TIME_WINDOW:
                    features = features[-config.TIME_WINDOW:]
                else:
                    pad_width = config.TIME_WINDOW - len(features)
                    features = np.pad(features, ((pad_width, 0), (0, 0)), mode='constant', constant_values=0)
                
                X_cnn.append(features)
                y_cnn.append(label)
                patient_ids.append(patient_id)
            
            X_cnn = np.array(X_cnn)
            y_cnn = np.array(y_cnn)
            
            if X_cnn.shape[0] == 0:
                logger.warning("No data generated after preprocessing.")
                return X_cnn, y_cnn, pd.Series(patient_ids)

            # 4. ESCALADO (LA CORRECCIÓN)
            original_shape = X_cnn.shape
            # Asegurarse de que X_cnn.shape[-1] sea el número de características (51)
            num_features = X_cnn.shape[-1]
            logger.info(f"Shape antes de escalar: {original_shape}. Número de características: {num_features}")

            X_reshaped = X_cnn.reshape(-1, num_features)
            
            if is_training:
                # Se ajusta solo con datos de entrenamiento (X_train)
                X_scaled = self.data_loader.scaler.fit_transform(X_reshaped) 
                logger.info("Scaler FITTED and transformed.")
            else:
                # Se usa el escalador YA ajustado (para X_test o predicción)
                X_scaled = self.data_loader.scaler.transform(X_reshaped)
                logger.info("Scaler only TRANSFORMED.")
                
            X_scaled = X_scaled.reshape(original_shape)
            
            logger.info(f"CNN data prepared: {X_scaled.shape}")
            return X_scaled, y_cnn, pd.Series(patient_ids)
            
        except Exception as e:
            logger.error(f"Error preparing CNN data: {e}")
            raise