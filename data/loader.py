import pandas as pd
import numpy as np
from typing import Optional
import logging
from sklearn.preprocessing import RobustScaler

from config.settings import config

logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """Handles data loading, imputing missing values, and preprocessing"""
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.data = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        
    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            if not self.filepath:
                raise ValueError("No filepath specified")
                
            logger.info(f"Loading data from {self.filepath}...")
            self.data = pd.read_csv(self.filepath, nrows=nrows)
            
            # Clean up columns
            if 'Unnamed: 0' in self.data.columns:
                self.data = self.data.drop(columns=['Unnamed: 0'])
            
            # Optimize dtypes
            self.data = self._optimize_dtypes(self.data)
            
            # Impute missing values
            self.data = self._impute_missing_values(self.data)
            
            logger.info(f"Data loaded: {self.data.shape}")
            logger.info(f"Unique patients: {self.data['Patient_ID'].nunique()}")
            logger.info(f"Sepsis cases: {self.data['SepsisLabel'].sum()}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        for col in ['Gender', 'SepsisLabel']:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        numeric_cols = df.select_dtypes(include=[np.float64]).columns
        for col in numeric_cols:
            df[col] = df[col].astype('float32')
        
        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values by patient and globally if needed"""
        if 'Patient_ID' in df.columns:
            # Fill forward and backward per patient
            df = df.groupby("Patient_ID").apply(
                lambda group: group.ffill().bfill()
            ).reset_index(drop=True)
        
        # For any remaining NaNs (entire column empty), fill with median
        for col in df.columns:
            if df[col].isna().all():
                logger.warning(f"Column {col} is completely NaN, filling with 0")
                df[col] = 0
            elif df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def get_feature_columns(self) -> list:
        """Get list of feature columns"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        return [col for col in self.data.columns if col not in config.EXCLUDE_COLS]

    def scale_features(self) -> pd.DataFrame:
        """Apply robust scaling to feature columns"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        self.feature_columns = self.get_feature_columns()
        self.data[self.feature_columns] = self.scaler.fit_transform(
            self.data[self.feature_columns]
        )
        return self.data
