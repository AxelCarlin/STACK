import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from models.base_models import BaseModel

logger = logging.getLogger(__name__)

class SepsisDataset(Dataset):
    """PyTorch Dataset for sepsis data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None  # Usar long para CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class SepsisCNN(nn.Module):
    """CNN model for temporal sepsis data"""
    
    def __init__(self, num_features: int, time_window: int):
        super(SepsisCNN, self).__init__()
        
        try:
            self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            
            # Calculate output size after convolutions
            conv_output_size = self._get_conv_output_size(time_window)
            
            self.fc1 = nn.Linear(128 * conv_output_size, 128)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 2)  # Salida binaria para softmax
            logger.info(f"Initialized SepsisCNN with {num_features} features and {time_window} time window")
        except Exception as e:
            logger.error(f"Error initializing SepsisCNN: {e}")
            raise

    def _get_conv_output_size(self, time_window: int) -> int:
        """Calculate the output size after convolutional and pooling layers"""
        size = time_window
        size = size // 2  # After pool1 (kernel_size=2)
        size = size // 2  # After pool2 (kernel_size=2)
        return max(1, size)  # Asegurar que no sea 0

    def forward(self, x):
        """Forward pass"""
        x = x.transpose(1, 2)  # [batch, num_features, time_window]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class OptimizedCNNModel(BaseModel):
    """CNN model wrapper"""
    
    def __init__(self, num_features: int, time_window: int):
        super().__init__("CNN_Medical")
        try:
            self.model = SepsisCNN(num_features, time_window)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.criterion = nn.CrossEntropyLoss()  # Usar CrossEntropyLoss para salida binaria
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.batch_size = 32
            self.epochs = 10
            logger.info(f"Initialized {self.name} with {num_features} features and {time_window} time window")
        except Exception as e:
            logger.error(f"Error initializing OptimizedCNNModel: {e}")
            raise

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the CNN model"""
        try:
            dataset = SepsisDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model.train()
            
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 2 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
            
            self.is_fitted = True
            logger.info(f"Model {self.name} trained successfully")
        except Exception as e:
            logger.error(f"Error training {self.name}: {e}")
            raise

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not trained")
        if self.model is None:
            raise ValueError(f"Model {self.name} is not initialized")
        
        self.model.eval()
        dataset = SepsisDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probs = []
        
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                batch_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                probs.append(batch_probs)
        
        return np.concatenate(probs)  # Probabilidad de la clase positiva

    def get_activations(self, X):
        """Get activations from first conv layer"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device).transpose(1, 2)
        with torch.no_grad():
            x = self.model.conv1(X_tensor)
            x = self.model.relu1(x)
        return x.cpu().numpy()