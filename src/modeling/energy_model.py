"""
GPU energy modeling

This module implements energy models for GPU power prediction. The implementation
is based on established methodologies from the following literature:

1. Hong, S., & Kim, H. (2010). An integrated GPU power and performance model.
   ACM SIGARCH Computer Architecture News, 38(3), 280-289.

2. Kasichayanula, K., Terpstra, D., Luszczek, P., Tomov, S., Moore, S., & 
   Peterson, G. D. (2012, May). Power aware computing on GPUs. 
   In 2012 Symposium on Application Accelerators in High Performance Computing.

3. Mei, X., Chu, X., Liu, H., Leung, Y. W., & Li, Z. (2017). 
   Energy efficient real-time task scheduling on CPU-GPU hybrid clusters.
   In IEEE INFOCOM 2017-IEEE Conference on Computer Communications.

The LinearEnergyModel class implements a linear regression approach that has been
shown to effectively model the relationship between performance counters and power
consumption in modern GPUs.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class BaseEnergyModel:
    """Base class for energy models"""
    
    def __init__(self, model_name: str):
        """
        Initialize model
        
        Args:
            model_name: Name of this model
        """
        self.model_name = model_name
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target values (power/energy)
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mean_abs_error': np.mean(np.abs(y - y_pred)),
            'max_abs_error': np.max(np.abs(y - y_pred))
        }
        
        return metrics


class LinearEnergyModel(BaseEnergyModel):
    """Linear regression model for energy prediction"""
    
    def __init__(self, model_name: str = "linear_model", alpha: float = 0.0):
        """
        Initialize linear energy model
        
        Args:
            model_name: Name of this model
            alpha: Regularization strength (0 for standard linear regression)
        """
        super().__init__(model_name)
        self.alpha = alpha
        self.scaler = StandardScaler()
        
        # Choose model type based on alpha
        if alpha > 0:
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LinearRegression()
            
        self.feature_importance = {}
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the linear energy model
        
        Args:
            X: Feature matrix
            y: Target values (power/energy)
            
        Returns:
            Dictionary with training metrics
        """
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        val_metrics = {
            'mse': mean_squared_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'r2': r2_score(y_val, y_val_pred)
        }
        
        # Set trained flag
        self.is_trained = True
        
        # Store feature importance (coefficients)
        # For linear regression, the coefficients indicate feature importance
        if hasattr(self.model, 'coef_'):
            self.feature_importance = {
                f"feature_{i}": coef for i, coef in enumerate(self.model.coef_)
            }
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': self.feature_importance
        }
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_component_contribution(self, X: np.ndarray, 
                                  feature_groups: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
        """
        Calculate contribution of different component groups to total power
        
        Args:
            X: Feature matrix
            feature_groups: Dictionary mapping component names to feature indices
            
        Returns:
            Dictionary with component power contributions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before analysis")
            
        X_scaled = self.scaler.transform(X)
        contributions = {}
        
        for group_name, feature_indices in feature_groups.items():
            # Create a version of X with only this group's features
            X_group = np.zeros_like(X_scaled)
            X_group[:, feature_indices] = X_scaled[:, feature_indices]
            
            # For linear models, we can directly compute the contribution
            if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                # Divide intercept equally among all groups for simplicity
                group_count = len(feature_groups)
                intercept_contribution = self.model.intercept_ / group_count
                
                # Calculate the contribution from this group
                contributions[group_name] = np.dot(X_group, self.model.coef_) + intercept_contribution
            else:
                # Fallback method for other model types
                contributions[group_name] = self.model.predict(X_group)
                
        return contributions