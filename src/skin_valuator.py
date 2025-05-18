import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
from data_collector import SkinDataCollector

class SkinValuator:
    def __init__(self, model_dir: str = "models"):
        """Initialize the skin valuator.
        
        Args:
            model_dir (str): Directory to store trained models
        """
        self.model_dir = model_dir
        self.data_collector = SkinDataCollector()
        self.model = None
        self.label_encoders = {}
        self._ensure_model_directory()
    
    def _ensure_model_directory(self):
        """Create model directory if it doesn't exist."""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for the model.
        
        Args:
            df (pd.DataFrame): Processed skin data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Create a copy to avoid modifying original
        features = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['weapon', 'rarity', 'collection']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col])
            else:
                features[col] = self.label_encoders[col].transform(features[col])
        
        # Select features for the model
        feature_cols = ['weapon', 'rarity', 'collection', 'price_avg']
        target_col = 'stattrack_price_avg'
        
        X = features[feature_cols]
        y = features[target_col]
        
        return X, y
    
    def train_model(self, test_size: float = 0.2):
        """Train the skin valuation model.
        
        Args:
            test_size (float): Proportion of data to use for testing
        """
        # Collect and process data
        data = self.data_collector.collect_historical_data()
        
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save model and encoders
        self.save_model()
    
    def save_model(self):
        """Save the trained model and encoders."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model
        model_path = os.path.join(self.model_dir, "skin_valuator.joblib")
        joblib.dump(self.model, model_path)
        
        # Save encoders
        encoders_path = os.path.join(self.model_dir, "label_encoders.joblib")
        joblib.dump(self.label_encoders, encoders_path)
        
        print(f"Model saved to {model_path}")
        print(f"Encoders saved to {encoders_path}")
    
    def load_model(self):
        """Load the trained model and encoders."""
        model_path = os.path.join(self.model_dir, "skin_valuator.joblib")
        encoders_path = os.path.join(self.model_dir, "label_encoders.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(encoders_path):
            raise FileNotFoundError("Model or encoders not found. Train the model first.")
        
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
    
    def predict_skin_value(self, skin_data: Dict) -> Dict:
        """Predict the value of a skin.
        
        Args:
            skin_data (Dict): Skin data including weapon, rarity, collection, and price
            
        Returns:
            Dict: Prediction results including predicted value and confidence
        """
        if self.model is None:
            self.load_model()
        
        # Prepare features
        features = pd.DataFrame([skin_data])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            features[col] = encoder.transform(features[col])
        
        # Make prediction
        predicted_value = self.model.predict(features)[0]
        
        # Calculate confidence (using feature importances)
        feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        return {
            'predicted_value': predicted_value,
            'current_price': skin_data['price_avg'],
            'predicted_premium': predicted_value - skin_data['price_avg'],
            'predicted_premium_pct': ((predicted_value - skin_data['price_avg']) / skin_data['price_avg']) * 100,
            'feature_importance': feature_importance
        }

# Example usage
if __name__ == "__main__":
    valuator = SkinValuator()
    
    # Train the model
    print("Training skin valuation model...")
    valuator.train_model()
    
    # Example prediction
    test_skin = {
        'weapon': 'AK-47',
        'rarity': 'covert',
        'collection': 'bravo',
        'price_avg': 100.0
    }
    
    prediction = valuator.predict_skin_value(test_skin)
    print("\nExample Prediction:")
    print(f"Predicted StatTrak Value: ${prediction['predicted_value']:.2f}")
    print(f"Current Price: ${prediction['current_price']:.2f}")
    print(f"Predicted Premium: ${prediction['predicted_premium']:.2f}")
    print(f"Predicted Premium %: {prediction['predicted_premium_pct']:.2f}%")
    print("\nFeature Importance:")
    for feature, importance in prediction['feature_importance'].items():
        print(f"{feature}: {importance:.4f}") 