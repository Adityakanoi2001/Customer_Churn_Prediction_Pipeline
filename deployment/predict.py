
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class ChurnPredictor:
    def __init__(self, model_path):
        """Initialize the predictor with the trained model"""
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
    def predict(self, features):
        """
        Make churn predictions
        
        Args:
            features: pandas DataFrame with the same features used in training
            
        Returns:
            dict: predictions and probabilities
        """
        try:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'churn_probability': probabilities[:, 1].tolist(),  # Probability of churn
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_single(self, feature_dict):
        """
        Make prediction for a single customer
        
        Args:
            feature_dict: dictionary with feature names and values
            
        Returns:
            dict: prediction result
        """
        try:
            # Convert dict to DataFrame
            features_df = pd.DataFrame([feature_dict])
            result = self.predict(features_df)
            
            if 'error' not in result:
                # Return single customer result
                return {
                    'customer_will_churn': bool(result['predictions'][0]),
                    'churn_probability': result['churn_probability'][0],
                    'timestamp': result['timestamp']
                }
            else:
                return result
                
        except Exception as e:
            return {'error': str(e)}

# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor('deployment/churn_model_deployed.pkl')
    
    # Example feature dict (replace with actual customer data)
    sample_customer = {
        'tenure': 12,
        'MonthlyCharges': 65.0,
        'TotalCharges': 780.0,
        'SeniorCitizen': 0,
        'Partner': 1,
        # ... add all other features your model expects
    }
    
    # Make prediction
    result = predictor.predict_single(sample_customer)
    print("Prediction Result:", result)
