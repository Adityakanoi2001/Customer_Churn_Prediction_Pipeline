import os
import logging
import tempfile
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import shutil

# ---------------- Logging setup ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'deployment.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')
DEPLOYMENT_DIR = 'deployment'
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

# MLflow configuration
mlflow.set_tracking_uri("file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns")
client = MlflowClient()


def load_best_run_id():
    """Load the best run ID from the temp file created during training"""
    temp_dir = tempfile.gettempdir()
    runid_file = os.path.join(temp_dir, "best_run_id.txt")
    
    if os.path.exists(runid_file):
        with open(runid_file, "r") as f:
            run_id = f.read().strip()
        logger.info(f"Loaded best run ID: {run_id}")
        return run_id
    else:
        logger.warning(f"Best run ID file not found at {runid_file}")
        return None


def get_latest_model_version():
    """Get the latest version of the registered model"""
    try:
        model_name = "churn_model"
        latest_version = client.get_latest_versions(model_name, stages=["None"])
        if latest_version:
            version = latest_version[0].version
            logger.info(f"Latest model version: {version}")
            return version
        else:
            logger.warning("No model versions found")
            return None
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return None


def load_model_for_deployment():
    """Load the trained model for deployment"""
    try:
        # Try loading from MLflow first
        run_id = load_best_run_id()
        if run_id:
            logger.info(f"Loading model from MLflow run: {run_id}")
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully from MLflow")
            return model, "mlflow"
        
        # Fallback to local file
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from local file: {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully from local file")
            return model, "local"
        
        raise FileNotFoundError("No model found in MLflow or local storage")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def create_model_info_file(model, source, deployment_path):
    """Create a model information file for deployment"""
    info = {
        'deployment_timestamp': datetime.now().isoformat(),
        'model_source': source,
        'model_type': type(model).__name__,
        'deployment_path': deployment_path,
        'run_id': load_best_run_id(),
        'model_version': get_latest_model_version()
    }
    
    info_file = os.path.join(DEPLOYMENT_DIR, 'model_info.txt')
    with open(info_file, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Model info saved to {info_file}")
    return info


def create_prediction_script():
    """Create a simple prediction script for the deployed model"""
    script_content = '''
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
'''

    script_path = os.path.join(DEPLOYMENT_DIR, 'predict.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Prediction script created at {script_path}")
    return script_path


def create_api_endpoint():
    """Create a simple Flask API endpoint for the model"""
    api_content = '''
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the deployed model
MODEL_PATH = 'churn_model_deployed.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a customer
    
    Expected JSON payload:
    {
        "features": {
            "tenure": 12,
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        features_dict = data['features']
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
        response = {
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability[1]),
            'non_churn_probability': float(probability[0])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': MODEL_PATH})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

    api_path = os.path.join(DEPLOYMENT_DIR, 'api.py')
    with open(api_path, 'w') as f:
        f.write(api_content)
    
    logger.info(f"API endpoint script created at {api_path}")
    
    # Create requirements.txt for the API
    requirements_content = '''flask
joblib
pandas
numpy
scikit-learn
'''
    
    req_path = os.path.join(DEPLOYMENT_DIR, 'requirements.txt')
    with open(req_path, 'w') as f:
        f.write(requirements_content)
    
    logger.info(f"Requirements file created at {req_path}")


def create_deployment_summary(model_info, model_path, prediction_script_path):
    """Create a markdown summary of the deployment"""
    summary_content = f"""
# Deployment Summary

- Deployment Timestamp: {model_info['deployment_timestamp']}
- Model Source: {model_info['model_source']}
- Model Type: {model_info['model_type']}
- Deployment Path: {model_path}
- Run ID: {model_info.get('run_id', 'not available')}
- Model Version: {model_info.get('model_version', 'not available')}

## Prediction Script

Located at: `{prediction_script_path}`
"""
    summary_path = os.path.join(DEPLOYMENT_DIR, 'deployment_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    logger.info(f"Deployment summary created at: {summary_path}")


def deploy_model():
    """Main deployment function"""
    logger.info("Starting model deployment...")
    
    try:
        # Load the trained model
        model, source = load_model_for_deployment()
        
        # Create deployment directory structure
        deployment_model_path = os.path.join(DEPLOYMENT_DIR, 'churn_model_deployed.pkl')
        
        # Copy/save model to deployment directory
        if source == "local":
            shutil.copy2(MODEL_PATH, deployment_model_path)
        else:
            # Save MLflow model as joblib for consistency
            joblib.dump(model, deployment_model_path)
        
        logger.info(f"Model deployed to: {deployment_model_path}")
        
        # Create model information file
        model_info = create_model_info_file(model, source, deployment_model_path)
        
        # Create prediction script
        prediction_script = create_prediction_script()
        
        # Create a simple API endpoint script (optional)
        create_api_endpoint()
        
        # Create deployment summary file
        create_deployment_summary(model_info, deployment_model_path, prediction_script)
        
        logger.info("="*50)
        logger.info("MODEL DEPLOYMENT SUCCESSFUL!")
        logger.info("="*50)
        logger.info(f"Deployed Model Path: {deployment_model_path}")
        logger.info(f"Model Source: {source}")
        logger.info(f"Deployment Timestamp: {model_info['deployment_timestamp']}")
        logger.info(f"Prediction Script: {prediction_script}")
        logger.info("="*50)
        
        return deployment_model_path
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    deploy_model()