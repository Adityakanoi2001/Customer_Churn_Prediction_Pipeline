
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
