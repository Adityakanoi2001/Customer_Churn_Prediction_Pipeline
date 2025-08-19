# Model Deployment Stage – Detailed Documentation

This document describes the purpose, workflow, and implementation details of the `deploy_model.py` script in the Customer Churn Prediction Pipeline.

---

## Purpose

The `deploy_model.py` script automates the deployment of the trained churn prediction model. It loads the best model (from MLflow or local storage), prepares it for production use, generates supporting scripts (for batch and API predictions), and documents the deployment for traceability and reproducibility.

---

## Workflow Steps

1. **Logging Setup**
   - Initializes logging to both a file (`logs/deployment.log`) and the console for monitoring deployment activities.

2. **Configuration**
   - Sets up paths for the model, deployment directory, and MLflow tracking.

3. **Load Best Model**
   - Attempts to load the best model using the run ID from MLflow.
   - If unavailable, falls back to the local model file (`models/churn_model.pkl`).
   - Ensures the model is ready for deployment regardless of its source.

4. **Prepare Deployment Directory**
   - Creates a `deployment/` directory to store all deployment artifacts.

5. **Copy/Save Model for Deployment**
   - Copies the model to `deployment/churn_model_deployed.pkl`.
   - If loaded from MLflow, saves it in joblib format for consistency.

6. **Create Model Information File**
   - Generates a `model_info.txt` file with deployment timestamp, model source, type, path, run ID, and version.

7. **Generate Prediction Script**
   - Creates a `predict.py` script for batch and single-customer predictions using the deployed model.
   - The script provides a `ChurnPredictor` class for easy integration.

8. **Generate API Endpoint Script**
   - Creates an `api.py` Flask application for serving predictions via HTTP POST requests.
   - Includes a `/predict` endpoint for predictions and a `/health` endpoint for health checks.
   - Generates a `requirements.txt` for API dependencies.

9. **Create Deployment Summary**
   - Writes a `deployment_summary.md` summarizing the deployment details, model info, and script locations.

10. **Logging and Completion**
    - Logs all key actions and the locations of deployment artifacts.
    - Reports success or failure of the deployment process.

---

## Mermaid Flowchart

```mermaid
flowchart TD
    A[Start Deployment] --> B[Load best model (MLflow/local)]
    B --> C[Prepare deployment directory]
    C --> D[Copy/save model for deployment]
    D --> E[Create model info file]
    E --> F[Generate prediction script]
    F --> G[Generate API endpoint script]
    G --> H[Create deployment summary]
    H --> I[Log completion & finish]
```

---

## Inputs

- Trained model file: `models/churn_model.pkl` or MLflow run/model
- Best run ID: stored in a temp file from training

## Outputs

- Deployed model: `deployment/churn_model_deployed.pkl`
- Model info: `deployment/model_info.txt`
- Prediction script: `deployment/predict.py`
- API endpoint script: `deployment/api.py`
- API requirements: `deployment/requirements.txt`
- Deployment summary: `deployment/deployment_summary.md`
- Deployment logs: `logs/deployment.log`

---

## Key Implementation Details

- **Robust Model Loading:** Tries MLflow first, then local file, ensuring deployment works in various environments.
- **Self-Contained Deployment:** All necessary files (model, scripts, info, API) are placed in the `deployment/` directory.
- **Prediction Script:** Enables batch and single-customer predictions for easy integration into other systems.
- **API Endpoint:** Provides a ready-to-use Flask API for real-time predictions.
- **Documentation:** Every deployment is documented with a summary and model info for traceability.
- **Logging:** All actions and errors are logged for transparency and debugging.

---

## Example Usage

```bash
python src/deploy_model.py
```

---

## Notes

- The deployment process is automated and repeatable, supporting both batch and real-time use cases.
- The generated API and prediction scripts can be further customized as needed.
- All deployment artifacts are versioned and documented for future reference.

---

This deployment stage ensures that the churn prediction model is production-ready, well-documented, and easy to integrate into business applications or services.