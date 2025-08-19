# Model Training Stage – Detailed Documentation

This document describes the purpose, workflow, and implementation details of the `train_model.py` script in the Customer Churn Prediction Pipeline. It also covers the supporting `load_features.py` script, which is responsible for loading and preparing features for training. A Mermaid flowchart is included to visualize the model training process.

---

## Purpose

The `train_model.py` script is responsible for training machine learning models to predict customer churn. It loads features and the target variable from a feature store (using `load_features.py`), splits the data, trains multiple models, evaluates them, logs results with MLflow, and saves the best-performing model for deployment.

---

## Workflow Steps

### 1. Load Features and Target (`load_features.py`)

- Connects to a SQLite database (`data/features/churn_features.db`) and loads the feature table.
- Normalizes the target column (`Churn`) to binary (0/1), handling various formats (numeric, string, etc.).
- Drops index-like columns if present.
- Ensures all numeric features are of type `float64` for compatibility with MLflow and downstream tasks.
- Returns feature matrix `X` and target vector `y`.

### 2. Validate Target Classes

- Checks that the target variable contains at least two classes (required for classification).
- Logs the class distribution for transparency.

### 3. Train/Test Split

- Uses stratified shuffle split to divide the data into training and test sets, preserving class distribution.

### 4. Model Training and Evaluation

- **Logistic Regression:**
  - Trains a logistic regression model with balanced class weights.
  - Evaluates on the test set using recall, accuracy, precision, and F1-score.
  - Logs parameters and metrics to MLflow.
  - Logs a classification report.

- **Random Forest:**
  - Trains a random forest classifier with balanced class weights.
  - Evaluates and logs metrics as above.

- The model with the highest recall is selected as the best model.

### 5. Save and Register Best Model

- Saves the best model as `models/churn_model.pkl` using `joblib`.
- Writes the best MLflow run ID to a temporary file for downstream use.
- Attempts to log and register the model in the MLflow model registry (if available).
- Handles any MLflow registry errors gracefully.

### 6. Logging

- All steps, metrics, and errors are logged to `logs/training.log` and the console for traceability.

---

## Mermaid Flowchart

```mermaid
flowchart TD
    A[Start Training] --> B[Load features & target<br>(load_features.py)]
    B --> C[Validate target classes]
    C --> D[Stratified train/test split]
    D --> E[Train Logistic Regression]
    D --> F[Train Random Forest]
    E --> G[Evaluate & log metrics (MLflow)]
    F --> H[Evaluate & log metrics (MLflow)]
    G --> I{Best recall?}
    H --> I
    I --> J[Save best model<br>as churn_model.pkl]
    J --> K[Register/log model in MLflow]
    K --> L[Write best run ID to temp file]
    L --> M[Finish]
```

---

## Inputs

- Feature database: `data/features/churn_features.db` (table: `churn_features`)
- Target column: `Churn`

## Outputs

- Trained model file: `models/churn_model.pkl`
- MLflow experiment logs and metrics
- Training logs: `logs/training.log`
- Best run ID: temporary file in system temp directory

---

## Key Implementation Details

### `load_features.py`

- **Robust Target Normalization:** Handles numeric and string targets, mapping all common churn labels to 0/1.
- **Data Type Consistency:** Ensures all numeric features are `float64` for MLflow compatibility.
- **Logging:** Provides detailed logs for debugging and data transparency.

### `train_model.py`

- **Model Selection:** Trains both logistic regression and random forest, selecting the model with the highest recall (important for churn prediction).
- **MLflow Integration:** Logs all parameters, metrics, and models to MLflow for experiment tracking and reproducibility.
- **Error Handling:** Gracefully handles errors in model registration and data loading, logging all issues for review.
- **Reproducibility:** Ensures all random states are fixed for consistent results.

---

## Example Usage

```bash
python src/train_model.py
```

---

## Notes

- The pipeline is designed to be robust, with extensive logging and error handling.
- MLflow experiment tracking enables easy comparison of different model runs and hyperparameters.
- The best model is saved locally and optionally registered in the MLflow model registry for deployment.

---

This model training stage is central to the pipeline, transforming engineered features into a predictive model that can be evaluated, versioned, and deployed for customer churn prediction.