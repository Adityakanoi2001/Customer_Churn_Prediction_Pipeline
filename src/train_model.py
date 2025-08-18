import os
import logging
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, classification_report
from sklearn.utils import resample
import joblib
from load_features import get_features_for_training

# ---------------- Logging setup ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Model output path ----------------
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')

# ---------------- MLflow config ----------------
mlflow.set_tracking_uri("file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns")
from mlflow.tracking import MlflowClient

client = MlflowClient()

EXP_NAME = "Churn Prediction (Airflow)"
artifact_location = "file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns"

# Create experiment if it doesn't exist
exp = client.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = client.create_experiment(name=EXP_NAME, artifact_location=artifact_location)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(EXP_NAME)

def validate_data_classes(y):
    """Validate that we have at least 2 classes in the target variable"""
    unique_classes = np.unique(y)
    logger.info(f"Unique classes in target: {unique_classes}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
    
    if len(unique_classes) < 2:
        raise ValueError(f"Insufficient classes for training. Found classes: {unique_classes}. Need at least 2 classes.")
    
    return unique_classes

def handle_class_imbalance(X, y, method='oversample', random_state=42):
    """Handle class imbalance in the dataset"""
    unique_classes, counts = np.unique(y, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique_classes, counts))}")
    
    # If classes are severely imbalanced (ratio > 10:1), apply resampling
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 10:
        logger.info(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying {method}...")
        
        if method == 'oversample':
            # Oversample minority class
            minority_class = unique_classes[np.argmin(counts)]
            majority_class = unique_classes[np.argmax(counts)]
            
            # Separate majority and minority classes
            X_majority = X[y == majority_class]
            X_minority = X[y == minority_class]
            y_majority = y[y == majority_class]
            y_minority = y[y == minority_class]
            
            # Oversample minority class
            X_minority_upsampled, y_minority_upsampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=len(X_majority) // 2,  # Make it 1:2 ratio instead of 1:1
                random_state=random_state
            )
            
            # Combine majority class with upsampled minority class
            X_balanced = np.vstack((X_majority, X_minority_upsampled))
            y_balanced = np.hstack((y_majority, y_minority_upsampled))
            
        elif method == 'undersample':
            # Undersample majority class
            majority_class = unique_classes[np.argmax(counts)]
            minority_class = unique_classes[np.argmin(counts)]
            
            X_majority = X[y == majority_class]
            X_minority = X[y == minority_class]
            y_majority = y[y == majority_class]
            y_minority = y[y == minority_class]
            
            # Undersample majority class
            X_majority_downsampled, y_majority_downsampled = resample(
                X_majority, y_majority,
                replace=False,
                n_samples=len(X_minority) * 2,  # Make it 2:1 ratio
                random_state=random_state
            )
            
            # Combine downsampled majority class with minority class
            X_balanced = np.vstack((X_majority_downsampled, X_minority))
            y_balanced = np.hstack((y_majority_downsampled, y_minority))
        
        # Convert back to original format if needed
        if hasattr(X, 'iloc'):  # If it's a pandas DataFrame
            import pandas as pd
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
        
        logger.info(f"Balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
        return X_balanced, y_balanced
    else:
        logger.info(f"Class imbalance is acceptable (ratio: {imbalance_ratio:.2f}). No resampling applied.")
        return X, y

if __name__ == "__main__":
    try:
        logger.info("Loading features and target from feature store...")
        X, y = get_features_for_training()
        y = y.astype(int)
        logger.info(f"Loaded features shape: {X.shape}, target shape: {y.shape}")

        # Validate that we have at least 2 classes
        unique_classes = validate_data_classes(y)
        
        # Handle class imbalance if necessary
        X_balanced, y_balanced = handle_class_imbalance(X, y, method='oversample')
        
        logger.info("Splitting data into train and test sets (80/20, stratified)...")
        # Use stratified split to ensure both classes are in train and test sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X_balanced, y_balanced))
        
        X_train = X_balanced.iloc[train_idx] if hasattr(X_balanced, 'iloc') else X_balanced[train_idx]
        X_test = X_balanced.iloc[test_idx] if hasattr(X_balanced, 'iloc') else X_balanced[test_idx]
        y_train = y_balanced.iloc[train_idx] if hasattr(y_balanced, 'iloc') else y_balanced[train_idx]
        y_test = y_balanced.iloc[test_idx] if hasattr(y_balanced, 'iloc') else y_balanced[test_idx]
        
        # Final validation
        logger.info(f"Train set class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"Test set class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        best_model = None
        best_recall = 0
        best_model_name = ""
        best_run_id = None

        # -------- Logistic Regression --------
        with mlflow.start_run(run_name="LogisticRegression") as run:
            logger.info("Training Logistic Regression...")
            # Use balanced class weights to handle any remaining imbalance
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            recall = recall_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metrics({
                "recall": recall,
                "accuracy": acc,
                "precision": prec,
                "f1": f1
            })

            logger.info(
                f"Logistic Regression recall: {recall:.4f}, "
                f"accuracy: {acc:.4f}, precision: {prec:.4f}, f1: {f1:.4f}"
            )
            logger.info("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

            if recall > best_recall:
                best_recall = recall
                best_model = lr
                best_model_name = "LogisticRegression"
                best_run_id = run.info.run_id

        # -------- Random Forest --------
        with mlflow.start_run(run_name="RandomForest") as run:
            logger.info("Training Random Forest Classifier...")
            # Use balanced class weights
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            recall = recall_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metrics({
                "recall": recall,
                "accuracy": acc,
                "precision": prec,
                "f1": f1
            })

            logger.info(
                f"Random Forest recall: {recall:.4f}, "
                f"accuracy: {acc:.4f}, precision: {prec:.4f}, f1: {f1:.4f}"
            )
            logger.info("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

            if recall > best_recall:
                best_recall = recall
                best_model = rf
                best_model_name = "RandomForestClassifier"
                best_run_id = run.info.run_id

        # -------- Save best model --------
        if best_model is not None:
            joblib.dump(best_model, MODEL_PATH)
            logger.info(
                f"Best model ({best_model_name}) saved to {MODEL_PATH} "
                f"with recall {best_recall:.4f}"
            )

            input_example = X_test.iloc[:5].copy() if hasattr(X_test, 'iloc') else X_test[:5]
            if hasattr(input_example, 'select_dtypes'):
                # Robustly cast numeric cols to float64 for MLflow schema
                input_example = input_example.astype({
                    col: "float64" for col in input_example.select_dtypes(include=["number"]).columns
                })

            # Log into the winning run
            with mlflow.start_run(run_id=best_run_id):
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",  
                    input_example=input_example,
                    registered_model_name="churn_model"
                )

            # -------- Save best run_id --------
            temp_dir = tempfile.gettempdir()
            runid_file = os.path.join(temp_dir, "best_run_id.txt")
            with open(runid_file, "w") as f:
                f.write(best_run_id)

            logger.info(f"Best model run_id written to {runid_file}")

        else:
            logger.error("No model was trained successfully.")
            
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise