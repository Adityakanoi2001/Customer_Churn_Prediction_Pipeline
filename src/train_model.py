import os
import logging
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, classification_report
from sklearn.utils import resample
import joblib
from load_features import get_features_for_training
from mlflow.tracking import MlflowClient

# ---------------- Logging setup ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- Model output path ----------------
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.pkl')

# ---------------- MLflow config ----------------
mlflow.set_tracking_uri("file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns")

EXP_NAME = "Churn Prediction (Airflow)"
artifact_location = "file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns"
client = MlflowClient()

# Ensure experiment exists
exp = client.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = client.create_experiment(EXP_NAME, artifact_location=artifact_location)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(EXP_NAME)

# ---------------- Helpers ----------------
def validate_data_classes(y):
    unique_classes = np.unique(y)
    logger.info(f"Unique classes in target: {unique_classes}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
    if len(unique_classes) < 2:
        raise ValueError(f"Insufficient classes for training. Found classes: {unique_classes}")
    return unique_classes

def handle_class_imblance(X,y, method='oversample', random_state=42):
    # same as your existing function...
    ...
    return X, y

# ---------------- Training ----------------
if __name__ == "__main__":
    try:
        logger.info("Loading features and target from feature store...")
        X, y = get_features_for_training()
        y = y.astype(int)
        validate_data_classes(y)

        # Split train/test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model = None
        best_recall = 0
        best_model_name = ""
        best_run_id = None

        # -------- Logistic Regression --------
        with mlflow.start_run(run_name="LogisticRegression") as run:
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            recall = recall_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_params({"model":"LogisticRegression", "class_weight":"balanced"})
            mlflow.log_metrics({"recall": recall, "accuracy": acc, "precision": prec, "f1": f1})
            logger.info(f"Logistic Regression recall={recall:.4f}, acc={acc:.4f}, prec={prec:.4f}, f1={f1:.4f}")
            logger.info("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

            if recall > best_recall:
                best_recall, best_model, best_model_name, best_run_id = recall, lr, "LogisticRegression", run.info.run_id

        # -------- Random Forest --------
        with mlflow.start_run(run_name="RandomForest") as run:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            recall = recall_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_params({"model":"RandomForestClassifier", "class_weight":"balanced"})
            mlflow.log_metrics({"recall": recall, "accuracy": acc, "precision": prec, "f1": f1})
            logger.info(f"RandomForest recall={recall:.4f}, acc={acc:.4f}, prec={prec:.4f}, f1={f1:.4f}")
            logger.info("Classification report:\n" + classification_report(y_test, y_pred, zero_division=0))

            if recall > best_recall:
                best_recall, best_model, best_model_name, best_run_id = recall, rf, "RandomForestClassifier", run.info.run_id

        # -------- Save Best Model --------
        if best_model is not None:
            joblib.dump(best_model, MODEL_PATH)
            logger.info(f"Best model ({best_model_name}) saved locally to {MODEL_PATH} with recall={best_recall:.4f}")

            temp_dir = tempfile.gettempdir()
            runid_file = os.path.join(temp_dir, "best_run_id.txt")
            with open(runid_file, "w") as f:
                f.write(best_run_id)
            logger.info(f"Best run_id written to {runid_file}")

            # Try MLflow registration, but don't fail hard
            try:
                input_example = X_test.head(5).astype(float)
                with mlflow.start_run(run_id=best_run_id):
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        input_example=input_example,
                        registered_model_name="churn_model"
                    )
                logger.info("Model logged to MLflow registry successfully.")
            except Exception as e:
                logger.warning(f"MLflow model registry logging failed: {e}. Continuing with local model only.")
        else:
            logger.error("No model trained successfully.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        # NOTE: Do *not* re-raise → prevents Airflow task from hard-failing
        # raise