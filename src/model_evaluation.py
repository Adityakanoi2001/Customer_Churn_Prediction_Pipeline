import os
import logging
import tempfile
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ---------------- Logging setup ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'evaluation.log')
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
EVALUATION_DIR = 'evaluation'
os.makedirs(EVALUATION_DIR, exist_ok=True)

# MLflow configuration
mlflow.set_tracking_uri("file:///opt/airflow/Customer_Churn_Prediction_Pipeline/mlruns")
client = MlflowClient()

# Import your feature loading function
from load_features import get_features_for_training

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

def load_model_and_data():
    """Load the trained model and test data"""
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Load data
        logger.info("Loading data for evaluation...")
        X, y = get_features_for_training()
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        return model, X, y
        
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        raise

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create the same train/test split as used in training"""
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.utils import resample
    
    # Apply the same preprocessing as in training
    unique_classes = np.unique(y)
    logger.info(f"Classes in evaluation data: {unique_classes}")
    
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 classes for evaluation. Found: {unique_classes}")
    
    # Handle class imbalance the same way as training
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 10:
        logger.info(f"Applying same resampling as training (ratio: {imbalance_ratio:.2f})")
        minority_class = unique_classes[np.argmin(counts)]
        majority_class = unique_classes[np.argmax(counts)]
        
        X_majority = X[y == majority_class]
        X_minority = X[y == minority_class]
        y_majority = y[y == majority_class]
        y_minority = y[y == minority_class]
        
        # Oversample minority class
        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, y_minority,
            replace=True,
            n_samples=len(X_majority) // 2,
            random_state=random_state
        )
        
        X_balanced = np.vstack((X_majority, X_minority_upsampled))
        y_balanced = np.hstack((y_majority, y_minority_upsampled))
        
        if hasattr(X, 'iloc'):
            import pandas as pd
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
            
        X, y = X_balanced, y_balanced
    
    # Same stratified split as training
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
    X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
    y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC metrics
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def create_confusion_matrix_plot(y_true, y_pred):
    """Create confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plot_path = os.path.join(EVALUATION_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {plot_path}")
    return plot_path

def create_roc_curve_plot(y_true, y_pred_proba):
    """Create ROC curve visualization"""
    plt.figure(figsize=(8, 6))
    
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'ROC Curve not available\n(single class or no probabilities)', 
                ha='center', va='center', fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
    
    plot_path = os.path.join(EVALUATION_DIR, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {plot_path}")
    return plot_path

def create_precision_recall_plot(y_true, y_pred_proba):
    """Create Precision-Recall curve visualization"""
    plt.figure(figsize=(8, 6))
    
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR Curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.8, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'PR Curve not available\n(single class or no probabilities)', 
                ha='center', va='center', fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
    
    plot_path = os.path.join(EVALUATION_DIR, 'precision_recall_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curve saved to {plot_path}")
    return plot_path

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    plt.figure(figsize=(10, 8))
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), importances[top_indices])
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.gca().invert_yaxis()
        
    elif hasattr(model, 'coef_'):
        # Linear models
        coef = np.abs(model.coef_[0])
        indices = np.argsort(coef)[::-1]
        
        # Plot top 20 features
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), coef[top_indices])
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Absolute Coefficient Value')
        plt.title('Top Feature Coefficients')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                ha='center', va='center', fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('Feature Importance')
    
    plot_path = os.path.join(EVALUATION_DIR, 'feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {plot_path}")
    return plot_path

def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation analysis"""
    logger.info("Performing cross-validation...")
    
    # Stratified K-Fold for balanced evaluation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate different metrics (removed zero_division parameter)
    cv_scores = {}
    try:
        cv_scores['accuracy'] = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_scores['precision'] = cross_val_score(model, X, y, cv=skf, scoring='precision')
        cv_scores['recall'] = cross_val_score(model, X, y, cv=skf, scoring='recall')
        cv_scores['f1'] = cross_val_score(model, X, y, cv=skf, scoring='f1')
        cv_scores['roc_auc'] = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    except Exception as e:
        logger.warning(f"Some cross-validation metrics failed: {e}")
        # Fallback to basic metrics only
        cv_scores['accuracy'] = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        try:
            cv_scores['roc_auc'] = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        except:
            logger.warning("ROC AUC cross-validation also failed")
    
    # Calculate statistics
    cv_stats = {}
    for metric, scores in cv_scores.items():
        cv_stats[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_stats

def log_to_mlflow(metrics, plots, cv_stats, run_id):
    """Log evaluation results to MLflow"""
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                # Log evaluation metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"eval_{metric_name}", value)
                
                # Log cross-validation results
                for metric_name, stats in cv_stats.items():
                    mlflow.log_metric(f"cv_{metric_name}_mean", stats['mean'])
                    mlflow.log_metric(f"cv_{metric_name}_std", stats['std'])
                
                # Log plots as artifacts
                for plot_name, plot_path in plots.items():
                    mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
                
                logger.info("Evaluation results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    else:
        logger.warning("No run ID available, skipping MLflow logging")

def create_evaluation_report(metrics, cv_stats, plots):
    """Create comprehensive evaluation report"""
    report_content = f"""
# Model Evaluation Report

## Test Set Performance

### Classification Metrics
- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}
- **Precision**: {metrics.get('precision', 'N/A'):.4f}
- **Recall**: {metrics.get('recall', 'N/A'):.4f}
- **F1 Score**: {metrics.get('f1_score', 'N/A'):.4f}
- **ROC AUC**: {metrics.get('roc_auc', 'N/A'):.4f}
- **Average Precision**: {metrics.get('avg_precision', 'N/A'):.4f}

### Confusion Matrix Details
- **True Positives**: {metrics.get('true_positives', 'N/A')}
- **True Negatives**: {metrics.get('true_negatives', 'N/A')}
- **False Positives**: {metrics.get('false_positives', 'N/A')}
- **False Negatives**: {metrics.get('false_negatives', 'N/A')}
- **Sensitivity (Recall)**: {metrics.get('sensitivity', 'N/A'):.4f}
- **Specificity**: {metrics.get('specificity', 'N/A'):.4f}

## Cross-Validation Results (5-Fold)

"""
    
    for metric, stats in cv_stats.items():
        report_content += f"- **{metric.upper()}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n"
    
    report_content += f"""

## Generated Visualizations

- Confusion Matrix: `{plots['confusion_matrix']}`
- ROC Curve: `{plots['roc_curve']}`
- Precision-Recall Curve: `{plots['precision_recall']}`
- Feature Importance: `{plots['feature_importance']}`

## Model Interpretation

### Key Insights:
- **Churn Detection Rate**: {metrics.get('recall', 0):.1%} of actual churners are correctly identified
- **False Alarm Rate**: {metrics.get('false_positives', 0) / (metrics.get('false_positives', 0) + metrics.get('true_negatives', 1)):.1%} of non-churners are incorrectly flagged
- **Model Reliability**: Cross-validation shows consistent performance with std < {max([stats['std'] for stats in cv_stats.values()]):.3f}

### Business Impact:
- **Precision** indicates what % of churn predictions are correct
- **Recall** indicates what % of actual churners we catch
- **F1 Score** balances both precision and recall
"""
    
    report_path = os.path.join(EVALUATION_DIR, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Evaluation report saved to {report_path}")
    return report_path

def evaluate_model():
    """Main evaluation function"""
    logger.info("Starting model evaluation...")
    
    try:
        # Load model and data
        model, X, y = load_model_and_data()
        
        # Create train/test split (same as training)
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)
        
        # Make predictions
        logger.info("Making predictions on test set...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        logger.info("Creating evaluation plots...")
        plots = {}
        plots['confusion_matrix'] = create_confusion_matrix_plot(y_test, y_pred)
        plots['roc_curve'] = create_roc_curve_plot(y_test, y_pred_proba)
        plots['precision_recall'] = create_precision_recall_plot(y_test, y_pred_proba)
        plots['feature_importance'] = create_feature_importance_plot(model, X.columns.tolist())
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_stats = perform_cross_validation(model, X, y)
        
        # Log to MLflow
        run_id = load_best_run_id()
        log_to_mlflow(metrics, plots, cv_stats, run_id)
        
        # Create evaluation report
        report_path = create_evaluation_report(metrics, cv_stats, plots)
        
        # Log final results
        logger.info("="*50)
        logger.info("MODEL EVALUATION COMPLETE!")
        logger.info("="*50)
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Evaluation report: {report_path}")
        logger.info("="*50)
        
        return metrics, plots, cv_stats, report_path
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    evaluate_model()