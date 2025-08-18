
# Model Evaluation Report

## Test Set Performance

### Classification Metrics
- **Accuracy**: 0.7306
- **Precision**: 0.4958
- **Recall**: 0.7807
- **F1 Score**: 0.6064
- **ROC AUC**: 0.8307
- **Average Precision**: 0.6356

### Confusion Matrix Details
- **True Positives**: 292
- **True Negatives**: 736
- **False Positives**: 297
- **False Negatives**: 82
- **Sensitivity (Recall)**: 0.7807
- **Specificity**: 0.7125

## Cross-Validation Results (5-Fold)

- **ACCURACY**: 0.7470 ± 0.0039
- **PRECISION**: 0.5155 ± 0.0049
- **RECALL**: 0.8010 ± 0.0193
- **F1**: 0.6272 ± 0.0069
- **ROC_AUC**: 0.8446 ± 0.0049


## Generated Visualizations

- Confusion Matrix: `evaluation/confusion_matrix.png`
- ROC Curve: `evaluation/roc_curve.png`
- Precision-Recall Curve: `evaluation/precision_recall_curve.png`
- Feature Importance: `evaluation/feature_importance.png`

## Model Interpretation

### Key Insights:
- **Churn Detection Rate**: 78.1% of actual churners are correctly identified
- **False Alarm Rate**: 28.8% of non-churners are incorrectly flagged
- **Model Reliability**: Cross-validation shows consistent performance with std < 0.019

### Business Impact:
- **Precision** indicates what % of churn predictions are correct
- **Recall** indicates what % of actual churners we catch
- **F1 Score** balances both precision and recall
