# Data Transformation Stage – Detailed Documentation

This document describes the purpose, workflow, and implementation details of the `data_transformation.py` script in the Customer Churn Prediction Pipeline. A Mermaid flowchart is included to visualize the data transformation process.

---

## Purpose

The `data_transformation.py` script applies further transformations to the prepared data, such as scaling, normalization, or dimensionality reduction. These transformations ensure that features are on comparable scales and that the data is optimized for machine learning algorithms.

---

## Workflow Steps

1. **Load Prepared Data:**  
   - Reads the prepared data file (e.g., `data/processed/prepared_data.csv`).

2. **Feature Scaling:**  
   - Applies scaling techniques (e.g., StandardScaler, MinMaxScaler) to numerical features so they have similar ranges or distributions.
   - Ensures that features contribute equally to model training.

3. **Normalization (if required):**  
   - Normalizes data to a specific range or distribution (e.g., unit norm).
   - Useful for algorithms sensitive to the magnitude of input features.

4. **Dimensionality Reduction (optional):**  
   - Applies techniques such as Principal Component Analysis (PCA) to reduce the number of features while retaining most of the variance.
   - Helps in reducing noise and improving computational efficiency.

5. **Save Transformed Data:**  
   - Writes the transformed data to a new file (e.g., `data/processed/transformed_data.csv`) for use in model training.

6. **Logging:**  
   - Logs all transformations applied, including scaling parameters and any dimensionality reduction steps.

## Inputs

- Prepared data file (e.g., `data/processed/prepared_data.csv`)

## Outputs

- Transformed data file (e.g., `data/processed/transformed_data.csv`)
- Transformation logs

---

## Notes

- The script is designed to be flexible, allowing for different scaling and reduction techniques as needed.
- Transformation parameters (e.g., scaler means and variances) should be saved for use during inference.
- Proper scaling and transformation can significantly improve model performance and convergence.

---

## Example Usage

```bash
python src/data_transformation.py
```

---

This transformation stage ensures that the data is in the optimal format for training robust and accurate machine learning models.