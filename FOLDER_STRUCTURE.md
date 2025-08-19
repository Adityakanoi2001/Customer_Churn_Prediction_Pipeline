# Project Folder Structure

Below is the recommended folder structure for your Customer Churn Prediction Pipeline project, including all code, data, models, Airflow DAGs, logs, deployment artifacts, and documentation files.

```
Customer_Churn_Prediction_Pipeline/
│
├── data/
│   ├── raw/
│   │   └── Telco-Dataset.csv
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── ingestion.py
│   ├── data_validation.py
│   ├── preparation.py
│   ├── data_transformation.py
│   ├── load_features.py
│   ├── train_model.py
│   ├── model_evaluation.py
│   └── deploy_model.py
│
├── dags/
│   └── customer_churn_pipeline_dag.py
│
├── logs/
│   └── (log files)
│
├── deployment/
│   └── (deployment artifacts)
│
├── doc/
│   ├── pipeline_overview.md
│   ├── ingestion.md
│   ├── data_validation.md
│   ├── preparation.md
│   ├── data_transformation.md
│   ├── train_model_and_load_features.md
│   ├── model_evaluation.md
│   ├── deploy_model.md
│   └── data_versioning.md
│
├── .dvc/
│   └── (DVC config files)
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Tip:**  
> - Place all your Markdown documentation files in the `doc/` folder.
> - Adjust or expand the structure as needed for your project.
> - This structure supports clarity, modularity, and easy navigation for both code and documentation.