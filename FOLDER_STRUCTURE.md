# Project Folder Structure

This project follows a modular structure for an end-to-end data management pipeline for customer churn prediction:

```
Customer_Churn_Prediction_Pipeline/
│
├── data/
│   ├── raw/         # Raw ingested data (by source, type, timestamp)
│   │   ├── train/   # Training data (raw)
│   │   └── test/    # Test data (raw)
│   ├── processed/   # Cleaned and preprocessed data
│   ├── features/    # Feature-engineered datasets
│   └── external/    # Data from third-party APIs or external sources
│
├── logs/            # Ingestion, validation, and pipeline logs
├── models/          # Trained and versioned ML models
├── src/             # Source code (ingestion, validation, transformation, orchestration)
├── docker/          # Docker-related files for containerization
├── dvc/             # DVC metadata and pipelines for data versioning
├── dags/            # DAG definitions for pipeline orchestration (e.g., Airflow)
├── reports/         # EDA images and reports (by date)
├── README.md        # Project overview and instructions
└── customer_churn_pipeline_overview.ipynb  # Main pipeline notebook
```

- **data/raw/**: Partitioned by source, type, and timestamp for efficient storage and retrieval.
- **logs/**: Stores logs for monitoring and debugging.
- **models/**: Contains serialized and versioned models.
- **src/**: All scripts and modules for pipeline automation.

This structure supports scalability, reproducibility, and best practices in data engineering and MLOps.
