# Customer Churn Prediction Pipeline

## Project Overview
This project implements an end-to-end machine learning pipeline for predicting customer churn using the Telco dataset. It leverages Airflow for orchestration, DVC for data versioning, and MLflow for experiment tracking and model management.

## Features
- Automated data ingestion, validation, preparation, and transformation
- Model training, evaluation, and deployment
- Data and model versioning with DVC
- Experiment tracking with MLflow
- Modular, reproducible, and scalable pipeline

## Folder Structure
See the [doc/Pipeline_overview.md](doc/Pipeline_overview.md) and the folder structure section for details on project organization.

## Getting Started
- **Prerequisites:** Python, Docker, DVC, MLflow, and other dependencies listed in `requirements.txt`
- **Setup:**  
  1. Clone the repository  
  2. Install dependencies  
  3. Configure DVC remote storage  
  4. Set up Airflow and MLflow tracking servers as needed
- **Run the pipeline:**  
  - Using Airflow or  
  - With Docker Compose:  
    ```bash
    docker-compose up
    ```

## Usage
- Add new data to `data/raw/`
- Retrain the model by running the pipeline
- View logs in the `logs/` directory
- Access metrics and experiment results via MLflow UI

## Data
- The project uses the Telco customer churn dataset, located at `data/raw/Telco-Dataset.csv`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
[Specify your license here]

## Acknowledgements
- Telco dataset source
- Open-source libraries: Airflow, DVC, MLflow, scikit-learn, etc.

```# Customer Churn Prediction Pipeline

## Project Overview
This project implements an end-to-end machine learning pipeline for predicting customer churn using the Telco dataset. It leverages Airflow for orchestration, DVC for data versioning, and MLflow for experiment tracking and model management.

## Features
- Automated data ingestion, validation, preparation, and transformation
- Model training, evaluation, and deployment
- Data and model versioning with DVC
- Experiment tracking with MLflow
- Modular, reproducible, and scalable pipeline

## Folder Structure
See the [doc/pipeline_overview.md](doc/pipeline_overview.md) and the folder structure section for details on project organization.

## Getting Started
- **Prerequisites:** Python, Docker, DVC, MLflow, and other dependencies listed in `requirements.txt`
- **Setup:**  
  1. Clone the repository  
  2. Install dependencies  
  3. Configure DVC remote storage  
  4. Set up Airflow and MLflow tracking servers as needed
- **Run the pipeline:**  
  - Using Airflow or  
  - With Docker Compose:  
    ```bash
    docker-compose up
    ```

## Usage
- Add new data to `data/raw/`
- Retrain the model by running the pipeline
- View logs in the `logs/` directory
- Access metrics and experiment results via MLflow UI

## Data
- The project uses the Telco customer churn dataset, located at `data/raw/Telco-Dataset.csv`.

## Acknowledgements
- Telco dataset source
- Open-source libraries: Airflow, DVC, MLflow, scikit-learn, etc.
