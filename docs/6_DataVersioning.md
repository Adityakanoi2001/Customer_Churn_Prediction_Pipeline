# Data Versioning Implementation in the Customer Churn Prediction Pipeline

This document details the data versioning practices implemented in the project, explaining each change and its benefit.

---

## 1. DVC Initialization and Configuration

**What:**  
The project is initialized with DVC (`dvc init`), creating a `.dvc` directory and configuration files.

**Why:**  
Sets up the project to track data files and model artifacts separately from code, enabling reproducibility and collaboration.

---

## 2. Tracking Data Files with DVC

**What:**  
Raw datasets (e.g., `data/raw/Telco-Dataset.csv`) and processed data files are tracked using DVC (`dvc add data/raw/Telco-Dataset.csv`).

**How:**  
This creates `.dvc` metafiles (e.g., `Telco-Dataset.csv.dvc`) that record the file’s hash and location.

**Why:**  
Ensures every version of the dataset is tracked, so you can always retrieve or revert to a specific version used in any experiment.

---

## 3. Tracking Model Artifacts with DVC

**What:**  
Trained model files (e.g., `models/churn_model.pkl`) are tracked with DVC (`dvc add models/churn_model.pkl`).

**How:**  
DVC creates a `.dvc` file for the model, and updates `.gitignore` to prevent large files from being stored in Git.

**Why:**  
Allows you to version and share models efficiently, ensuring the exact model used for any prediction or deployment can be retrieved.

---

## 4. Remote Storage Configuration

**What:**  
A DVC remote (cloud storage, shared drive, or local directory) is configured (`dvc remote add -d myremote <remote-url>`).

**How:**  
Data and models are pushed to this remote (`dvc push`), making them accessible to all collaborators.

**Why:**  
Keeps large files out of Git and enables team members to pull the exact data/model versions needed for their work.

---

## 5. Integration with Git

**What:**  
DVC metafiles (`*.dvc`, `.dvc/config`, and `.gitignore`) are committed to Git (`git add`, `git commit`, `git push`).

**How:**  
Links data and model versions to specific code commits, creating a complete snapshot of the project state.

**Why:**  
Ensures that anyone cloning the repository can reproduce results by running `git pull` and `dvc pull`.

---

## 6. Automated Data Retrieval in Pipeline

**What:**  
The Airflow DAG includes a DVC pull step (`dvc pull --allow-missing`) at the start of the pipeline.

**How:**  
Automatically retrieves the correct versions of data and models before any processing or training begins.

**Why:**  
Guarantees that the pipeline always runs with the intended data and model versions, supporting reproducibility and auditability.

---

## 7. Data and Model Versioning in Experiments

**What:**  
Each experiment or model training run is associated with specific data and model versions, as tracked by DVC and referenced in Git commits.

**How:**  
You can reproduce any experiment by checking out the corresponding Git commit and running `dvc pull`.

**Why:**  
Facilitates experiment tracking, comparison, and rollback to previous states if needed.

---

## Summary Table

| Change Implemented                | Description                                                                 | Benefit                        |
|-----------------------------------|-----------------------------------------------------------------------------|--------------------------------|
| DVC Initialization                | Sets up DVC tracking in the project                                         | Enables data versioning        |
| Data & Model Tracking             | Tracks datasets and models with DVC                                         | Ensures reproducibility        |
| Remote Storage                    | Stores large files in remote storage                                        | Collaboration & scalability    |
| Git Integration                   | Commits DVC metafiles to Git                                                | Links code and data versions   |
| Automated Retrieval in Pipeline   | Airflow DAG pulls correct data/model versions before running                | Consistent, reproducible runs  |
| Experiment Versioning             | Associates experiments with specific data/model versions                     | Traceability & rollback        |

---

**In summary:**  
Your pipeline uses DVC to version control all key data and model artifacts, integrates with Git for code-data linkage, and automates retrieval in the workflow. This ensures full reproducibility, traceability, and collaboration for all data science and machine learning activities in the project.