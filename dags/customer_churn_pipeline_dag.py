from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_HOME = "/opt/airflow/Customer_Churn_Prediction_Pipeline"

with DAG(
    dag_id='customer_churn_pipeline',
    start_date=datetime(2025, 8, 15),
    schedule='@daily',
    catchup=False,
    tags=['churn_prediction', 'dm4ml'],
    doc_md="""
    ### Customer Churn Prediction Pipeline
    This DAG orchestrates the end-to-end process of ingesting, validating,
    preparing, transforming, and training a model for customer churn prediction.
    """
) as dag:

    BASE_COMMAND = f"cd {PROJECT_HOME} && "

    dvc_pull_task = BashOperator(
        task_id='dvc_pull_data',
        bash_command=BASE_COMMAND + 'dvc pull --allow-missing || echo "DVC pull completed with missing files - continuing"',
        dag=dag,
    )

    ingestion_task = BashOperator(
        task_id='data_ingestion',
        bash_command=BASE_COMMAND + "python src/ingestion.py"
    )

    validation_task = BashOperator(
        task_id='data_validation',
        bash_command=BASE_COMMAND + "python src/data_validation.py"
    )

    preparation_task = BashOperator(
        task_id='data_preparation',
        bash_command=BASE_COMMAND + "python src/preparation.py"
    )

    transformation_task = BashOperator(
        task_id='data_transformation',
        bash_command=BASE_COMMAND + "python src/data_transformation.py"
    )

    training_task = BashOperator(
        task_id='model_training',
        bash_command=BASE_COMMAND + "python src/train_model.py"
    )

    model_evaluation_task = BashOperator(
        task_id='model_evaluation',
        bash_command=BASE_COMMAND + "python src/model_evaluation.py",
        dag=dag,
    )

    dvc_add_commit_model_task = BashOperator(
        task_id='dvc_add_commit_model',
        bash_command=BASE_COMMAND + "git config --global --add safe.directory /opt/airflow/Customer_Churn_Prediction_Pipeline && dvc add models/churn_model.pkl && git add models/churn_model.pkl.dvc models/.gitignore && git commit -m 'Add trained model to DVC' || echo 'Git commit completed or nothing to commit'",
        dag=dag,
    )

    model_deployment_task = BashOperator(
        task_id='model_deployment',
        bash_command=BASE_COMMAND + "python src/deploy_model.py",
        dag=dag,
    )

    dvc_pull_task >> ingestion_task >> validation_task >> preparation_task >> transformation_task >> training_task >> model_evaluation_task >> dvc_add_commit_model_task >> model_deployment_task
