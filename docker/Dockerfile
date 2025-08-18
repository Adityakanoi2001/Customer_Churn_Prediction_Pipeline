# Dockerfile for Customer Churn Prediction Pipeline
FROM apache/airflow:2.8.3

USER root

# Install build dependencies and git as root
RUN apt-get update && \
    apt-get install -y gcc git python3-dev libffi-dev libssl-dev build-essential

USER airflow

# Upgrade pip and install Python packages as airflow user
RUN pip install --upgrade pip setuptools wheel
RUN pip install dvc[s3]

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .