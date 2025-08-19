# Customer Churn Prediction Pipeline Documentation

## 1. Problem Formulation

Customer churn is a significant challenge for telecom companies, as losing customers directly affects revenue and market share. The problem at hand is to build a machine learning pipeline that can predict which customers are likely to leave the company (churn) in the near future. By analyzing historical customer data, the pipeline should identify patterns and risk factors associated with churn, enabling the business to take proactive measures to retain valuable customers.

## 2. Business Problem

In the highly competitive telecom industry, acquiring new customers is often more expensive than retaining existing ones. High churn rates can signal customer dissatisfaction, better offers from competitors, or gaps in service quality. If not addressed, churn can lead to substantial revenue loss and increased marketing costs. The business needs a reliable way to identify customers at risk of leaving so that targeted retention strategies (such as personalized offers or improved service) can be implemented, ultimately reducing churn and improving profitability.

## 3. Key Business Objectives

- **Reduce Churn Rate:** The primary objective is to lower the percentage of customers who discontinue their service, thereby stabilizing the customer base.
- **Increase Customer Lifetime Value (CLV):** By retaining more customers, the company can maximize the revenue generated from each customer over time.
- **Optimize Retention Campaigns:** Focus marketing and retention resources on customers who are most likely to churn, improving the efficiency and ROI of these campaigns.
- **Enhance Customer Experience:** Use insights from churn prediction to address common pain points, improve service offerings, and boost overall customer satisfaction.
- **Data-Driven Decision Making:** Empower business teams with actionable insights derived from data, supporting strategic planning and operational improvements.

## 4. Data Sources and Attributes

- **Data Source:**  
  The primary dataset used is `data/raw/Telco-Dataset.csv`, which contains detailed records of telecom customers, their demographics, account information, service usage, and whether they have churned.

- **Key Attributes:**
    - `customerID`: Unique identifier for each customer.
    - `gender`: Gender of the customer (Male/Female).
    - `SeniorCitizen`: Indicates if the customer is a senior citizen (1, 0).
    - `Partner`: Whether the customer has a partner (Yes/No).
    - `Dependents`: Whether the customer has dependents (Yes/No).
    - `tenure`: Number of months the customer has stayed with the company.
    - `PhoneService`: Whether the customer has phone service (Yes/No).
    - `MultipleLines`: Whether the customer has multiple lines (Yes/No/No phone service).
    - `InternetService`: Type of internet service (DSL, Fiber optic, No).
    - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Availability of additional services (Yes/No/No internet service).
    - `Contract`: Contract type (Month-to-month, One year, Two year).
    - `PaperlessBilling`: Whether the customer uses paperless billing (Yes/No).
    - `PaymentMethod`: Payment method (Electronic check, Mailed check, Bank transfer, Credit card).
    - `MonthlyCharges`: The amount charged to the customer monthly.
    - `TotalCharges`: The total amount charged to the customer.
    - `Churn`: Target variable indicating if the customer has churned (Yes/No).

## 5. Measurable Evaluation Metrics

To assess the effectiveness of the churn prediction model, the following metrics will be used:

- **Accuracy:** Measures the proportion of correct predictions out of all predictions made. Useful for balanced datasets.
- **Precision:** Indicates the proportion of positive identifications (predicted churn) that were actually correct. Important when the cost of false positives is high.
- **Recall (Sensitivity):** Measures the proportion of actual churn cases that were correctly identified. Crucial when missing a churn case is costly.
- **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two, especially useful for imbalanced datasets.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Evaluates the model’s ability to distinguish between churned and non-churned customers across all classification thresholds.
- **Confusion Matrix:** Provides a detailed breakdown of true positives, false positives, true negatives, and false negatives, offering deeper insight into model performance.
