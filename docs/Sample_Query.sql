-- Select the first 10 customers with their churn status and monthly charges
SELECT 
    customerID,
    gender,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM 
    churn_features
LIMIT 10;