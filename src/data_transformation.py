import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sqlite3

# Paths
TODAY = datetime.now().strftime('%d-%m-%Y')
PROCESSED_DIR = os.path.join('data', 'processed', TODAY)
COMBINED_PATH = os.path.join(PROCESSED_DIR, 'Telco-Dataset-combined.csv')
DB_PATH = os.path.join('data', 'features', 'churn_features.db')

# Load combined data
df = pd.read_csv(COMBINED_PATH)

# --- Feature Engineering ---
# Ensure TotalCharges and tenure are numeric
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
if 'tenure' in df.columns:
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
# Example: Create 'TotalChargesPerTenure' (if not present)
if 'TotalCharges' in df.columns and 'tenure' in df.columns:
    df['TotalChargesPerTenure'] = df['TotalCharges'] / (df['tenure'].replace(0, np.nan))
    df['TotalChargesPerTenure'] = df['TotalChargesPerTenure'].fillna(0)

# Example: Customer tenure group
if 'tenure' in df.columns:
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=['0-12','13-24','25-48','49-60','61+'])

# One-hot encode categorical features (excluding customerID and target)
categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in ['customerID','Churn']]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Remove non-numeric columns (e.g., customerID) if present
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Remove rows with missing values in features
if df.isnull().any().any():
    print('Dropping rows with missing values.')
    df = df.dropna()

# --- Scaling ---
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Ensure all numeric columns (including 'Churn' if present) are float64 for MLflow and downstream compatibility
float64_cols = df.select_dtypes(include=['number']).columns
if len(float64_cols) > 0:
    df[float64_cols] = df[float64_cols].astype('float64')

# --- Store in SQLite ---
conn = sqlite3.connect(DB_PATH)
df.to_sql('churn_features', conn, if_exists='replace', index=False)
conn.close()

print(f"Feature engineered and scaled data saved to SQLite DB at {DB_PATH} (table: churn_features)")
