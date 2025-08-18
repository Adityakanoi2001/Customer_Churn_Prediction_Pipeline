import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
raw_data_path = os.path.join('data', 'raw', 'Telco-Dataset.csv')
train_path = os.path.join('data', 'raw', 'train', 'Telco-Dataset-train.csv')
test_path = os.path.join('data', 'raw', 'test', 'Telco-Dataset-test.csv')

# Read the raw dataset
print(f"Reading data from {raw_data_path}")
df = pd.read_csv(raw_data_path)

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['Churn'])

# Save the splits
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train set saved to {train_path} ({len(train_df)} rows)")
print(f"Test set saved to {test_path} ({len(test_df)} rows)")
