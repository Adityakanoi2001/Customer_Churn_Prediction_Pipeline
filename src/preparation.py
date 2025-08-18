import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths
TODAY = datetime.now().strftime('%d-%m-%Y')
TRAIN_PATH = os.path.join('data', 'raw', 'training_data', TODAY, 'Telco-Dataset-train.csv')
TEST_PATH = os.path.join('data', 'raw', 'API_test_data', TODAY, 'Telco-Dataset-test.csv')
PROCESSED_DIR = os.path.join('data', 'processed', TODAY)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

# Encode binary categorical columns
def encode_binary_columns(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    for col in binary_cols:
        df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
    return df

# EDA: Save histograms and boxplots for numeric columns
def perform_eda(df, prefix):
    import shutil
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    reports_dir = os.path.join('reports', TODAY)
    os.makedirs(reports_dir, exist_ok=True)
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'{prefix} Histogram - {col}')
        hist_path = os.path.join(reports_dir, f'{prefix}_hist_{col}.png')
        plt.savefig(hist_path)
        plt.close()
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'{prefix} Boxplot - {col}')
        box_path = os.path.join(reports_dir, f'{prefix}_box_{col}.png')
        plt.savefig(box_path)
        plt.close()
    # Save summary statistics
    stats = df.describe(include='all')
    stats.to_csv(os.path.join(reports_dir, f'{prefix}_summary.csv'))

# Main data preparation pipeline
def main():
    print('Loading data...')
    train_df, test_df = load_data()
    print('Encoding binary categorical columns...')
    train_df = encode_binary_columns(train_df)
    test_df = encode_binary_columns(test_df)
    print('Saving processed data...')
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'Telco-Dataset-train-processed.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'Telco-Dataset-test-processed.csv'), index=False)
    # Combine train and test data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_path = os.path.join(PROCESSED_DIR, 'Telco-Dataset-combined.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f'Combined data saved to {combined_path}')
    print('Performing EDA...')
    perform_eda(train_df, 'train')
    perform_eda(test_df, 'test')
    print(f'Processed data and EDA outputs saved in {PROCESSED_DIR}')

if __name__ == '__main__':
    main()
