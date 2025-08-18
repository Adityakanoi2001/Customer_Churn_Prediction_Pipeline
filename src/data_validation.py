import os
import pandas as pd
import numpy as np
from datetime import datetime

TODAY = datetime.now().strftime('%d-%m-%Y')
TRAIN_PATH = os.path.join('data', 'raw', 'training_data', TODAY, 'Telco-Dataset-train.csv')
TEST_PATH = os.path.join('data', 'raw', 'API_test_data', TODAY, 'Telco-Dataset-test.csv')
REPORTS_DIR = os.path.join('reports', TODAY)
os.makedirs(REPORTS_DIR, exist_ok=True)

VALIDATION_REPORT = os.path.join(REPORTS_DIR, 'data_validation_report.txt')

def log(msg):
    print(msg)
    with open(VALIDATION_REPORT, 'a') as f:
        f.write(msg + '\n')

def log_csv(rows, header):
    csv_path = os.path.join(REPORTS_DIR, 'data_validation_report.csv')
    import csv
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def schema_validation(train_df, test_df):
    log('--- Schema Validation ---')
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    if train_cols == test_cols:
        log('PASS: Train and test columns match.')
    else:
        log(f'FAIL: Columns mismatch.\nTrain: {train_cols}\nTest: {test_cols}')
    for col in train_cols:
        if col in test_df.columns:
            if train_df[col].dtype != test_df[col].dtype:
                log(f'FAIL: Data type mismatch in column {col}: train({train_df[col].dtype}) vs test({test_df[col].dtype})')


def missing_values_check(df, name):
    log(f'--- Missing Values in {name} ---')
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            log(f'{col}: {count} missing')
    empty_str = (df == '').sum()
    for col, count in empty_str.items():
        if count > 0:
            log(f'{col}: {count} empty strings')


def value_ranges_and_uniqueness(df, name):
    log(f'--- Value Ranges & Uniqueness in {name} ---')
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            log(f'{col}: min={df[col].min()}, max={df[col].max()}')
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            unique_vals = df[col].unique()
            log(f'{col}: unique values={unique_vals}')
    # Check for duplicate rows
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        log(f'WARNING: {dup_rows} duplicate rows in {name}')
    # Check for duplicate IDs if CustomerID exists
    if 'customerID' in df.columns:
        dup_ids = df['customerID'].duplicated().sum()
        if dup_ids > 0:
            log(f'WARNING: {dup_ids} duplicate customerID values in {name}')


def category_overlap(train_df, test_df):
    log('--- Category Overlap (Test in Train) ---')
    cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or isinstance(train_df[col].dtype, pd.CategoricalDtype)]
    for col in cat_cols:
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        unseen = test_cats - train_cats
        if unseen:
            log(f'WARNING: {col} has unseen categories in test: {unseen}')


def distribution_comparison(train_df, test_df):
    log('--- Distribution Comparison (Train vs Test) ---')
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        train_stats = train_df[col].describe()
        test_stats = test_df[col].describe()
        log(f'{col}: train_mean={train_stats["mean"]:.2f}, test_mean={test_stats["mean"]:.2f}, train_std={train_stats["std"]:.2f}, test_std={test_stats["std"]:.2f}')


def outlier_detection(df, name):
    log(f'--- Outlier Detection in {name} ---')
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            log(f'{col}: {len(outliers)} outliers detected')


def target_variable_check(train_df):
    log('--- Target Variable Check (Train Only) ---')
    if 'Churn' in train_df.columns:
        counts = train_df['Churn'].value_counts()
        log(f'Churn class distribution: {dict(counts)}')
    else:
        log('Churn column not found in training data.')


def main():
    if os.path.exists(VALIDATION_REPORT):
        os.remove(VALIDATION_REPORT)
    csv_path = os.path.join(REPORTS_DIR, 'data_validation_report.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    log('=== DATA VALIDATION REPORT ===')
    train_df, test_df = load_data()
    # Schema validation (CSV)
    schema_rows = []
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    schema_rows.append(['Schema Validation', 'Train Columns', str(train_cols)])
    schema_rows.append(['Schema Validation', 'Test Columns', str(test_cols)])
    for col in train_cols:
        if col in test_df.columns:
            if train_df[col].dtype != test_df[col].dtype:
                schema_rows.append(['Schema Validation', f'Data type mismatch in {col}', f'train({train_df[col].dtype}) vs test({test_df[col].dtype})'])
    log_csv(schema_rows, ['Section', 'Check', 'Details'])
    # Missing values (CSV)
    missing_rows = []
    for name, df in [('Train', train_df), ('Test', test_df)]:
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                missing_rows.append([f'Missing Values {name}', col, f'{count} missing'])
        empty_str = (df == '').sum()
        for col, count in empty_str.items():
            if count > 0:
                missing_rows.append([f'Missing Values {name}', col, f'{count} empty strings'])
    log_csv(missing_rows, ['Section', 'Column', 'Details'])
    # Value ranges & uniqueness (CSV)
    value_rows = []
    for name, df in [('Train', train_df), ('Test', test_df)]:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                value_rows.append([f'Value Ranges {name}', col, f'min={df[col].min()}, max={df[col].max()}'])
            if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                unique_vals = df[col].unique()
                value_rows.append([f'Unique Values {name}', col, f'{unique_vals}'])
        dup_rows = df.duplicated().sum()
        if dup_rows > 0:
            value_rows.append([f'Duplicates {name}', 'Rows', f'{dup_rows} duplicate rows'])
        if 'customerID' in df.columns:
            dup_ids = df['customerID'].duplicated().sum()
            if dup_ids > 0:
                value_rows.append([f'Duplicates {name}', 'customerID', f'{dup_ids} duplicate customerID values'])
    log_csv(value_rows, ['Section', 'Column', 'Details'])
    # Category overlap (CSV)
    overlap_rows = []
    cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or isinstance(train_df[col].dtype, pd.CategoricalDtype)]
    for col in cat_cols:
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        unseen = test_cats - train_cats
        if unseen:
            overlap_rows.append(['Category Overlap', col, f'unseen in train: {unseen}'])
    log_csv(overlap_rows, ['Section', 'Column', 'Details'])
    # Distribution comparison (CSV)
    dist_rows = []
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        train_stats = train_df[col].describe()
        test_stats = test_df[col].describe()
        dist_rows.append(['Distribution Comparison', col, f'train_mean={train_stats["mean"]:.2f}, test_mean={test_stats["mean"]:.2f}, train_std={train_stats["std"]:.2f}, test_std={test_stats["std"]:.2f}'])
    log_csv(dist_rows, ['Section', 'Column', 'Details'])
    # Outlier detection (CSV)
    outlier_rows = []
    for name, df in [('Train', train_df), ('Test', test_df)]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                outlier_rows.append([f'Outlier Detection {name}', col, f'{len(outliers)} outliers'])
    log_csv(outlier_rows, ['Section', 'Column', 'Details'])
    # Target variable check (CSV)
    target_rows = []
    if 'Churn' in train_df.columns:
        counts = train_df['Churn'].value_counts()
        target_rows.append(['Target Variable', 'Churn', f'class distribution: {dict(counts)}'])
    else:
        target_rows.append(['Target Variable', 'Churn', 'Churn column not found in training data.'])
    log_csv(target_rows, ['Section', 'Column', 'Details'])
    log('=== END OF REPORT ===')

if __name__ == '__main__':
    main()
