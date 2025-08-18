import sqlite3
import pandas as pd
import os

# Path to the SQLite feature store
db_path = os.path.join('data', 'features', 'churn_features.db')

def load_features_and_target(table_name='churn_features', target_col='Churn'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    return X, y

if __name__ == '__main__':
    X, y = load_features_and_target()
    print(f'Features shape: {X.shape}')
    if y is not None:
        print(f'Target shape: {y.shape}')
        print(f'Target value counts:\n{y.value_counts()}')
    else:
        print('Target column not found.')
