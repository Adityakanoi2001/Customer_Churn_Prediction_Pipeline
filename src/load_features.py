import sqlite3
import pandas as pd
import os
import logging
import numpy as np

# 1. Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 2. Configuration
DB_PATH = os.path.join('data', 'features', 'churn_features.db')
TABLE_NAME = 'churn_features'
TARGET_COL = 'Churn'  # expects 0/1, "Yes/No", "True/False", etc.

def _normalize_target(y: pd.Series) -> pd.Series:
    """
    Map common churn labels to 0/1 and return int dtype.
    FIXED: Preserve both 0 and 1 classes properly.
    """
    logger.info(f"Original target values before normalization: {y.value_counts()}")
    logger.info(f"Original target dtype: {y.dtype}")
    
    # If already numeric, handle carefully
    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(0)
        unique_vals = sorted(y.unique())
        logger.info(f"Unique numeric values in target: {unique_vals}")
        
        # If already binary (0,1), keep as is
        if set(unique_vals).issubset({0, 1}):
            logger.info("Target is already binary (0,1). Keeping as is.")
            return y.astype(int)
        
        # If it's exactly (0,1) or (1,2) or similar, map appropriately
        if len(unique_vals) == 2:
            logger.info(f"Binary target with values {unique_vals}. Mapping to (0,1).")
            min_val, max_val = unique_vals
            y_norm = (y == max_val).astype(int)  # Map larger value to 1
            logger.info(f"Mapped {min_val}→0, {max_val}→1")
            return y_norm
        
        # For multi-class or other numeric, be more careful
        logger.warning(f"Multi-class numeric target detected: {unique_vals}")
        # Could map to binary based on your business logic
        # For now, treat 0 as 0, everything else as 1
        y_norm = (y != 0).astype(int)
        logger.info("Mapped: 0→0, everything else→1")
        return y_norm

    # Handle strings like "Yes"/"No", "True"/"False"
    logger.info("Processing string target values")
    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "0": 0
    }
    
    # Show what we're working with
    unique_str_vals = y.astype(str).str.strip().str.lower().unique()
    logger.info(f"Unique string values (lowercased): {unique_str_vals}")
    
    y_norm = (
        y.astype(str)
         .str.strip()
         .str.lower()
         .map(mapping)
         .fillna(0)  # unknowns → 0
         .astype(int)
    )
    
    # Log the mapping results
    logger.info(f"After string mapping: {y_norm.value_counts()}")
    return y_norm

def _coerce_numeric_to_float64(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all numeric columns to float64 so MLflow schema works cleanly with NaNs.
    """
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if num_cols:
        # Cast bool -> int -> float to avoid pandas warnings
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
        df[num_cols] = df[num_cols].astype("float64")
    return df

# 3. Feature Retrieval Function
def get_features_for_training(db_path=DB_PATH, table_name=TABLE_NAME, target_col=TARGET_COL):
    logger.info(f'Connecting to SQLite DB at {db_path}')
    conn = sqlite3.connect(db_path)
    try:
        # List all tables for debugging
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        logger.info(f"Available tables: {tables['name'].tolist()}")
        df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        logger.info(f'Retrieved {df.shape[0]} rows and {df.shape[1]} columns from table {table_name}')
    except Exception as e:
        logger.error(f'Error reading table {table_name}: {e}')
        conn.close()
        raise
    finally:
        conn.close()

    # Drop any obvious index-like columns if present
    for idx_col in ["index", "Idx", "ID", "Id", "id"]:
        if idx_col in df.columns and df[idx_col].is_monotonic_increasing:
            logger.info(f"Dropping index column: {idx_col}")
            df = df.drop(columns=[idx_col])

    if target_col in df.columns:
        logger.info(f"Found target column '{target_col}'. Processing...")
        
        # Show raw target stats before normalization
        raw_target = df[target_col]
        logger.info(f"Raw target column stats:")
        logger.info(f"  - Type: {raw_target.dtype}")
        logger.info(f"  - Unique values: {sorted(raw_target.unique())}")
        logger.info(f"  - Value counts:\n{raw_target.value_counts()}")
        
        y = _normalize_target(raw_target)
        X = df.drop(columns=[target_col])
        
        # Final check
        logger.info(f"Final normalized target stats:")
        logger.info(f"  - Unique values: {sorted(y.unique())}")
        logger.info(f"  - Value counts:\n{y.value_counts()}")
        logger.info(f'Features shape before dtype fix: {X.shape}, Target shape: {y.shape}')
        
    else:
        X = df
        y = None
        logger.warning(f'Target column {target_col} not found in table {table_name}')
        logger.info(f"Available columns: {df.columns.tolist()}")

    # Ensure all numeric features are float64 (prevents MLflow schema warning)
    X = _coerce_numeric_to_float64(X)

    logger.info(f'Features dtypes after fix:\n{X.dtypes}')
    return X, y

# 4. Test Block with Enhanced Debugging
if __name__ == '__main__':
    logger.info("="*50)
    logger.info("TESTING load_features.py")
    logger.info("="*50)
    
    X, y = get_features_for_training()
    
    print('\n' + "="*30)
    print('FEATURES (X) SUMMARY:')
    print("="*30)
    print(f'Shape: {X.shape}')
    print('First 5 rows:')
    print(X.head())
    
    if y is not None:
        print('\n' + "="*30)
        print('TARGET (y) SUMMARY:')
        print("="*30)
        print(f'Shape: {y.shape}')
        print(f'Data type: {y.dtype}')
        print('First 10 values:', y.head(10).tolist())
        print('\nFull value counts:')
        print(y.value_counts().sort_index())
        
        # Check if we have both classes
        unique_classes = sorted(y.unique())
        print(f'\nUnique classes: {unique_classes}')
        if len(unique_classes) >= 2:
            print("✅ SUCCESS: Multiple classes detected!")
        else:
            print("❌ WARNING: Only one class detected!")
            print("This will cause training to fail.")
    else:
        print('\n❌ ERROR: Target column not found.')
        
    logger.info("Testing complete.")