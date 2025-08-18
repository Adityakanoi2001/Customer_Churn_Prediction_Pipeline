import os
import logging
from datetime import datetime
import pandas as pd
import requests

# Logging setup
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ingestion.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ingest_local_data():
    today = datetime.now().strftime('%d-%m-%Y')
    src_path = os.path.join('data', 'raw', 'train', 'Telco-Dataset-train.csv')
    dest_dir = os.path.join('data', 'raw', 'training_data', today)
    dest_path = os.path.join(dest_dir, 'Telco-Dataset-train.csv')
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f'Starting local data ingestion from {src_path}')
    try:
        if not os.path.exists(src_path):
            logger.error(f'Source file not found: {src_path}')
            return False
        df = pd.read_csv(src_path)
        logger.info(f'Read {len(df)} rows from {src_path}')
        try:
            df.to_csv(dest_path, index=False)
            file_size = os.path.getsize(dest_path)
            logger.info(f'Local training data ingested and saved to {dest_path} (size: {file_size} bytes)')
            return True
        except Exception as write_err:
            logger.error(f'Unexpected error while saving to {dest_path}: {write_err}', exc_info=True)
            return False
    except Exception as e:
        logger.error(f'Failed to ingest local training data from {src_path}: {e}', exc_info=True)
        return False

def ingest_remote_data():
    today = datetime.now().strftime('%d-%m-%Y')
    url = 'https://raw.githubusercontent.com/yugipersonalspace/ML-Assignment-PS3/main/Telco-Dataset-test.csv'
    dest_dir = os.path.join('data', 'raw', 'API_test_data', today)
    dest_path = os.path.join(dest_dir, 'Telco-Dataset-test.csv')
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f'Starting remote data ingestion from {url}')
    try:
        response = requests.get(url)
        response.raise_for_status()
        try:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            file_size = os.path.getsize(dest_path)
            logger.info(f'Remote test data downloaded and saved to {dest_path} (size: {file_size} bytes)')
            return True
        except Exception as write_err:
            logger.error(f'Unexpected error while saving remote data to {dest_path}: {write_err}', exc_info=True)
            return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f'Failed to download remote test data from {url}: {req_err}', exc_info=True)
        return False
    except Exception as e:
        logger.error(f'Unexpected error during remote data ingestion: {e}', exc_info=True)
        return False

def main():
    logger.info('--- Data Ingestion Pipeline Started ---')
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    local_status = ingest_local_data()
    remote_status = ingest_remote_data()
    if local_status and remote_status:
        logger.info('--- Data Ingestion Pipeline Completed Successfully ---')
    else:
        logger.warning('--- Data Ingestion Pipeline Completed with Errors ---')

if __name__ == '__main__':
    main()
