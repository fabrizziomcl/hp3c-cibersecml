import os
import pandas as pd
import numpy as np
import time
import concurrent.futures
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from pathlib import Path
import joblib

def process_chunk(chunk, feature_cols):
    """
    Function to process a single chunk of data in parallel.
    Applying scaling logic similar to the notebooks.
    """
    try:
        X = chunk.copy()
        X = X[feature_cols]
        
        # Scaling logic: using standard scaling for simplicity in parallel demo
        # In a production pipeline, this would use a fitted scaler object
        for col in X.columns:
            if X[col].std() > 0:
                X[col] = (X[col] - X[col].mean()) / X[col].std()
            else:
                X[col] = 0.0
        
        return X.values
    except Exception as e:
        raise CustomException(e, sys)

class DataTransformation:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.FEATURE_COLS = []

    def initiate_data_transformation(self, num_workers=4, force_imbalance=True):
        try:
            logging.info("Starting robust data transformation aligned with notebooks...")
            
            # Mature MLOps Logic: Load and concatenate all CSVs in the raw directory
            raw_path = self.data_path
            all_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]
            
            if not all_files:
                raise CustomException(f"No CSV files found in {raw_path}", sys)
            
            logging.info(f"Found {len(all_files)} files in raw directory: {all_files}")
            
            df_list = []
            for file in all_files:
                file_path = os.path.join(raw_path, file)
                df_list.append(pd.read_csv(file_path))
            
            df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Concatenated dataset size: {df.shape}")
            
            # 1. Clean Category (remove hashes)
            df['Category'] = df['Category'].str.split('-').str[:2].str.join('-')
            
            # 2. Drop Duplicates
            df.drop_duplicates(keep='first', inplace=True)
            
            # 3. Handle Zero Variance Columns (from notebook findings)
            zero_var_cols = ['pslist.nprocs64bit', 'handles.nport', 'svcscan.interactive_process_services']
            df.drop(columns=[c for c in zero_var_cols if c in df.columns], inplace=True)
            
            # 4. Map Class
            df['Class'] = df['Class'].map({'Benign': 0, 'Malware': 1})
            
            # 5. FORCE IMBALANCE (Simulate real-world scenario)
            if force_imbalance:
                logging.info("Forcing class imbalance for experiment simulation...")
                df_benign = df[df['Class'] == 0]
                df_malware = df[df['Class'] == 1]
                # Sample only 20% of malware to force imbalance
                df_malware_sampled = df_malware.sample(frac=0.2, random_state=42)
                df = pd.concat([df_benign, df_malware_sampled]).sample(frac=1, random_state=42)
                logging.info(f"Imbalance forced. New distribution: {df['Class'].value_counts().to_dict()}")

            # 6. Select numeric features
            target_col = 'Class'
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_features:
                numeric_features.remove(target_col)
            if 'Category' in numeric_features:
                numeric_features.remove('Category')
            
            self.FEATURE_COLS = numeric_features
            
            # --- HPC MEASUREMENT: SEQUENTIAL ---
            start_seq = time.time()
            X_seq = process_chunk(df, self.FEATURE_COLS)
            end_seq = time.time()
            time_seq = end_seq - start_seq

            # --- HPC MEASUREMENT: PARALLEL ---
            chunks = np.array_split(df, num_workers)
            start_par = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_chunk, chunks, [self.FEATURE_COLS]*num_workers))
            X_par = np.vstack(results)
            end_par = time.time()
            time_par = end_par - start_par

            # --- METRICS ---
            speedup = time_seq / time_par
            efficiency = speedup / num_workers
            
            # 7. PCA
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_par)
            
            # Save artifacts
            processed_dir = Path("data/processed")
            os.makedirs(processed_dir, exist_ok=True)
            np.save(processed_dir / "transformed_data.npy", X_pca)
            np.save(processed_dir / "labels.npy", df['Class'].values)
            
            return X_pca, df['Class'].values, {
                "time_seq": time_seq,
                "time_par": time_par,
                "speedup": speedup,
                "efficiency": efficiency
            }

        except Exception as e:
            raise CustomException(e, sys)
