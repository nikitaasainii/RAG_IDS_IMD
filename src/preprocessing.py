import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .config import COLUMNS

def load_arff_to_dataframe(file_path):
    """
    Manually parses NSL-KDD .arff files into a Pandas DataFrame.
    This handles the metadata lines starting with '@' and empty lines.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        data_lines = []
        for line in lines:
            line = line.strip()
            # Skip ARFF header lines and empty space
            if not line or line.startswith('@') or line.startswith('%'):
                continue
            data_lines.append(line)

        # Convert the list of strings into a single CSV-formatted string
        clean_data = '\n'.join(data_lines)
        
        # Load into DataFrame using the column names from config.py
        df = pd.read_csv(StringIO(clean_data), names=COLUMNS, header=None)
        
        # Drop difficulty level as requested (it's not used for detection)
        if 'difficulty_level' in df.columns:
            df = df.drop('difficulty_level', axis=1)
            
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def preprocess_dataframe(df):
    """
    Prepares the dataframe for both RAG and Machine Learning models.
    Includes cleaning, encoding, and scaling.
    """
    df_copy = df.copy()
    
    # 1. Handle missing values
    df_copy = df_copy.dropna(subset=['label'])
    
    # 2. Encode Categorical Features (protocol_type, service, flag)
    le = LabelEncoder()
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    
    # 3. Normalize Numerical Features
    # We scale everything EXCEPT the target 'label'
    scaler = MinMaxScaler()
    num_cols = df_copy.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != 'label']
    
    df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
    
    return df_copy

def row_to_string(row):
    """
    Converts a single dataframe row into a readable string for ChromaDB/LLM.
    Example: "duration: 0, protocol_type: 1, service: 20..."
    """
    return ", ".join([f"{col}: {val}" for col, val in row.items() if col != 'label'])