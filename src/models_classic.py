import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from kan import KAN

class ClassicModels:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.le = LabelEncoder()
        self.kan_model = None

    def prepare_data(self, train_df, test_df):
        X_train = train_df.drop('label', axis=1)
        y_train = self.le.fit_transform(train_df['label'])
        
        X_test = test_df.drop('label', axis=1)
        y_test = self.le.transform(test_df['label'])
        
        return X_train, y_train, X_test, y_test

    def train_traditional(self, X_train, y_train):
        print("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        print("Training XGBoost...")
        self.xgb.fit(X_train, y_train)

    def train_kan(self, X_train, y_train, steps=3):
        print(f"Initializing KAN on {len(X_train)} samples...")
        
        # Convert to Tensors (X as float, y reshaped for KAN)
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        
        # width=[inputs, hidden, outputs], grid=3 for speed
        self.kan_model = KAN(width=[X_train.shape[1], 2, 1], grid=3, k=2)
        
        dataset = {
            'train_input': X_tensor,
            'train_label': y_tensor,
            'test_input': X_tensor[:100],
            'test_label': y_tensor[:100]
        }
        
        print("Training KAN with Adam (Lightweight mode)...")
        # Use .fit() instead of .train() for newer KAN versions
        self.kan_model.fit(dataset, opt="Adam", steps=steps, lr=0.01)

    def get_predictions(self, X_test):
        rf_preds = self.rf.predict(X_test)
        xgb_preds = self.xgb.predict(X_test)
        
        # KAN Inference
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        kan_logits = self.kan_model(X_test_tensor)
        
        # Convert continuous KAN output to binary (0 or 1)
        kan_preds = (kan_logits > 0.5).int().flatten().numpy()
        
        return {
            'rf': self.le.inverse_transform(rf_preds),
            'xgb': self.le.inverse_transform(xgb_preds),
            'kan': self.le.inverse_transform(kan_preds)
        }
        