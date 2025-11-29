
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import gc

class OptimizedXGBImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_numeric, cols_object, n_jobs=1):
        self.cols_numeric = cols_numeric
        self.cols_object = cols_object
        self.models = {} 
        self.encoders = {} 
        self.n_jobs = n_jobs 

    def fit(self, X, y=None):
        print("--- Bắt đầu huấn luyện Imputer (Optimized) ---")
        data = X.copy()
        
        # 1. Fit Encoders
        for col in self.cols_object:
            if col in data.columns:
                valid_data = data[col].dropna().values.reshape(-1, 1)
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(valid_data)
                self.encoders[col] = encoder
                
                non_null_idx = data[col].notna()
                data.loc[non_null_idx, col] = encoder.transform(data.loc[non_null_idx, col].values.reshape(-1, 1)).flatten()
                data[col] = data[col].astype(float)

        all_cols = self.cols_object + self.cols_numeric
        
        # 2. Train XGBoost
        for col in all_cols:
            if col not in data.columns: continue
            
            missing_count = data[col].isna().sum()
            if missing_count > 0 and missing_count < len(data):
                print(f"Training model cho cột: {col} (Missing: {missing_count})")
                train_data = data[data[col].notna()]
                X_train = train_data.drop(columns=[col])
                y_train = train_data[col]
                
                if col in self.cols_numeric:
                    model = XGBRegressor(n_jobs=self.n_jobs, random_state=42, n_estimators=100, tree_method='hist')
                else:
                    model = XGBClassifier(n_jobs=self.n_jobs, random_state=42, eval_metric='logloss', use_label_encoder=False, n_estimators=100, tree_method='hist')
                
                model.fit(X_train, y_train)
                self.models[col] = model
                
                del X_train, y_train, train_data
                gc.collect()
        print("--- Huấn luyện hoàn tất ---")
        return self

    def transform(self, X):
        data = X.copy()
        
        # 1. Encode
        for col in self.cols_object:
            if col in data.columns and col in self.encoders:
                encoder = self.encoders[col]
                non_null_idx = data[col].notna()
                if non_null_idx.sum() > 0:
                    encoded_vals = encoder.transform(data.loc[non_null_idx, col].values.reshape(-1, 1)).flatten()
                    data.loc[non_null_idx, col] = encoded_vals
                data[col] = data[col].astype(float)

        all_cols = self.cols_object + self.cols_numeric
        
        # 2. Predict Missing
        for col in all_cols:
            if col in data.columns and data[col].isna().sum() > 0:
                if col in self.models:
                    model = self.models[col]
                    null_idx = data.index[data[col].isna()]
                    X_pred = data.loc[null_idx].drop(columns=[col])
                    preds = model.predict(X_pred)
                    data.loc[null_idx, col] = preds
                    del X_pred, preds
                    gc.collect()

        # 3. Decode
        for col in self.cols_object:
            if col in data.columns and col in self.encoders:
                encoder = self.encoders[col]
                if pd.api.types.is_float_dtype(data[col]):
                     data[col] = data[col].round().astype(int)
                try:
                    valid_mask = data[col] != -1
                    if valid_mask.any():
                        reshaped = data.loc[valid_mask, col].values.reshape(-1, 1)
                        data.loc[valid_mask, col] = encoder.inverse_transform(reshaped).flatten()
                except Exception as e:
                    print(f"Lỗi decode cột {col}: {e}")
        return data
