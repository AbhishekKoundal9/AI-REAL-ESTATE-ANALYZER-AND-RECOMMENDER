import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.download_data import download_dataset
from ml.data_preprocessing import preprocess_bengaluru_data

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def train_and_evaluate():
    # 1. Download and preprocess data
    csv_path = download_dataset()
    raw_df = pd.read_csv(csv_path)
    print(f"Loaded raw dataset with {len(raw_df)} records.")
    
    clean_df = preprocess_bengaluru_data(raw_df)
    print(f"Preprocessed dataset shape: {clean_df.shape}")
    
    # Save cleaned dataset for quick recommendation search
    clean_csv_path = os.path.join(SAVED_MODELS_DIR, "cleaned_dataset.csv")
    clean_df.to_csv(clean_csv_path, index=False)
    
    # Unique locations list
    locations = sorted(clean_df['location'].unique().tolist())
    with open(os.path.join(SAVED_MODELS_DIR, "locations.json"), "w") as f:
        json.dump(locations, f, indent=2)
        
    # 2. Separate Features (X) and Target (y)
    feature_cols = ['location', 'total_sqft', 'bhk', 'bath', 'balcony']
    X = clean_df[feature_cols]
    y = clean_df['price']  # Price in Lakhs
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Create Preprocessing Pipeline (One-Hot Encode location, pass through numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            ('loc_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['location']),
            ('num_pass', 'passthrough', ['total_sqft', 'bhk', 'bath', 'balcony'])
        ]
    )
    
    # 4. Train Model 1: Linear Regression
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    mae_lr = float(mean_absolute_error(y_test, y_pred_lr))
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    r2_lr = float(r2_score(y_test, y_pred_lr))
    
    # 5. Train Model 2: Random Forest Regressor
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    r2_rf = float(r2_score(y_test, y_pred_rf))
    
    print("\n================ MODEL EVALUATION RESULTS ================")
    print(f"Linear Regression    -> MAE: {mae_lr:.2f} Lakhs | RMSE: {rmse_lr:.2f} Lakhs | R2 Score: {r2_lr:.4f}")
    print(f"Random Forest        -> MAE: {mae_rf:.2f} Lakhs | RMSE: {rmse_rf:.2f} Lakhs | R2 Score: {r2_rf:.4f}")
    print("===========================================================\n")
    
    metrics = {
        "linear_regression": {
            "name": "Linear Regression",
            "mae": round(mae_lr, 2),
            "rmse": round(rmse_lr, 2),
            "r2_score": round(r2_lr, 4)
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "mae": round(mae_rf, 2),
            "rmse": round(rmse_rf, 2),
            "r2_score": round(r2_rf, 4)
        }
    }
    
    # Select best model based on R2 Score
    if r2_rf >= r2_lr:
        best_model_name = "Random Forest Regressor"
        best_pipeline = rf_pipeline
    else:
        best_model_name = "Linear Regression"
        best_pipeline = lr_pipeline
        
    metrics["best_model"] = best_model_name
    print(f"Best Performing Model Selected: {best_model_name}")
    
    # Save best pipeline & metrics
    joblib.dump(best_pipeline, os.path.join(SAVED_MODELS_DIR, "best_model.joblib"))
    joblib.dump(lr_pipeline, os.path.join(SAVED_MODELS_DIR, "lr_model.joblib"))
    joblib.dump(rf_pipeline, os.path.join(SAVED_MODELS_DIR, "rf_model.joblib"))
    
    with open(os.path.join(SAVED_MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Artifacts successfully saved to {SAVED_MODELS_DIR}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
