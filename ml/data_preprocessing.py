import os
import re
import pandas as pd
import numpy as np

def convert_sqft_to_num(x):
    """Parses total_sqft string into float. Handles ranges like '2100 - 2850'."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    tokens = x.split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
        except ValueError:
            return None
    try:
        # Extract numeric float using regex if strings like '34.46Sq. Meter' appear
        match = re.search(r"[-+]?\d*\.\d+|\d+", x)
        if match:
            return float(match.group())
        return float(x)
    except ValueError:
        return None

def extract_bhk(size_val):
    """Extract integer BHK from size string (e.g., '2 BHK', '4 Bedroom')."""
    if pd.isna(size_val):
        return 2  # Default median BHK
    size_str = str(size_val).strip()
    match = re.search(r'\d+', size_str)
    if match:
        return int(match.group())
    return 2

def preprocess_bengaluru_data(df):
    """Full data preprocessing pipeline for Kaggle Bengaluru House Price Dataset."""
    df = df.copy()
    
    # 1. Clean location
    df['location'] = df['location'].apply(lambda x: x.strip() if isinstance(x, str) else 'other')
    
    # 2. Extract BHK
    if 'bhk' not in df.columns:
        if 'size' in df.columns:
            df['bhk'] = df['size'].apply(extract_bhk)
        else:
            df['bhk'] = 2
            
    # 3. Clean total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna(subset=['total_sqft', 'price'])
    
    # 4. Handle bath and balcony missing values
    bath_median = df['bath'].median() if 'bath' in df.columns and not df['bath'].isna().all() else 2.0
    balcony_median = df['balcony'].median() if 'balcony' in df.columns and not df['balcony'].isna().all() else 1.0
    
    df['bath'] = df['bath'].fillna(bath_median)
    df['balcony'] = df['balcony'].fillna(balcony_median)
    
    # 5. Outlier Removal 1: Unrealistic sqft per BHK (less than 300 sqft per BHK)
    df = df[~(df['total_sqft'] / df['bhk'] < 300)]
    
    # 6. Calculate price per sqft (Price is in Lakhs, so price * 100000 / sqft)
    df['price_per_sqft'] = (df['price'] * 100000.0) / df['total_sqft']
    
    # 7. Dimensionality reduction on location (locations < 10 rows -> 'other')
    location_stats = df['location'].value_counts()
    locations_less_than_10 = location_stats[location_stats <= 10].index
    df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)
    
    # 8. Outlier Removal 2: Remove price_per_sqft outliers per location (outside 1 std dev)
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        
    if len(df_out) < 100:  # Fallback if too strict
        df_out = df
        
    return df_out.reset_index(drop=True)

if __name__ == "__main__":
    from data.download_data import download_dataset
    csv_path = download_dataset()
    raw_df = pd.read_csv(csv_path)
    print(f"Raw shape: {raw_df.shape}")
    clean_df = preprocess_bengaluru_data(raw_df)
    print(f"Cleaned shape: {clean_df.shape}")
    print(clean_df.head())
