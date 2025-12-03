import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('oil_palm_fertility_dataset_extended.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'oil_palm_fertility_dataset_extended.csv' is in the folder.")
        return None

# --- DATA AUGMENTATION ---
def augment_data(df):
    df_aug = df.copy()
    target_count = 100 
    
    # Generate Synthetic FERTILE Samples
    current_fertile = len(df_aug[df_aug['Fertility_Class'] == 'Fertile'])
    if current_fertile < target_count:
        needed = target_count - current_fertile
        new_samples = []
        for _ in range(needed):
            new_samples.append({
                'N_Total_Pct': np.random.uniform(0.16, 0.25),
                'P_Avail_ppm': np.random.uniform(30.0, 60.0),
                'K_Exch_cmolkg': np.random.uniform(0.25, 0.60),
                'Soil_pH': np.random.uniform(4.5, 5.5),
                'Annual_Rainfall_mm': np.random.randint(2000, 3000),
                'Mean_Temp_C': np.random.uniform(26.0, 28.0),
                'Soil_Moisture_Pct': np.random.uniform(20.0, 35.0),
                'Fertility_Class': 'Fertile'
            })
        df_aug = pd.concat([df_aug, pd.DataFrame(new_samples)], ignore_index=True)

    # Generate Synthetic MARGINAL Samples
    current_marginal = len(df_aug[df_aug['Fertility_Class'] == 'Marginal'])
    if current_marginal < target_count:
        needed = target_count - current_marginal
        new_samples = []
        for _ in range(needed):
            new_samples.append({
                'N_Total_Pct': np.random.uniform(0.12, 0.15),
                'P_Avail_ppm': np.random.uniform(15.0, 24.0),
                'K_Exch_cmolkg': np.random.uniform(0.15, 0.19),
                'Soil_pH': np.random.uniform(3.8, 4.2),
                'Annual_Rainfall_mm': np.random.randint(1600, 1900),
                'Mean_Temp_C': np.random.uniform(27.0, 29.0),
                'Soil_Moisture_Pct': np.random.uniform(10.0, 15.0),
                'Fertility_Class': 'Marginal'
            })
        df_aug = pd.concat([df_aug, pd.DataFrame(new_samples)], ignore_index=True)
        
    return df_aug

# --- TRAIN MODELS ---
@st.cache_resource
def train_models(original_df):
    data = augment_data(original_df)
    
    feature_cols = ['N_Total_Pct', 'P_Avail_ppm', 'K_Exch_cmolkg', 'Soil_pH', 
                    'Annual_Rainfall_mm', 'Mean_Temp_C', 'Soil_Moisture_Pct']
    target_col = 'Fertility_Class'
    
    data = data.dropna(subset=feature_cols)
    
    X = data[feature_cols]
    y = data[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Deep Learning (MLP)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            trained_models[name] = model
        except Exception as e:
            results[name] = 0.0
    
    # RETURN X_test and y_test NOW
    return results, trained_models, scaler, le, X_test_scaled, y_test