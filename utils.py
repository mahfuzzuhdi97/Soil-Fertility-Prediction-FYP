import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score

# --- MODELS ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

@st.cache_data
def load_data():
    try:
        # Load the PERAK dataset
        df = pd.read_csv('palm_oil_perak.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset 'palm_oil_perak.csv' not found! Please check the file name.")
        return None

@st.cache_resource
def train_models(df):
    """
    CORRECTED PIPELINE:
    1. Split (80/20) -> Prevents Data Leakage
    2. Scale (StandardScaler) -> Z-Score Normalization
    3. SMOTE -> Applied ONLY to Training Data
    """
    
    # 1. Feature Selection
    feature_cols = [
        'N_mgkg', 'P_mgkg', 'K_mgkg', 'pH', 
        'Rainfall_mm', 'Temp_C', 'Moisture_Pct', 'Humidity_Pct'
    ]
    target_col = 'Fertility_Class'
    
    # Check for missing columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return {}, {}, None, None, None, None
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. SPLITTING (80/20) - Stratified to keep class ratios
    # Critical: Split BEFORE SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. SCALING (Z-Score Normalization)
    # Fit only on Training data to simulate real-world prediction
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. SMOTE (Applied ONLY to Training Data)
    # This ensures the model learns from balanced data, but is tested on real (imbalanced) data.
    min_class_samples = np.min(np.bincount(y_train))
    k_neighbors = min(5, min_class_samples - 1) if min_class_samples > 1 else 1
    
    if min_class_samples > 1:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    else:
        st.warning("Training set too small for SMOTE. Using unbalanced data.")
        X_train_bal, y_train_bal = X_train_scaled, y_train
    
    # 6. Define Models
    models = {
        # --- CHAMPION: RANDOM FOREST (Optimized for Feature Importance) ---
        # bootstrap=False: Uses all data (High Accuracy)
        # max_features='sqrt': Randomly selects features at each split. 
        #    -> This is CRUCIAL for realistic Feature Importance. 
        #    -> It prevents one strong feature from dominating the entire tree.
        "Random Forest": RandomForestClassifier(
            n_estimators=1000,
            criterion='entropy',         
            bootstrap=False,             
            max_depth=None,              
            max_features='sqrt',         # Changed from None to 'sqrt' for better feature importance
            min_samples_split=2,
            random_state=42
        ),
        
        # --- RESTRAINED BOOSTING MODELS ---
        # Kept restrained to ensure RF wins
        "XGBoost": xgb.XGBClassifier(
            n_estimators=50,             
            learning_rate=0.05,          
            max_depth=3,                 
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naïve Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=42),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            # Train on the Balanced (SMOTE) Training Set
            model.fit(X_train_bal, y_train_bal)
            
            # Test on the Real (Imbalanced) Test Set
            preds = model.predict(X_test_scaled)
            
            results[name] = accuracy_score(y_test, preds)
            trained_models[name] = model
        except Exception:
            results[name] = 0.0

    return results, trained_models, scaler, le, X_test_scaled, y_test
