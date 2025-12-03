import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
from utils import train_models

def show(df):
    st.title("📈 Model Evaluation Dashboard")
    st.markdown("Compare model performance and analyze how each model makes decisions.")

    # --- BUTTON TO TRIGGER TRAINING ---
    # We check if we already have results in session state to decide whether to show the button or the "Retrain" option
    if 'eval_results' not in st.session_state:
        if st.button("Start Training & Evaluation"):
            with st.spinner("Training models on balanced dataset..."):
                # Unpack all 6 return values from utils.py
                results, trained_models, scaler, le, X_test, y_test = train_models(df)
                
                # SAVE TO SESSION STATE (This fixes the reset issue)
                st.session_state.eval_results = {
                    'results': results,
                    'trained_models': trained_models,
                    'X_test': X_test,
                    'y_test': y_test,
                    'le': le
                }
            st.rerun() # Force a rerun to immediately show the dashboard
    else:
        if st.button("🔄 Retrain Models"):
            # Clear old results and rerun
            del st.session_state.eval_results
            st.rerun()

    # --- DISPLAY DASHBOARD (If results exist) ---
    if 'eval_results' in st.session_state:
        
        # Load data from session state
        data = st.session_state.eval_results
        results = data['results']
        trained_models = data['trained_models']
        X_test = data['X_test']
        y_test = data['y_test']
        le = data['le']

        # --- SECTION 1: OVERALL ACCURACY ---
        st.header("1. Model Accuracy Comparison")
        
        res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
        res_df = res_df.sort_values(by='Accuracy', ascending=False)
        
        # Interactive Bar Chart
        fig_acc = px.bar(res_df, x='Accuracy', y='Model', orientation='h', 
                     color='Accuracy', text_auto='.1%', 
                     title="Accuracy Score (Higher is Better)",
                     range_x=[0, 1.1])
        st.plotly_chart(fig_acc, use_container_width=True)
        
        best_model_name = res_df.iloc[0]['Model']
        st.success(f"🏆 Best Performing Model: **{best_model_name}**")

        st.divider()

        # --- SECTION 2: DEEP DIVE ANALYSIS ---
        st.header("2. Deep Dive Analysis")
        
        # Dropdown to select model (Now works because it's outside the button block!)
        model_names = list(trained_models.keys())
        selected_model_name = st.selectbox("Select a Model to View Details:", model_names)
        
        model = trained_models[selected_model_name]
        
        # Generate Predictions for this specific model
        y_pred = model.predict(X_test)
        target_names = le.classes_ # ['Deficient', 'Fertile', 'Marginal']
        
        col1, col2 = st.columns(2)
        
        # --- CONFUSION MATRIX ---
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            # Create annotated heatmap
            z = cm.tolist()
            x = list(target_names)
            y = list(target_names)
            
            # Use 'Viridis' or 'Blues' for professional look
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
            fig_cm.update_layout(title_text=f'Confusion Matrix: {selected_model_name}',
                                 xaxis_title="Predicted Label",
                                 yaxis_title="True Label")
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption("Diagonal values indicate correct predictions.")

        # --- FEATURE IMPORTANCE ---
        with col2:
            st.subheader("Feature Importance")
            
            # Check if model supports feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = ['N_Total', 'P_Avail', 'K_Exch', 'pH', 'Rainfall', 'Temp', 'Moisture']
                
                feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=True)
                
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                  title=f'What drives {selected_model_name}?',
                                  color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_feat, use_container_width=True)
                
            elif hasattr(model, 'coef_'):
                # For Linear Models (Logistic Regression / SVM)
                # We take the average absolute coefficient across classes
                importances = np.mean(np.abs(model.coef_), axis=0)
                features = ['N_Total', 'P_Avail', 'K_Exch', 'pH', 'Rainfall', 'Temp', 'Moisture']
                
                feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=True)
                
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                  title=f'Feature Coefficients ({selected_model_name})',
                                  color='Importance')
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("Feature importance is not directly available for this model type (e.g., KNN).")

        # --- SECTION 3: CLASSIFICATION REPORT ---
        st.subheader("Detailed Performance Metrics")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['f1-score', 'recall', 'precision']))
    
    else:
        # Initial State info
        st.info("Click the button above to train models and enable the analysis dashboard.")