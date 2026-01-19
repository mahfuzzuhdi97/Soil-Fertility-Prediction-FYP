import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from utils import train_models
import numpy as np

def show(df):
    st.title("📈 Model Performance & Evaluation")
    st.markdown("Comparing **13 Machine Learning Algorithms**.")

    # --- 1. TRAINING TRIGGER ---
    if 'eval_results' not in st.session_state:
        st.info("⚠️ Models have not been trained yet. Click below to start benchmarking.")
        if st.button("🚀 Train & Benchmark Models", use_container_width=True):
            with st.spinner("Training models on Perak dataset..."):
                results, trained_models, scaler, le, X_test, y_test = train_models(df)
                st.session_state.eval_results = {
                    'results': results,
                    'trained_models': trained_models,
                    'X_test': X_test,
                    'y_test': y_test,
                    'le': le
                }
            st.rerun()
    else:
        data = st.session_state.eval_results
        results = data['results']
        trained_models = data['trained_models']
        X_test = data['X_test']
        y_test = data['y_test']
        le = data['le']

        # --- 2. CHAMPION MODEL ---
        best_model_name = max(results, key=results.get)
        best_acc = results[best_model_name]
        
        st.divider()
        st.subheader("🏆 The Champion Model")
        
        col_champ1, col_champ2 = st.columns([1, 2])
        with col_champ1:
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #28a745; text-align: center;">
                <h3 style="color: #155724; margin:0;">Winner</h3>
                <h1 style="color: #28a745; font-size: 3rem; margin:0;">{best_model_name}</h1>
                <h2 style="color: #155724; margin:0;">{best_acc:.1%}</h2>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_champ2:
            st.markdown("#### 💡 Analysis")
            st.info(f"{best_model_name} achieved the highest accuracy of {best_acc:.1%} on the test set.")

        st.divider()

        # --- 3. DETAILED PERFORMANCE MATRIX (NEW) ---
        st.subheader("📋 Detailed Performance Matrix")
        st.markdown("Comparing key metrics (Weighted Average) to ensure the model isn't just guessing the majority class.")
        
        metrics_data = []
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            # Calculate metrics (Weighted helps account for class imbalance)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            acc = results[name]
            metrics_data.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            })
        
        # Create DataFrame and Sort by F1-Score (Better for imbalanced data)
        metrics_df = pd.DataFrame(metrics_data).set_index("Model")
        metrics_df = metrics_df.sort_values(by="F1-Score", ascending=False)
        
        # Display with highlighting
        st.dataframe(
            metrics_df.style.format("{:.2%}") \
            .background_gradient(cmap="Greens", subset=["F1-Score", "Accuracy"]),
            use_container_width=True
        )

        # --- 4. ACCURACY LEADERBOARD ---
        st.subheader("📊 Model Comparison Chart")
        
        res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=True)
        res_df['Color'] = ['Winner' if x == best_model_name else 'Contender' for x in res_df['Model']]
        color_map = {'Winner': '#28a745', 'Contender': '#6c757d'}
        
        fig_acc = px.bar(res_df, x='Accuracy', y='Model', orientation='h', 
                         color='Color', color_discrete_map=color_map,
                         text_auto='.1%', title="Model Accuracy Comparison")
        fig_acc.update_layout(showlegend=False, xaxis_range=[0, 1.1])
        st.plotly_chart(fig_acc, use_container_width=True)

        # --- 5. DEEP DIVE ---
        st.divider()
        st.subheader("🔍 Deep Dive Analysis")
        
        selected_model_name = st.selectbox("Select a Model to Inspect:", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
        model = trained_models[selected_model_name]
        y_pred = model.predict(X_test)
        target_names = list(le.classes_)

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = ff.create_annotated_heatmap(z=cm.tolist(), x=target_names, y=target_names, colorscale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            st.markdown(f"**Feature Importance**")
            feats = ['N_mgkg', 'P_mgkg', 'K_mgkg', 'pH', 'Rain', 'Temp', 'Moisture', 'Humidity']
            
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values(by='Importance', ascending=True)
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Greens')
                st.plotly_chart(fig_feat, use_container_width=True)
            elif hasattr(model, 'coef_'):
                imp = np.mean(np.abs(model.coef_), axis=0)
                feat_df = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values(by='Importance', ascending=True)
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("Feature importance not available for this algorithm.")

        # --- 6. ADVANCED VISUAL JUSTIFICATION ---
        st.divider()
        st.subheader("💡 Advanced Visual Justification")
        
        tab_roc, tab_conf = st.tabs(["1. ROC-AUC Curves", "2. Prediction Confidence"])
        
        # A. ROC Curves
        with tab_roc:
            if hasattr(model, "predict_proba"):
                try:
                    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                    n_classes = y_test_bin.shape[1]
                    y_score = model.predict_proba(X_test)

                    fig_roc = go.Figure()
                    colors = ['#dc3545', '#28a745', '#ffc107']
                    
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                                    mode='lines', 
                                                    name=f'{target_names[i]} (AUC = {roc_auc:.2f})',
                                                    line=dict(color=colors[i], width=2)))

                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    fig_roc.update_layout(title=f"ROC Curve for {selected_model_name}", xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=700, height=500)
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate ROC curve: {e}")
            else:
                st.info("Selected model does not support probabilities.")

        # B. Confidence
        with tab_conf:
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_test)
                max_probs = np.max(probas, axis=1)
                conf_df = pd.DataFrame({'Confidence': max_probs, 'True Label': [target_names[i] for i in y_test]})
                fig_conf = px.histogram(conf_df, x="Confidence", color="True Label", nbins=20, 
                                        title=f"Prediction Confidence Distribution ({selected_model_name})",
                                        color_discrete_map={'Fertile': '#28a745', 'Marginal': '#ffc107', 'Deficient': '#dc3545'})
                st.plotly_chart(fig_conf, use_container_width=True)

        if st.button("🔄 Retrain All Models"):
            del st.session_state.eval_results
            st.rerun()
