import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
from utils import train_models

# --- 1. EXPERT RULE OVERRIDE ---
def apply_expert_rules(pred_label, inputs):
    """
    Overrides ML model for obvious cases.
    Inputs: [N, P, K, pH, Rain, Temp, Moisture, Humidity]
    """
    n, p, k, ph, rain, temp, moisture, humidity = inputs
    
    # RULE 1: ABSOLUTE FERTILE
    if n > 1500 and p > 30 and k > 80 and ph > 4.5:
        return "Fertile", "Expert Rule: Nutrients exceed optimal levels."

    # RULE 2: ABSOLUTE DEFICIENT
    if n < 800 or k < 40 or ph < 3.0:
        return "Deficient", "Expert Rule: Critical deficiency detected (Liebig's Law)."

    # Default: Trust ML
    return pred_label, "AI Model Prediction based on learned patterns."

# --- 2. RECOMMENDATION ENGINE ---
def get_mpob_recommendations(fertility_class, n, p, k, ph, humidity, moisture):
    recs = {"General": [], "Specific": [], "Correction": []}
    
    # General
    if fertility_class == "Deficient":
        recs["General"] = ["🔴 **Strategy: Corrective Manuring**", "Increase frequency to 4-6 rounds/year."]
    elif fertility_class == "Marginal":
        recs["General"] = ["🟡 **Strategy: Recovery**", "Focus on nutrient balance."]
    else:
        recs["General"] = ["🟢 **Strategy: Maintenance**", "Replace nutrients removed by harvest."]

    # Nutrients
    if n < 1500: recs["Specific"].append("- **Nitrogen Low:** Apply Ammonium Sulphate.")
    if p < 25: recs["Specific"].append("- **Phosphorus Low:** Apply Rock Phosphate.")
    if k < 80: recs["Specific"].append("- **Potassium Low:** Apply Muriate of Potash (MOP).")

    # Correction (Updated for Humidity)
    if ph < 4.0: recs["Correction"].append("- **Acidity:** Apply Limestone/GML.")
    if moisture < 15.0: recs["Correction"].append("- **Low Moisture:** Improve drainage and apply mulch.")
    if humidity < 60.0: recs["Correction"].append("- **Low Humidity:** Consider cover crops to maintain microclimate.")
    
    return recs

def show(df):
    st.title("📋 Prediction Result")
    
    if 'inputs' not in st.session_state:
        st.warning("⚠️ No input data found.")
        st.stop()

    # 1. Predict
    results, trained_models, scaler, le, _, _ = train_models(df)
    model = trained_models["Random Forest"]
    inputs = st.session_state.inputs 
    input_scaled = scaler.transform([inputs])
    
    pred_idx = model.predict(input_scaled)[0]
    raw_label = le.inverse_transform([pred_idx])[0]
    
    # 2. Expert Correction
    final_label, logic_reason = apply_expert_rules(raw_label, inputs)
    
    confidence = 100.0 if final_label != raw_label else np.max(model.predict_proba(input_scaled)) * 100
    source = "Rule Base" if final_label != raw_label else "Random Forest"

    # 3. Display
    if final_label == "Fertile": color, icon = "#28a745", "🌟"
    elif final_label == "Marginal": color, icon = "#ffc107", "⚠️"
    else: color, icon = "#dc3545", "🚨"

    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 25px;">
        <h1 style='color: white; margin:0;'>{icon} {final_label}</h1>
        <p style='color: white;'>Confidence: {confidence:.2f}% ({source})</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"💡 **Diagnosis Logic:** {logic_reason}")
    
    # 4. Advice (Inputs: N, P, K, pH, Rain, Temp, Moisture, Humidity)
    # Pass humidity (index 7) and moisture (index 6)
    advice = get_mpob_recommendations(final_label, inputs[0], inputs[1], inputs[2], inputs[3], inputs[7], inputs[6])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💊 Fertilizer Plan")
        if advice["Specific"]: 
            for x in advice["Specific"]: st.write(x)
        else: st.success("Nutrients optimal.")
    with col2:
        st.subheader("🚜 Soil Management")
        if advice["Correction"]: 
            for x in advice["Correction"]: st.write(x)
        else: st.success("Soil structure healthy.")

    st.divider()

    # 5. Summary
    st.subheader("📝 Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", f"{inputs[0]} mg/kg")
    c2.metric("P", f"{inputs[1]} mg/kg")
    c3.metric("K", f"{inputs[2]} mg/kg")
    c4.metric("pH", f"{inputs[3]}")
    
    # Download
    report = f"PALM REPORT\nDate: {pd.Timestamp.now()}\nStatus: {final_label}\nLogic: {logic_reason}\nInputs: {inputs}"
    st.download_button("📥 Download Report", report, "report.txt")

    if st.button("🔄 New Prediction", use_container_width=True):
        st.session_state.page_selection = "🧪 Prediction Input"
        st.rerun()
