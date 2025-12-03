import streamlit as st
import numpy as np
import pandas as pd
from utils import train_models

def show(df):
    st.title("📊 Prediction Result & Recommendations")
    
    # 1. Validation: Check if inputs exist in Session State
    if 'inputs' not in st.session_state:
        st.warning("⚠️ No input data found. Please go to the **Prediction Input** page first.")
        st.stop()

    # 2. Load Model & Predict
    # NOTE: We use _, _ to ignore X_test and y_test because we don't need them on this page
    results, trained_models, scaler, le, _, _ = train_models(df)
    
    # Select Random Forest as the primary predictor
    model = trained_models["Random Forest"]
    
    # Prepare Input
    # inputs = [N, P, K, pH, Rain, Temp, Moist]
    inputs = st.session_state.inputs
    input_data = np.array([inputs])
    input_scaled = scaler.transform(input_data)
    
    # Generate Prediction
    pred_idx = model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0] # Returns 'Deficient', 'Marginal', or 'Fertile'
    probs = model.predict_proba(input_scaled)
    confidence = np.max(probs) * 100

    # 3. Dynamic UI for Prediction Class (Color Logic)
    st.divider()
    
    # Define colors based on label
    if pred_label == "Fertile":
        color_code = "#28a745" # Green
        msg = "Optimal Soil Condition"
        icon = "✅"
    elif pred_label == "Marginal":
        color_code = "#ffc107" # Yellow (Warning)
        text_color = "black" # Dark text for yellow background
        msg = "Warning: Nutrient Imbalance Detected"
        icon = "⚠️"
    else: # Deficient
        color_code = "#dc3545" # Red (Danger)
        msg = "Critical: Immediate Action Required"
        icon = "🚨"

    # Render the colored card using HTML/CSS
    st.markdown(f"""
    <div style="
        background-color: {color_code};
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;">
        <h4 style='color: white; margin:0; opacity: 0.9;'>Prediction Status</h4>
        <h1 style='color: white; margin:0; font-size: 3rem; font-weight: bold;'>{icon} {pred_label}</h1>
        <p style='color: white; margin-top:10px; font-size: 1.1rem; font-weight: 500;'>{msg}</p>
        <p style='color: white; opacity: 0.8; font-size: 0.9rem;'>Confidence Score: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # 4. Detailed Recommendations Engine (MPOB / Agronomy Standards)
    st.subheader("📝 Nutrient Management Plan")
    st.caption("Recommendations based on MPOB Guidelines for Mature Oil Palm.")

    # Unpack inputs for clearer logic logic below
    # [N, P, K, pH, Rain, Temp, Moist]
    val_n, val_p, val_k, val_ph, val_rain, val_moist = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[6]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primary Nutrients")
        
        # NITROGEN (N) - Vegetative Growth
        if val_n < 0.12:
            st.error(f"**Nitrogen (N) is Very Low ({val_n:.2f}%)**")
            st.markdown("- **Action:** Apply **Urea** (46% N) or **Ammonium Sulphate**.")
            st.markdown("- **Dosage:** 0.75 - 1.0 kg/palm/application.")
            st.markdown("- **Note:** Ensure weed control (Circle Weeding) before application.")
        elif val_n < 0.15:
            st.warning(f"**Nitrogen (N) is Marginal ({val_n:.2f}%)**")
            st.markdown("- **Action:** Apply **Ammonium Sulphate**.")
            st.markdown("- **Strategy:** Split application into 3 rounds per year to reduce leaching.")
        else:
            st.success(f"**Nitrogen (N) is Adequate ({val_n:.2f}%)**")
            st.caption("Maintain standard maintenance program.")

        # POTASSIUM (K) - Fruit Bunch Production
        if val_k < 0.15:
            st.error(f"**Potassium (K) is Critical ({val_k:.2f} cmol/kg)**")
            st.markdown("- **Action:** Apply **Muriate of Potash (MOP)** (60% K2O).")
            st.markdown("- **Dosage:** 2.0 - 3.0 kg/palm/year (High Rate).")
            st.markdown("- **Impact:** Deficiency severely reduces fruit bunch weight.")
        elif val_k < 0.20:
            st.warning(f"**Potassium (K) is Marginal ({val_k:.2f} cmol/kg)**")
            st.markdown("- **Action:** Apply **Muriate of Potash (MOP)**.")
            st.markdown("- **Strategy:** Broadcast evenly on the frond pile area.")
        else:
            st.success(f"**Potassium (K) is Adequate ({val_k:.2f} cmol/kg)**")

    with col2:
        st.markdown("#### Soil & Environment")
        
        # PHOSPHORUS (P) - Root Growth
        if val_p < 15:
            st.error(f"**Phosphorus (P) is Low ({val_p:.0f} ppm)**")
            st.markdown("- **Action:** Apply **Rock Phosphate (RP)** or **TSP**.")
            st.markdown("- **Role:** Essential for root development and energy transfer.")
        else:
            st.success(f"**Phosphorus (P) is Adequate ({val_p:.0f} ppm)**")

        # SOIL pH - Nutrient Availability
        if val_ph < 3.5:
            st.error(f"**Soil is Highly Acidic (pH {val_ph:.1f})**")
            st.markdown("- **Action:** Immediate Liming required.")
            st.markdown("- **Input:** Apply **Ground Magnesium Limestone (GML)** (2-3 kg/palm).")
            st.markdown("- **Risk:** Acid Sulphate toxicity may kill roots.")
        elif val_ph < 4.0:
            st.warning(f"**Soil is Acidic (pH {val_ph:.1f})**")
            st.markdown("- **Action:** Apply **GML** or **Dolomite** to neutralize acidity.")
        else:
            st.success(f"**Soil pH is Optimal (pH {val_ph:.1f})**")

        # MOISTURE / RAINFALL
        if val_moist < 15 or val_rain < 1700:
            st.info("💧 **Water Deficit Alert**")
            st.markdown("- **Management:** **Do NOT apply fertilizer** now (it will vaporize/waste).")
            st.markdown("- **Conservation:** Stack fronds in inter-rows to retain soil moisture.")

    # 5. Action Buttons
    st.divider()
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.page_selection = "Prediction Input"
            st.rerun()