import streamlit as st

def show():
    st.title("🧪 Soil Fertility Prediction Input")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Chemical Properties")
            n = st.number_input("Total Nitrogen (%)", 0.0, 2.0, 0.15)
            p = st.number_input("Available Phosphorus (ppm)", 0.0, 200.0, 25.0)
            k = st.number_input("Exchangeable Potassium (cmol/kg)", 0.0, 5.0, 0.20)
            ph = st.number_input("Soil pH", 0.0, 14.0, 5.0)

        with col2:
            st.subheader("Environmental Factors")
            rain = st.number_input("Annual Rainfall (mm)", 0.0, 5000.0, 2000.0)
            temp = st.number_input("Mean Temperature (°C)", 10.0, 40.0, 27.0)
            moist = st.number_input("Soil Moisture (%)", 0.0, 100.0, 15.0)

        submitted = st.form_submit_button("Generate Prediction")
        
        if submitted:
            # Store inputs in Session State
            st.session_state.inputs = [n, p, k, ph, rain, temp, moist]
            # Change the page variable to redirect
            st.session_state.page_selection = "Prediction Result"
            st.rerun()