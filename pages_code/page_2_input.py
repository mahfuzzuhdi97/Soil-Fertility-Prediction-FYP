import streamlit as st

def show(df):
    st.title("🧪 Prediction Input")
    st.markdown("Enter soil data (mg/kg) and climate parameters.")
    
    with st.form("input_form"):
        # --- Group 1: Soil Nutrients ---
        st.subheader("1. Soil Nutrients (mg/kg)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1: 
            n = st.number_input("N Total (mg/kg)", 0.0, 5000.0, 1300.0, step=10.0,
                                help="Target: > 1500 mg/kg")
        with col2: 
            p = st.number_input("P Avail (mg/kg)", 0.0, 500.0, 25.0, step=1.0,
                                help="Target: > 30 mg/kg")
        with col3: 
            k = st.number_input("K Exch (mg/kg)", 0.0, 1000.0, 80.0, step=10.0,
                                help="Target: > 80 mg/kg")
        with col4: 
            ph = st.number_input("pH Level", 0.0, 14.0, 4.5, step=0.1,
                                help="Optimal: 4.5 - 6.5")
        
        # --- Group 2: Climate & Physics (Updated Aeration -> Humidity) ---
        st.subheader("2. Physical & Climate")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5: 
            rain = st.number_input("Rainfall (mm)", 0.0, 5000.0, 2000.0, step=50.0,
                                   help="Optimal: 2000 - 3000 mm")
        with col6: 
            temp = st.number_input("Avg Temp (°C)", 10.0, 50.0, 28.0, step=0.5,
                                   help="Optimal: 24 - 32 °C")
        with col7: 
            moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 25.0, step=1.0,
                                       help="Target: > 25%")
        with col8: 
            # CHANGED: Aeration -> Humidity
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 75.0, step=1.0,
                                  help="Optimal: 70 - 90%")
        
        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)
        
        if submitted:
            # Save inputs (New Order: N, P, K, pH, Rain, Temp, Moisture, Humidity)
            st.session_state.inputs = [n, p, k, ph, rain, temp, moisture, humidity]
            st.session_state.page_selection = "📋 Prediction Result"
            st.rerun()
