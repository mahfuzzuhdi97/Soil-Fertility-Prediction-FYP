import streamlit as st
import plotly.graph_objects as go


def show(df):
    st.title("🌴 Tree Health Status Dashboard")
    
    if df is not None:
        sample_ids = df['Sample_ID'].unique()
        selected_id = st.selectbox("Select Plant ID (Sample ID):", sample_ids)
        
        row = df[df['Sample_ID'] == selected_id].iloc[0]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=150)
            st.metric("Tree Age", f"{row['Palm_Age_Yrs']} Years")
            st.metric("Fertility Class", row['Fertility_Class'])

        with col2:
            st.subheader("Nutrient Profile")
            categories = ['Nitrogen', 'Phosphorus', 'Potassium', 'Magnesium', 'pH']
            values = [
                row['N_Total_Pct']/0.25, row['P_Avail_ppm']/60, 
                row['K_Exch_cmolkg']/1.0, row['Mg_Exch_cmolkg']/0.7, row['Soil_pH']/7.0
            ]
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name=selected_id))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
            st.plotly_chart(fig)

        with col3:
            st.write(f"**Nitrogen:** {row['N_Status']}")
            st.write(f"**Phosphorus:** {row['P_Status']}")
            st.write(f"**Potassium:** {row['K_Status']}")
