import streamlit as st
import plotly.express as px
import pandas as pd

def show(df):
    st.title("📊 Plantation Health Dashboard")
    st.markdown("Real-time overview of nutrient levels across the plantation.")
    
    # --- 1. TRAFFIC LIGHT METRICS ---
    st.subheader("1. Farm Health Snapshot")
    
    if 'N_mgkg' in df.columns:
        # Calculate Averages
        avg_n = df['N_mgkg'].mean()
        avg_p = df['P_mgkg'].mean()
        avg_k = df['K_mgkg'].mean()
        avg_ph = df['pH'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        # New Thresholds based on generated dataset:
        # N > 1500 mg/kg | P > 25 mg/kg | K > 80 mg/kg | pH 4.5-6.5
        with col1: 
            st.metric("Avg Nitrogen (N)", f"{avg_n:.0f} mg/kg", 
                      delta="Target: >1500", 
                      delta_color="normal" if avg_n >= 1500 else "inverse")
        with col2: 
            st.metric("Avg Phosphorus (P)", f"{avg_p:.0f} mg/kg", 
                      delta="Target: >25", 
                      delta_color="normal" if avg_p >= 25 else "inverse")
        with col3: 
            st.metric("Avg Potassium (K)", f"{avg_k:.0f} mg/kg", 
                      delta="Target: >80", 
                      delta_color="normal" if avg_k >= 80 else "inverse")
        with col4: 
            st.metric("Avg pH Level", f"{avg_ph:.1f}", 
                      delta="Optimal: 4.5 - 6.5", 
                      delta_color="normal" if 4.5 <= avg_ph <= 6.5 else "off")
    else:
        st.error("Dataset columns not recognized. Please check CSV.")
        return

    st.divider()

    # --- 2. DEFICIENCY ANALYSIS ---
    col_chart1, col_chart2 = st.columns([1, 1.5])
    
    with col_chart1:
        st.subheader("2. Overall Status")
        if 'Fertility_Class' in df.columns:
            counts = df['Fertility_Class'].value_counts().reset_index()
            counts.columns = ['Status', 'Count']
            color_map = {'Fertile': '#28a745', 'Marginal': '#ffc107', 'Deficient': '#dc3545'}
            
            fig_pie = px.pie(counts, values='Count', names='Status', hole=0.4,
                             color='Status', color_discrete_map=color_map)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("3. Deficiency Analysis")
        st.markdown("**% of Samples Below Critical Levels**")
        
        # Calculate deficiency percentages (New Thresholds)
        def_n = (df['N_mgkg'] < 1500).mean() * 100
        def_p = (df['P_mgkg'] < 25).mean() * 100
        def_k = (df['K_mgkg'] < 80).mean() * 100
        def_acid = (df['pH'] < 4.0).mean() * 100
        
        data = pd.DataFrame({
            'Issue': ['Low Nitrogen (<1500)', 'Low Phosphorus (<25)', 'Low Potassium (<80)', 'High Acidity (pH<4.0)'],
            'Severity': [def_n, def_p, def_k, def_acid]
        })
        
        fig_bar = px.bar(data, x='Severity', y='Issue', orientation='h',
                         color='Severity', color_continuous_scale='Reds', text_auto='.1f%')
        fig_bar.update_layout(xaxis_title="% of Farm Affected", xaxis_range=[0,100])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- 4. DETAILED DISTRIBUTION ---
    st.subheader("4. Nutrient Distribution vs Target")
    tab1, tab2, tab3 = st.tabs(["Nitrogen", "Phosphorus", "Potassium"])
    
    def plot_dist(col, target, color):
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=[color])
        fig.add_vline(x=target, line_dash="dash", line_color="green", annotation_text="Target")
        return fig

    with tab1: st.plotly_chart(plot_dist('N_mgkg', 1500, '#1f77b4'), use_container_width=True)
    with tab2: st.plotly_chart(plot_dist('P_mgkg', 25, '#ff7f0e'), use_container_width=True)
    with tab3: st.plotly_chart(plot_dist('K_mgkg', 80, '#2ca02c'), use_container_width=True)
