import streamlit as st
from utils import load_data

# Import pages from the folder
from pages_code import page_1_dashboard, page_2_input, page_3_result, page_4_evaluation

# --- CONFIG ---
st.set_page_config(page_title="Palm Nutrient AI", page_icon="🌴", layout="wide")

# --- LOAD DATA GLOBALLY ---
# Updated to handle potential load failures cleanly
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# ==========================================
#      SIDEBAR SETUP (Logo & Title)
# ==========================================
with st.sidebar:
    # 1. Display Logo
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/UniKL_logo.svg/1200px-UniKL_logo.svg.png", use_container_width=True)
    
    # 2. Display Project Title
    st.markdown("""
    <h3 style='text-align: center; color: #333;'>
        Soil Fertility Prediction Using Machine Learning for Palm Tree Nutrient Management
    </h3>
    """, unsafe_allow_html=True)
    
    st.divider() # Adds a visual line separator
    
    # 3. Navigation Menu
    st.title("Navigation")

    # Initialize session state for page selection
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "📊 Dashboard"

    # Function to update page
    def set_page(page_name):
        st.session_state.page_selection = page_name

    # Navigation Buttons
    st.button("📊 Dashboard", on_click=set_page, args=("📊 Dashboard",), use_container_width=True)
    st.button("🧪 Prediction Input", on_click=set_page, args=("🧪 Prediction Input",), use_container_width=True)
    st.button("📋 Prediction Result", on_click=set_page, args=("📋 Prediction Result",), use_container_width=True)
    st.button("📈 Model Evaluation", on_click=set_page, args=("📈 Model Evaluation",), use_container_width=True)
    
    # Add a reset button for debugging data issues
    if st.button("🔄 Reload Data", type="secondary"):
        st.cache_data.clear()
        st.session_state.data = load_data()
        st.rerun()

# ==========================================
#           MAIN PAGE ROUTING
# ==========================================
page = st.session_state.page_selection

if st.session_state.data is not None:
    if page == "📊 Dashboard":
        page_1_dashboard.show(st.session_state.data)
    elif page == "🧪 Prediction Input":
        page_2_input.show(st.session_state.data)
    elif page == "📋 Prediction Result":
        page_3_result.show(st.session_state.data)
    elif page == "📈 Model Evaluation":
        page_4_evaluation.show(st.session_state.data)
else:
    st.error("⚠️ Application halted: Dataset could not be loaded. Please check 'oil_palm_perak.csv' and utils.py configuration.")
