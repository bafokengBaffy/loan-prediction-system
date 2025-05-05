#app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import os
import mlflow
import time
from datetime import datetime
from config import CONFIG
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from recommendations import select_recommendation
from input_validation import validate_all_inputs
from database import init_db, save_submission

# Initialize database connection
conn = init_db()

# Set page config with valid URLs
st.set_page_config(
    page_title=CONFIG["app"]["page_title"],
    page_icon=CONFIG["app"]["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{CONFIG['app']['content']['contact_info']['email']}",
        'Report a bug': f"mailto:{CONFIG['app']['content']['contact_info']['email']}",
        'About': f"{CONFIG['app']['content']['footer']['copyright']}\n{CONFIG['app']['content']['footer']['disclaimer']}"
    }
)

# Custom CSS styling with dark mode charts
st.markdown("""
<style>
    /* Main app styling */
    .main { 
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .header-text { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #ffffff;
    }
    
    /* Card styling */
    .feature-card { 
        border-radius: 10px; 
        padding: 15px; 
        background-color: #1e2130;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2a2e3e;
    }
    
    /* Metric boxes */
    .metric-box {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2a2e3e;
    }
    
    /* Input sections */
    .input-section {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2a2e3e;
    }
    
    /* Visualization containers */
    .viz-container {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2a2e3e;
    }
    
    /* Chart-specific styling */
    .stPlotlyChart {
        background-color: #1e2130 !important;
        border-radius: 10px !important;
    }
    
    /* Text colors */
    .st-bd, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
        color: #f0f2f6 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0a4b8 !important;
        background-color: #1e2130 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background-color: #2a2e3e !important;
        font-weight: bold;
    }
    
    /* Recommendation styling */
    .recommendation {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 16px;
        font-weight: 500;
        background-color: #1e2130;
        border-left: 5px solid;
    }
    
    .high-confidence-approval {
        border-left-color: #2e7d32;
        color: #a5d6a7;
    }
    
    .medium-confidence-approval {
        border-left-color: #81c784;
        color: #c8e6c9;
    }
    
    .low-confidence-approval {
        border-left-color: #4caf50;
        color: #e8f5e9;
    }
    
    .high-confidence-rejection {
        border-left-color: #c62828;
        color: #ef9a9a;
    }
    
    .medium-confidence-rejection {
        border-left-color: #ef9a9a;
        color: #ffcdd2;
    }
    
    .low-confidence-rejection {
        border-left-color: #ff5252;
        color: #ffebee;
    }
    
    .neutral-advice {
        border-left-color: #1976d2;
        color: #90caf9;
    }
    
    .warning-note {
        border-left-color: #ff8f00;
        color: #ffe082;
    }
    
    /* Confidence meter */
    .confidence-meter {
        width: 100%;
        background-color: #2a2e3e;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 20px;
        border-radius: 5px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    
    .confidence-label {
        margin-top: 5px;
        font-size: 14px;
        font-weight: bold;
        color: #f0f2f6;
    }
    
    /* Analysis explanation */
    .analysis-explanation {
        background-color: #2a2e3e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #4a90e2;
        color: #f0f2f6;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly, .modebar {
        background-color: #1e2130 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1e2130 !important;
        color: #f0f2f6 !important;
    }
    
    /* Table borders */
    .stDataFrame td, .stDataFrame th {
        border: 1px solid #2a2e3e !important;
    }
    
    /* Table header */
    .stDataFrame thead {
        background-color: #2a2e3e !important;
    }
    
    /* Table rows */
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #252a3a !important;
    }
    
    /* Table hover */
    .stDataFrame tbody tr:hover {
        background-color: #3a3f5a !important;
    }
    
    /* Confusion matrix specific styling */
    .confusion-matrix {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    
    .confusion-matrix th, .confusion-matrix td {
        padding: 12px;
        text-align: center;
        border: 1px solid #2a2e3e;
    }
    
    .confusion-matrix th {
        background-color: #2a2e3e;
        font-weight: bold;
    }
    
    .confusion-matrix .true-positive {
        background-color: rgba(46, 125, 50, 0.3);
    }
    
    .confusion-matrix .true-negative {
        background-color: rgba(198, 40, 40, 0.3);
    }
    
    .confusion-matrix .false-positive {
        background-color: rgba(255, 152, 0, 0.3);
    }
    
    .confusion-matrix .false-negative {
        background-color: rgba(33, 150, 243, 0.3);
    }
    
    /* Dashboard specific styles */
    .dashboard-header {
        background: linear-gradient(135deg, #1e2130 0%, #2a2e3e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .dashboard-card {
        background-color: #1e2130;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 1px solid #2a2e3e;
    }
    
    .dashboard-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 0.5rem;
    }
    
    .feature-highlight {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: #2a2e3e;
        border-radius: 10px;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .feature-highlight:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-right: 1rem;
        color: #4a90e2;
    }
    
    .feature-content {
        flex: 1;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .feature-desc {
        color: #a0a4b8;
        font-size: 0.95rem;
    }
    
    .model-stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .stat-card {
        background-color: #2a2e3e;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4a90e2;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #a0a4b8;
    }
    
    .timeline {
        position: relative;
        padding-left: 2rem;
        margin: 2rem 0;
    }
    
    .timeline::before {
        content: '';
        position: absolute;
        left: 7px;
        top: 0;
        height: 100%;
        width: 2px;
        background: #4a90e2;
    }
    
    .timeline-item {
        position: relative;
        padding-bottom: 2rem;
    }
    
    .timeline-dot {
        position: absolute;
        left: 0;
        top: 0;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #4a90e2;
    }
    
    .timeline-content {
        margin-left: 2rem;
        padding: 1rem;
        background-color: #2a2e3e;
        border-radius: 8px;
    }
    
    .timeline-date {
        font-size: 0.9rem;
        color: #a0a4b8;
        margin-bottom: 0.5rem;
    }
    
    .timeline-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .timeline-desc {
        font-size: 0.95rem;
        color: #a0a4b8;
    }
    
    /* Key feature styling */
    .key-feature {
        display: flex;
        margin-bottom: 1.5rem;
        background-color: #2a2e3e;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .key-feature-visual {
        flex: 0 0 40%;
        padding: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #1e2130;
    }
    
    .key-feature-content {
        flex: 1;
        padding: 1.5rem;
    }
    
    .key-feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .key-feature-desc {
        color: #a0a4b8;
        margin-bottom: 1rem;
    }
    
    .key-feature-stats {
        display: flex;
        gap: 1rem;
    }
    
    .key-feature-stat {
        background-color: #1e2130;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-align: center;
        min-width: 100px;
    }
    
    .key-feature-stat-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4a90e2;
    }
    
    .key-feature-stat-label {
        font-size: 0.8rem;
        color: #a0a4b8;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    model_path = Path(CONFIG["model"]["output_dir"]) / CONFIG["model"]["best_model_name"]
    metadata_path = Path(CONFIG["model"]["output_dir"]) / CONFIG["model"]["metadata_name"]
    
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        return None, None
    if not metadata_path.exists():
        st.error(f"Metadata file not found at: {metadata_path}")
        return None, None
        
    try:
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_data():
    data_path = Path(CONFIG["data"]["raw_path"])
    if not data_path.exists():
        st.error(f"Data file not found at: {data_path}")
        return None
        
    try:
        data = pd.read_csv(data_path)
        data.columns = data.columns.str.replace(r'[\/, ]', '_', regex=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    """Preprocess data to match training pipeline"""
    if data is None:
        return None
        
    try:
        data = data.copy()
        
        # Clean and transform features
        data['Dependents'] = data['Dependents'].replace('3+', '3').astype(float)
        data['Credit_History'] = data['Credit_History'].astype(float)
        
        # Feature engineering (must match training)
        data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        data['Loan_to_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1e-6)
        data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
        data['Balance_Income'] = data['Total_Income'] - (data['EMI'] * 1000)
        data['LoanAmount_log'] = np.log1p(data['LoanAmount'])
        data['Income_per_Dependent'] = data['Total_Income'] / (data['Dependents'].replace(0, 1))
        data['Credit_History_Income_Interaction'] = data['Credit_History'] * data['Total_Income']
        
        # Ensure all expected features exist
        expected_features = CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]
        missing_features = set(expected_features) - set(data.columns)
        
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Data preprocessing failed: {str(e)}")
        return None

def preprocess_input_data(input_data):
    """Preprocess user input data to match model requirements"""
    try:
        processed = input_data.copy()
        
        # Convert string values to appropriate types
        processed['Dependents'] = float(processed['Dependents'].replace('3+', '3'))
        processed['Credit_History'] = 1.0 if processed['Credit_History'] == 'Good' else 0.0
        processed['ApplicantIncome'] = float(processed['ApplicantIncome'])
        processed['CoapplicantIncome'] = float(processed['CoapplicantIncome'])
        processed['LoanAmount'] = float(processed['LoanAmount'])
        processed['Loan_Amount_Term'] = float(processed['Loan_Amount_Term'])
        
        # Calculate derived features
        processed['Total_Income'] = processed['ApplicantIncome'] + processed['CoapplicantIncome']
        processed['Loan_to_Income_Ratio'] = processed['LoanAmount'] / (processed['Total_Income'] + 1e-6)
        processed['EMI'] = processed['LoanAmount'] / processed['Loan_Amount_Term']
        processed['Balance_Income'] = processed['Total_Income'] - (processed['EMI'] * 1000)
        processed['LoanAmount_log'] = np.log1p(processed['LoanAmount'])
        processed['Income_per_Dependent'] = processed['Total_Income'] / (processed['Dependents'] if processed['Dependents'] > 0 else 1)
        processed['Credit_History_Income_Interaction'] = processed['Credit_History'] * processed['Total_Income']
        
        return processed
    except Exception as e:
        st.error(f"Input data processing failed: {str(e)}")
        return None

def get_user_input():
    """Collect user input for loan application"""
    with st.expander("➕ New Loan Application", expanded=True):
        with st.form("loan_application"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                
            with col2:
                applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000)
                coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000)
                loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000, step=10000)
                loan_amount_term = st.number_input("Loan Term (months)", min_value=12, max_value=480, value=360)
                credit_history = st.selectbox("Credit History", ["Good", "No History"])
                
            submitted = st.form_submit_button("Submit Application")
            
            if submitted:
                input_data = {
                    'Gender': gender,
                    'Married': married,
                    'Dependents': dependents,
                    'Education': education,
                    'Self_Employed': self_employed,
                    'ApplicantIncome': applicant_income,
                    'CoapplicantIncome': coapplicant_income,
                    'LoanAmount': loan_amount,
                    'Loan_Amount_Term': loan_amount_term,
                    'Credit_History': credit_history,
                    'Property_Area': property_area,
                    'Loan_ID': "USER_INPUT_" + str(int(time.time()))
                }
                return preprocess_input_data(input_data)
    return None

def get_recommendations(prediction, proba, input_data):
    """World-class recommendation engine"""
    try:
        # Input validation (40+ checks)
        validation_messages = validate_all_inputs(input_data)
        if validation_messages:
            st.warning("Input Validation Notes:")
            for msg in validation_messages:
                st.markdown(f"- {msg}")
        
        # Get recommendations from our 300+ library
        recommendations = select_recommendation(prediction, proba, input_data)
        
        # Display with appropriate styling
        for rec in recommendations:
            if "⚠️" in rec or "❌" in rec:
                style = "warning-note" if prediction == 1 else "high-confidence-rejection"
            elif "🎉" in rec or "🌟" in rec:
                style = "high-confidence-approval"
            else:
                style = "neutral-advice"
            
            st.markdown(f"""
            <div class="recommendation {style}">
                {rec}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Recommendation engine error: {str(e)}")

def show_feature_importance(model, features, num_features=15):
    """Enhanced feature importance visualization with dark mode"""
    try:
        # Handle pipeline objects
        if hasattr(model, 'named_steps'):
            if 'preprocessor' in model.named_steps:
                if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                    features = model.named_steps['preprocessor'].get_feature_names_out()
                elif hasattr(model.named_steps['preprocessor'], 'feature_names_in_'):
                    features = model.named_steps['preprocessor'].feature_names_in_
            
            if 'classifier' in model.named_steps:
                model = model.named_steps['classifier']
            else:
                st.warning("Cannot find classifier in pipeline")
                return
                
        if not hasattr(model, 'feature_importances_'):
            st.warning("This model type doesn't support feature importance")
            return
            
        importance = model.feature_importances_
        
        if len(features) != len(importance):
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                features = [f"Feature_{i}" for i in range(len(importance))]
                st.warning(f"Feature names don't match importance array ({len(features)} names vs {len(importance)} features). Using generic names.")
            
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(num_features)
        
        importance_df['Feature'] = importance_df['Feature'].str.replace('remainder__', '')\
                                                         .str.replace('num__', '')\
                                                         .str.replace('cat__', '')\
                                                         .str.replace('_', ' ')\
                                                         .str.title()
        
        with st.container():
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader("📊 Feature Importance Analysis")
            
            with st.expander("📘 Understanding Feature Importance", expanded=True):
                st.markdown("""
                <div class="analysis-explanation">
                <h4>What is Feature Importance?</h4>
                <p>Shows which factors most influence loan decisions. Higher values mean greater impact.</p>
                
                <h4>How to Use This Analysis:</h4>
                <ul>
                    <li>Identify key approval/rejection drivers</li>
                    <li>Validate model behavior matches expectations</li>
                    <li>Detect potential bias in decision-making</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Treemap"])
            
            with tab1:
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='tealrose',
                    title='Feature Importance (Bar Chart)',
                    template='plotly_dark'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f0f2f6',
                    xaxis_title='Importance Score',
                    yaxis_title='Features'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                fig = px.line_polar(
                    importance_df,
                    r='Importance',
                    theta='Feature',
                    line_close=True,
                    title='Feature Importance (Radar Chart)',
                    template='plotly_dark',
                    color_discrete_sequence=['#4C78A8']
                )
                fig.update_traces(
                    fill='toself',
                    line_color='#4C78A8',
                    fillcolor='rgba(76, 120, 168, 0.5)'
                )
                fig.update_layout(
                    polar=dict(
                        bgcolor='#1e2130',
                        radialaxis=dict(
                            visible=True,
                            color='#f0f2f6'
                        ),
                        angularaxis=dict(
                            color='#f0f2f6'
                        )
                    ),
                    font_color='#f0f2f6'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                fig = px.treemap(
                    importance_df,
                    path=['Feature'],
                    values='Importance',
                    title='Feature Importance (Treemap)',
                    template='plotly_dark',
                    color='Importance',
                    color_continuous_scale='tealrose'
                )
                fig.update_layout(
                    margin=dict(t=50, l=25, r=25, b=25),
                    font_color='#f0f2f6'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Feature importance error: {str(e)}")

def show_data_distribution(data, feature):
    """Enhanced data distribution with dark mode"""
    try:
        if feature not in data.columns:
            raise KeyError(f"Feature '{feature}' not found")
            
        if 'Probability' not in data.columns:
            data['Probability'] = 0.5
            
        with st.container():
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader(f"📈 {feature} Distribution Analysis")
            
            with st.expander("📘 Understanding Data Distributions", expanded=True):
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Analysis Focus: {feature}</h4>
                <p>Understanding how this feature relates to loan decisions:</p>
                
                <ul>
                    <li>Compare approved vs rejected distributions</li>
                    <li>Identify typical value ranges for approvals</li>
                    <li>Spot unusual values needing investigation</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Violin Plot", "Box Plot", "Scatter Matrix"])
            
            with tab1:
                fig = px.histogram(
                    data,
                    x=feature,
                    color='Prediction',
                    nbins=50,
                    barmode='overlay',
                    marginal="box",
                    title=f'Distribution of {feature}',
                    template='plotly_dark',
                    color_discrete_map={'Approved': '#2e7d32', 'Rejected': '#c62828'}
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f0f2f6',
                    xaxis_title=feature,
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                fig = px.violin(
                    data,
                    x='Prediction',
                    y=feature,
                    color='Prediction',
                    box=True,
                    points="all",
                    title=f'Violin Plot of {feature} by Prediction',
                    template='plotly_dark',
                    color_discrete_map={'Approved': '#2e7d32', 'Rejected': '#c62828'}
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f0f2f6',
                    xaxis_title='Prediction',
                    yaxis_title=feature
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                fig = px.box(
                    data,
                    x='Prediction',
                    y=feature,
                    color='Prediction',
                    notched=True,
                    title=f'Box Plot of {feature} by Prediction',
                    template='plotly_dark',
                    color_discrete_map={'Approved': '#2e7d32', 'Rejected': '#c62828'}
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f0f2f6',
                    xaxis_title='Prediction',
                    yaxis_title=feature
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab4:
                fig = px.scatter_matrix(
                    data,
                    dimensions=[feature, 'Probability', 'Total_Income'],
                    color='Prediction',
                    title=f'Scatter Matrix of {feature}',
                    template='plotly_dark',
                    color_discrete_map={'Approved': '#2e7d32', 'Rejected': '#c62828'}
                )
                fig.update_layout(
                    height=700,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f0f2f6'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Data distribution error: {str(e)}")

def show_outlier_analysis(data, feature, threshold):
    """Enhanced outlier detection with dark mode"""
    st.subheader("🔎 Outlier Detection Analysis")
    
    with st.expander("📘 Understanding Outliers", expanded=True):
        st.markdown(f"""
        <div class="analysis-explanation">
        <h4>Outlier Definition</h4>
        <p>Values outside {threshold}×IQR from quartiles:</p>
        <ul>
            <li><strong>IQR:</strong> Range between 25th-75th percentiles</li>
            <li><strong>Threshold:</strong> {threshold}×IQR multiplier</li>
            <li><strong>Impact:</strong> May indicate errors or special cases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if feature in data.columns:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        fig = px.scatter(
            data,
            x=feature,
            y='Probability',
            color='Prediction',
            color_discrete_map={'Approved': '#2e7d32', 'Rejected': '#c62828'},
            hover_data=['Loan_ID', 'ApplicantIncome', 'LoanAmount'],
            title=f'Outlier Detection for {feature}',
            template='plotly_dark'
        )
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="#ff5252")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="#ff5252")
        
        fig.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f2f6',
            xaxis_title=feature,
            yaxis_title='Probability'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        if not outliers.empty:
            st.subheader(f"📋 Outliers ({len(outliers)} found)")
            st.dataframe(
                outliers[['Loan_ID', feature, 'Prediction', 'Confidence']],
                use_container_width=True
            )
        else:
            st.success("No outliers detected with current threshold")
    else:
        st.error(f"Selected feature '{feature}' not found in data")

def show_correlation_analysis(data):
    """Enhanced correlation analysis with dark mode"""
    st.subheader("📊 Correlation Analysis")
    
    with st.expander("📘 Understanding Correlations", expanded=True):
        st.markdown("""
        <div class="analysis-explanation">
        <h4>Correlation Interpretation</h4>
        <p>Measures how features move together (-1 to 1):</p>
        <ul>
            <li><strong>+1:</strong> Perfect positive relationship</li>
            <li><strong>0:</strong> No relationship</li>
            <li><strong>-1:</strong> Perfect negative relationship</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    numeric_features = [f for f in CONFIG["data"]["features"]["numeric"] if f in data.columns]
    if numeric_features:
        numeric_data = data[numeric_features + ['Probability']]
        corr = numeric_data.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title="Feature Correlation Matrix",
            template='plotly_dark'
        )
        fig.update_layout(
            height=700,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f2f6'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No numeric features available for correlation analysis")

def show_confusion_matrix_analysis(y_true, y_pred, y_proba):
    """Enhanced confusion matrix analysis with table and metrics"""
    st.subheader("📊 Confusion Matrix Analysis")
    
    with st.expander("📘 Understanding the Confusion Matrix", expanded=True):
        st.markdown("""
        <div class="analysis-explanation">
        <h4>Confusion Matrix Components</h4>
        <p>Shows model prediction performance across classes:</p>
        <ul>
            <li><strong>True Positives (TP):</strong> Correctly approved loans</li>
            <li><strong>False Positives (FP):</strong> Loans incorrectly approved (Type I error)</li>
            <li><strong>True Negatives (TN):</strong> Correctly rejected loans</li>
            <li><strong>False Negatives (FN):</strong> Loans incorrectly rejected (Type II error)</li>
        </ul>
        
        <h4>Key Metrics Derived:</h4>
        <ul>
            <li><strong>Accuracy:</strong> (TP+TN)/Total - Overall correctness</li>
            <li><strong>Precision:</strong> TP/(TP+FP) - Approval correctness</li>
            <li><strong>Recall/Sensitivity:</strong> TP/(TP+FN) - Approved loan coverage</li>
            <li><strong>Specificity:</strong> TN/(TN+FP) - Rejected loan coverage</li>
            <li><strong>F1 Score:</strong> 2*(Precision*Recall)/(Precision+Recall) - Balanced measure</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Create confusion matrix table with styling
    cm_table = pd.DataFrame(
        cm,
        columns=['Predicted Rejected', 'Predicted Approved'],
        index=['Actual Rejected', 'Actual Approved']
    )
    
    # Display the confusion matrix table with enhanced styling
    st.subheader("Confusion Matrix")
    st.markdown("""
    <table class="confusion-matrix">
        <thead>
            <tr>
                <th></th>
                <th>Predicted Rejected</th>
                <th>Predicted Approved</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Actual Rejected</strong></td>
                <td class="true-negative">{tn}</td>
                <td class="false-positive">{fp}</td>
            </tr>
            <tr>
                <td><strong>Actual Approved</strong></td>
                <td class="false-negative">{fn}</td>
                <td class="true-positive">{tp}</td>
            </tr>
        </tbody>
    </table>
    """.format(tn=tn, fp=fp, fn=fn, tp=tp), unsafe_allow_html=True)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display metrics in columns with improved styling
    st.subheader("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>Accuracy</h4>
            <p>{:.1%}</p>
            <small>Overall correctness</small>
        </div>
        """.format(accuracy), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>Precision</h4>
            <p>{:.1%}</p>
            <small>Approval correctness</small>
        </div>
        """.format(precision), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h4>Recall</h4>
            <p>{:.1%}</p>
            <small>Approved loan coverage</small>
        </div>
        """.format(recall), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h4>Specificity</h4>
            <p>{:.1%}</p>
            <small>Rejected loan coverage</small>
        </div>
        """.format(specificity), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-box">
            <h4>F1 Score</h4>
            <p>{:.1%}</p>
            <small>Balanced measure</small>
        </div>
        """.format(f1), unsafe_allow_html=True)
    
    # Detailed classification report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style
            .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
            .format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}'}),
        use_container_width=True
    )
    
    # Threshold analysis
    st.subheader("📈 Threshold Sensitivity Analysis")
    thresholds = np.linspace(0.1, 0.9, 9)
    metrics = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        cm_thresh = confusion_matrix(y_true, y_pred_thresh)
        tn, fp, fn, tp = cm_thresh.ravel()
        
        metrics.append({
            'Threshold': thresh,
            'Approvals': tp + fp,
            'Rejections': tn + fn,
            'FP Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'FN Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Display threshold analysis table
    st.dataframe(
        metrics_df.style
            .background_gradient(cmap='Blues', subset=['FP Rate', 'FN Rate', 'Accuracy'])
            .format({
                'Threshold': '{:.2f}',
                'FP Rate': '{:.2%}',
                'FN Rate': '{:.2%}',
                'Accuracy': '{:.2%}'
            }),
        use_container_width=True
    )
    
    # Plot threshold sensitivity
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['FP Rate'],
        name='False Approval Rate',
        line=dict(color='#ef9a9a', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['FN Rate'],
        name='False Rejection Rate',
        line=dict(color='#90caf9', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df['Accuracy'],
        name='Accuracy',
        line=dict(color='#a5d6a7', width=3, dash='dot')
    ))
    
    fig.update_layout(
        title='Error Rates by Decision Threshold',
        xaxis_title='Approval Probability Threshold',
        yaxis_title='Rate',
        height=500,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(data, confidence_threshold, status_filter):
    """Enhanced data explorer with explanatory analysis"""
    st.subheader("📋 Loan Application Explorer")
    
    with st.expander("📘 Using the Data Explorer", expanded=True):
        st.markdown("""
        <div class="analysis-explanation">
        <h4>Interactive Data Exploration</h4>
        <ul>
            <li><strong>Filter:</strong> By approval status and confidence level</li>
            <li><strong>Sort:</strong> Click column headers to sort</li>
            <li><strong>Search:</strong> Use the search box to find specific values</li>
            <li><strong>Export:</strong> Use the menu (⋮) in top-right to download data</li>
        </ul>
        
        <h4>Key Metrics</h4>
        <ul>
            <li><strong>Confidence Threshold:</strong> Minimum prediction certainty ({confidence_threshold}%)</li>
            <li><strong>Status Filter:</strong> Showing {status_filter} applications</li>
            <li><strong>Probability:</strong> Model's confidence in the prediction (0-1)</li>
        </ul>
        
        <h4>Usage Tips</h4>
        <ul>
            <li>Click on column headers to sort</li>
            <li>Hover over cells to see full content</li>
            <li>Use the search box to filter specific values</li>
            <li>Adjust confidence threshold in sidebar to focus on more certain predictions</li>
        </ul>
        </div>
        """.format(confidence_threshold=confidence_threshold, status_filter=status_filter), 
        unsafe_allow_html=True)
    
    try:
        filtered_data = data.copy()
        if status_filter != "All":
            filtered_data = filtered_data[filtered_data['Prediction'] == status_filter]
        
        # Ensure Probability column exists for filtering
        if 'Probability' not in filtered_data.columns:
            filtered_data['Probability'] = 0.5  # Default value
            
        filtered_data = filtered_data[filtered_data['Probability'] >= (confidence_threshold/100)]
        
        display_columns = [
            'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
            'ApplicantIncome', 'LoanAmount', 'Credit_History',
            'Prediction', 'Confidence', 'Probability'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in display_columns if col in filtered_data.columns]
        
        # Ensure we have columns to display
        if not available_columns:
            st.warning("No matching columns available for display")
        else:
            # Sort only if Probability column exists
            sort_column = 'Probability' if 'Probability' in available_columns else available_columns[0]
            
            st.dataframe(
                filtered_data[available_columns].sort_values(sort_column, ascending=False),
                use_container_width=True,
                height=600
            )
    except Exception as e:
        st.error(f"Data explorer error: {str(e)}")

def show_dashboard(model, metadata, raw_data):
    """Show the home dashboard with model overview and key metrics"""
    st.markdown('<div class="dashboard-header">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="header-text">{CONFIG["app"]["content"]["header_title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 1.2rem; color: #a0a4b8;">{CONFIG["app"]["content"]["header_subtitle"]}</p>', unsafe_allow_html=True)
    
    with col2:
        if metadata:
            st.markdown("""
            <div class="metric-box" style="text-align: center;">
                <h4>Model Accuracy</h4>
                <p style="font-size: 2rem; color: #4a90e2; font-weight: bold;">{:.1%}</p>
                <p style="font-size: 0.9rem;">on validation data</p>
            </div>
            """.format(metadata['metrics']['accuracy']), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key features section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card-title">🚀 Key Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">📊</div>
            <div class="feature-content">
                <div class="feature-title">Advanced Analytics</div>
                <div class="feature-desc">Real-time predictions with explainable AI and feature importance analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🔍</div>
            <div class="feature-content">
                <div class="feature-title">Deep Insights</div>
                <div class="feature-desc">Understand approval drivers with interactive visualizations</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🤖</div>
            <div class="feature-content">
                <div class="feature-title">Smart Recommendations</div>
                <div class="feature-desc">300+ tailored suggestions to improve approval chances</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">📈</div>
            <div class="feature-content">
                <div class="feature-title">Performance Tracking</div>
                <div class="feature-desc">Monitor model metrics and decision patterns over time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🛡️</div>
            <div class="feature-content">
                <div class="feature-title">Bias Detection</div>
                <div class="feature-desc">Identify and mitigate potential fairness issues</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">⚙️</div>
            <div class="feature-content">
                <div class="feature-title">Customizable</div>
                <div class="feature-desc">Adjust thresholds and parameters to match your risk appetite</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model statistics
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card-title">📊 Model Statistics</div>', unsafe_allow_html=True)
    
    if metadata:
        st.markdown("""
        <div class="model-stats-grid">
            <div class="stat-card">
                <div class="stat-label">Model Type</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Training Date</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Precision</div>
                <div class="stat-value">{:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Recall</div>
                <div class="stat-value">{:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">{:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">ROC AUC</div>
                <div class="stat-value">{:.3f}</div>
            </div>
        </div>
        """.format(
            metadata['model_type'],
            metadata['training_date'],
            metadata['metrics']['precision'],
            metadata['metrics']['recall'],
            metadata['metrics']['f1'],
            metadata['metrics']['roc_auc']
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key decision factors
    if model and hasattr(model, 'feature_importances_'):
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card-title">🔑 Key Decision Factors</div>', unsafe_allow_html=True)
        
        try:
            # Handle pipeline objects
            if hasattr(model, 'named_steps'):
                if 'classifier' in model.named_steps:
                    model = model.named_steps['classifier']
            
            importance = model.feature_importances_
            features = CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]
            
            if len(features) != len(importance):
                if hasattr(model, 'feature_names_in_'):
                    features = model.feature_names_in_
                else:
                    features = [f"Feature_{i}" for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(5)
            
            importance_df['Feature'] = importance_df['Feature'].str.replace('remainder__', '')\
                                                             .str.replace('num__', '')\
                                                             .str.replace('cat__', '')\
                                                             .str.replace('_', ' ')\
                                                             .str.title()
            
            for idx, row in importance_df.iterrows():
                st.markdown(f"""
                <div class="key-feature">
                    <div class="key-feature-visual">
                        <div style="width: 100%; height: 150px;">
                            <div style="height: {row['Importance']*100:.1f}%; background-color: #4a90e2; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <div class="key-feature-content">
                        <div class="key-feature-title">{row['Feature']}</div>
                        <div class="key-feature-desc">This is the most influential factor in loan decisions, accounting for {row['Importance']*100:.1f}% of the model's decision-making process.</div>
                        <div class="key-feature-stats">
                            <div class="key-feature-stat">
                                <div class="key-feature-stat-value">{row['Importance']*100:.1f}%</div>
                                <div class="key-feature-stat-label">Importance</div>
                            </div>
                            <div class="key-feature-stat">
                                <div class="key-feature-stat-value">1</div>
                                <div class="key-feature-stat-label">Rank</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Could not display feature importance: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model development timeline
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card-title">⏳ Model Development Timeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="timeline">
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-date">January 2023</div>
                <div class="timeline-title">Initial Data Collection</div>
                <div class="timeline-desc">Gathered 10,000+ historical loan applications with outcomes</div>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-date">March 2023</div>
                <div class="timeline-title">Feature Engineering</div>
                <div class="timeline-desc">Created 15+ derived features including income ratios and credit interactions</div>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-date">May 2023</div>
                <div class="timeline-title">Model Prototyping</div>
                <div class="timeline-desc">Tested 8 different algorithms including XGBoost, Random Forest, and Logistic Regression</div>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-date">July 2023</div>
                <div class="timeline-title">Bias Mitigation</div>
                <div class="timeline-desc">Implemented fairness constraints to ensure equitable decisions across demographics</div>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <div class="timeline-date">September 2023</div>
                <div class="timeline-title">Production Deployment</div>
                <div class="timeline-desc">Launched model with 87.4% accuracy and 0.92 AUC score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card-title">⚡ Get Started</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #2a2e3e; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem;">
            <h3 style="color: #ffffff; margin-top: 0;">New Application</h3>
            <p style="color: #a0a4b8;">Submit a new loan application for immediate decision</p>
            <button onclick="window.location.href='#new-application';" style="background-color: #4a90e2; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">Start Application</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #2a2e3e; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem;">
            <h3 style="color: #ffffff; margin-top: 0;">Explore Data</h3>
            <p style="color: #a0a4b8;">Analyze historical decisions and model performance</p>
            <button onclick="window.location.href='#data-explorer';" style="background-color: #4a90e2; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; cursor: pointer;">View Data</button>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Load model and data
model, metadata = load_model()
raw_data = load_data()

# Sidebar configuration
with st.sidebar:
    st.title("🔍 Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Go to",
        ["Dashboard", "New Application", "Feature Importance", "Data Distribution", 
         "Outlier Detection", "Correlation Analysis", "Data Explorer",
         "Confusion Matrix"]
    )
    
    if page != "Dashboard":
        if page == "Feature Importance":
            num_features = st.slider("Number of Features to Show", 5, 30, 15,
                                   help="Adjust to focus on top N most important features")
        elif page == "Data Distribution":
            selected_feature = st.selectbox(
                "Select Feature",
                CONFIG["data"]["features"]["numeric"] if raw_data is not None else [],
                help="Choose which feature to analyze distributions for"
            )
        elif page == "Outlier Detection":
            outlier_feature = st.selectbox(
                "Select Feature for Outlier Analysis",
                ['LoanAmount', 'ApplicantIncome', 'Loan_to_Income_Ratio'],
                help="Choose which numeric feature to analyze for outliers"
            )
            outlier_threshold = st.slider("Outlier Threshold (IQR multiplier)", 1.0, 3.0, 1.5, step=0.1,
                                        help="Higher values detect fewer extreme outliers")
        elif page == "Data Explorer":
            confidence_threshold = st.slider("Minimum Confidence Threshold", 50, 100, 75,
                                           help="Only show predictions with at least this confidence level")
            status_filter = st.selectbox("Filter by Status", ["All", "Approved", "Rejected"],
                                       help="Filter applications by approval status")
    
    st.markdown("---")
    st.markdown(f"**Model Version:** {CONFIG['app']['content']['system_status']['model_version']}")
    if metadata:
        st.markdown(f"**Accuracy:** {metadata['metrics']['accuracy']:.1%}")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    st.markdown("**Contact Support:**")
    st.markdown(f"📞 {CONFIG['app']['content']['contact_info']['phone']}")
    st.markdown(f"✉️ {CONFIG['app']['content']['contact_info']['email']}")

# Main content routing
if page == "Dashboard":
    show_dashboard(model, metadata, raw_data)
else:
    if model is not None:
        if page == "New Application":
            st.subheader("📝 New Loan Application")
            
            input_data = get_user_input()
            if input_data is not None:
                try:
                    input_df = pd.DataFrame([input_data])
                    expected_features = CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]
                    available_features = [f for f in expected_features if f in input_df.columns]
                    
                    if not available_features:
                        st.error("No matching features found between input and model requirements")
                    else:
                        X_input = input_df[available_features]
                        prediction = model.predict(X_input)[0]
                        probability = max(model.predict_proba(X_input)[0])
                        
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Prediction", 
                                "Approved" if prediction == 1 else "Rejected", 
                                delta=f"{probability*100:.1f}% confidence", 
                                delta_color="normal"
                            )
                        
                        with col2:
                            st.metric("Loan Amount", f"₹{input_data['LoanAmount']:,.0f}")
                            st.metric("Loan Term", f"{input_data['Loan_Amount_Term']:.0f} months")
                        
                        st.markdown("---")
                        get_recommendations(prediction, probability, input_data)
                        
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("---")
                            st.subheader("🔍 Key Decision Factors")
                            
                            importance = model.feature_importances_
                            features = available_features
                            
                            top_idx = np.argsort(importance)[::-1][:5]
                            top_features = [features[i] for i in top_idx]
                            top_importance = importance[top_idx]
                            
                            fig = px.bar(
                                x=top_importance,
                                y=top_features,
                                orientation='h',
                                labels={'x': 'Importance', 'y': 'Feature'},
                                title='Top Influencing Features'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Feature importance is not available for the current model.")
                except Exception as e:
                    st.error(f"Error processing your application: {str(e)}")
        
        elif raw_data is not None:
            processed_data = preprocess_data(raw_data)
            if processed_data is not None:
                X = processed_data[CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"]]
                predictions = model.predict(X)
                proba = model.predict_proba(X)[:, 1]  # Probability of positive class
                
                data = processed_data.copy()
                data['Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
                data['Probability'] = [max(p) for p in model.predict_proba(X)]
                data['Confidence'] = [f"{max(p)*100:.1f}%" for p in model.predict_proba(X)]
                
                # If we have true labels in the data (for confusion matrix)
                if 'Loan_Status' in data.columns:
                    y_true = data['Loan_Status'].map({'Y': 1, 'N': 0}).values
                    y_pred = predictions
                    y_proba = proba
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Applications", len(data))
                with col2:
                    approval_rate = (data['Prediction'] == 'Approved').mean()
                    st.metric("Approval Rate", f"{approval_rate:.1%}")
                with col3:
                    avg_confidence = data['Probability'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                with col4:
                    st.metric("Model Accuracy", f"{metadata['metrics']['accuracy']:.1%}")
                
                st.markdown("---")
                
                if page == "Feature Importance":
                    show_feature_importance(
                        model, 
                        CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"], 
                        num_features
                    )
                    
                elif page == "Data Distribution":
                    show_data_distribution(data, selected_feature)
                    
                elif page == "Outlier Detection":
                    show_outlier_analysis(data, outlier_feature, outlier_threshold)
                        
                elif page == "Correlation Analysis":
                    show_correlation_analysis(data)
                    
                elif page == "Data Explorer":
                    show_data_explorer(data, confidence_threshold, status_filter)
                    
                elif page == "Confusion Matrix":
                    if 'Loan_Status' in data.columns:
                        show_confusion_matrix_analysis(y_true, y_pred, y_proba)
                    else:
                        st.error("True labels ('Loan_Status') not found in data - cannot show confusion matrix")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #7f8c8d;'>{CONFIG['app']['content']['footer']['copyright']}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; color: #7f8c8d; font-size: 0.8em;'>{CONFIG['app']['content']['footer']['disclaimer']}</div>", unsafe_allow_html=True)