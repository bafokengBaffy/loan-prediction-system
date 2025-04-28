# app.py
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
from recommendations import select_recommendation
from input_validation import validate_all_inputs
from database import init_db, save_submission


# Initialize database connection
conn = init_db()

 #Set page config with valid URLs
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


    """Generate confidence-based recommendations with improved visibility"""
    recommendations = []
    confidence_level = proba * 100
    
    # Confidence visualization
    st.markdown('<div class="recommendation-title">📊 Decision Confidence</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        confidence_color = "#4caf50"  # Green for approval
        confidence_label = "Approval Confidence"
    else:
        confidence_color = "#f44336"  # Red for rejection
        confidence_label = "Rejection Confidence"
    
    st.markdown(f"""
    <div class="confidence-meter">
        <div class="confidence-fill" style="width: {confidence_level}%; background-color: {confidence_color};">
            {confidence_level:.1f}%
        </div>
    </div>
    <div class="confidence-label">
        {confidence_label} • {'High' if confidence_level >= 75 else 'Medium' if confidence_level >= 50 else 'Low'} Confidence
    </div>
    """, unsafe_allow_html=True)
    
    # Convert input values to numeric
    try:
        loan_to_income = float(input_data['Loan_to_Income_Ratio'])
        dependents = float(input_data['Dependents'])
        credit_history = float(input_data['Credit_History'])
        loan_amount = float(input_data['LoanAmount'])
        total_income = float(input_data['Total_Income'])
        loan_term = float(input_data['Loan_Amount_Term'])
    except KeyError as e:
        st.error(f"Missing required data for recommendations: {str(e)}")
        return []
    
    # Prediction-specific recommendations
    st.markdown('<div class="recommendation-title">📌 Actionable Recommendations</div>', unsafe_allow_html=True)
    
    if prediction == 1:  # Approved
        if confidence_level >= 75:
            recommendations.append({
                "type": "high-confidence-approval",
                "message": "🎉 Excellent! Your application is highly likely to be approved with favorable terms."
            })
        elif confidence_level >= 50:
            recommendations.append({
                "type": "medium-confidence-approval",
                "message": "👍 Good news! Your application is likely to be approved, but terms may vary."
            })
        else:
            recommendations.append({
                "type": "low-confidence-approval",
                "message": "🤞 Your application may be approved, but consider these improvements for better terms."
            })
        
        if loan_to_income > 0.4:
            ideal_loan = total_income * 0.35
            recommendations.append({
                "type": "warning-note",
                "message": f"⚠️ Your loan-to-income ratio is {loan_to_income:.2f} (recommended <0.4). Consider reducing loan amount to ₹{ideal_loan:,.0f} for better terms."
            })
            
        if credit_history == 0:
            recommendations.append({
                "type": "warning-note",
                "message": "📈 You might get better interest rates by building credit history with a credit card or smaller loan."
            })
            
        if dependents >= 2:
            recommendations.append({
                "type": "neutral-advice",
                "message": "👨‍👩‍👧‍👦 With multiple dependents, consider opting for loan insurance for added security."
            })
            
    else:  # Rejected
        if confidence_level >= 75:
            recommendations.append({
                "type": "high-confidence-rejection",
                "message": "❌ Strong indicators suggest your application may be declined based on current parameters."
            })
        elif confidence_level >= 50:
            recommendations.append({
                "type": "medium-confidence-rejection",
                "message": "⚠️ Your application currently doesn't meet all approval criteria, but improvements are possible."
            })
        else:
            recommendations.append({
                "type": "low-confidence-rejection",
                "message": "🤔 Your application is borderline. Small improvements could change the outcome."
            })
        
        rejection_factors = []
        
        if credit_history == 0:
            rejection_factors.append("no established credit history")
            recommendations.append({
                "type": "neutral-advice",
                "message": "🔹 Build credit: Apply for a secured credit card or small personal loan (₹10,000-50,000) to establish credit."
            })
            
        if loan_to_income > 0.5:
            rejection_factors.append(f"high debt burden (ratio: {loan_to_income:.2f})")
            ideal_loan = total_income * 0.35
            recommendations.append({
                "type": "neutral-advice",
                "message": f"🔹 Reduce loan amount: Try ₹{ideal_loan:,.0f} or add a co-signer with income ≥₹{total_income*0.3:,.0f}/month."
            })
            
        if dependents > 3:
            rejection_factors.append("high number of dependents")
            recommendations.append({
                "type": "neutral-advice",
                "message": "🔹 Strengthen application: Add a co-borrower with stable income (≥₹25,000/month)."
            })
        
        if loan_term > 300:
            recommendations.append({
                "type": "neutral-advice",
                "message": f"⏱️ Consider shorter term: {int(loan_term/12)} years is long. Try 15-20 years for better approval chances."
            })
        
        if not rejection_factors:
            rejection_factors.append("multiple risk factors")
        
        improvement_timeframe = "3 months" if confidence_level >= 50 else "6 months"
        recommendations.append({
            "type": "neutral-advice",
            "message": f"📅 Reapply in {improvement_timeframe} after improving these factors for better chances."
        })
    
    # Financial health tips (always shown)
    st.markdown('<div class="recommendation-title">💡 Financial Health Tips</div>', unsafe_allow_html=True)
    
    recommendations.append({
        "type": "neutral-advice",
        "message": f"💰 Emergency Fund: Maintain at least 3-6 months of expenses as savings (₹{total_income * 3:,.0f} based on your income)."
    })
    
    if loan_to_income > 0.3:
        recommendations.append({
            "type": "warning-note",
            "message": f"⚖️ Debt Management: Keep total monthly debt payments below 40% of income (₹{total_income * 0.4 / 12:,.0f}/month for you)."
        })
    
    # Display all recommendations
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation {rec['type']}">
            {rec['message']}
        </div>
        """, unsafe_allow_html=True)
    
    return recommendations

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

        
def show_data_distribution(data, feature):
    """Enhanced data distribution with explanatory analysis"""
    try:
        if feature not in data.columns:
            raise KeyError(f"Feature '{feature}' not found")
            
        # Ensure Probability column exists
        if 'Probability' not in data.columns:
            data['Probability'] = 0.5  # Default value if missing
            
        with st.container():
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader(f"📈 {feature} Distribution Analysis")
            
            # Explanatory section
            with st.expander("📘 Understanding Data Distributions", expanded=True):
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Analysis Focus: {feature}</h4>
                <p>Understanding how this feature relates to loan decisions:</p>
                
                <ul>
                    <li><strong>Approved vs Rejected Distributions:</strong> Compare values for approved/rejected applications</li>
                    <li><strong>Typical Ranges:</strong> Identify common value ranges for approvals</li>
                    <li><strong>Outlier Detection:</strong> Spot unusual values that might need investigation</li>
                    <li><strong>Distribution Shape:</strong> Normal, skewed, or bimodal distributions suggest different decision patterns</li>
                </ul>
                
                <p><strong>Key Questions:</strong></p>
                <ul>
                    <li>Do approved loans cluster in specific ranges?</li>
                    <li>Are there clear cutoff points between approvals/rejections?</li>
                    <li>How much overlap exists between approved/rejected distributions?</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Violin Plot", "Box Plot", "Scatter Matrix"])
            
            with tab1:
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Histogram Analysis</h4>
                <p>Shows value distribution frequency:</p>
                <ul>
                    <li><strong>Peaks:</strong> Most common values for approvals/rejections</li>
                    <li><strong>Skewness:</strong> Direction of value concentration (left or right)</li>
                    <li><strong>Gaps:</strong> Uncommon value ranges that might indicate data issues</li>
                    <li><strong>Bimodality:</strong> Two distinct peaks may indicate different applicant groups</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.histogram(
                    data,
                    x=feature,
                    color='Prediction',
                    nbins=50,
                    barmode='overlay',
                    marginal="box",
                    title=f'Distribution of {feature}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Violin Plot Analysis</h4>
                <p>Combines distribution and summary statistics:</p>
                <ul>
                    <li><strong>Width:</strong> Density of values at different points</li>
                    <li><strong>White Dot:</strong> Median value (middle point)</li>
                    <li><strong>Box:</strong> Interquartile range (25th-75th percentiles)</li>
                    <li><strong>Whiskers:</strong> Typical range of values (1.5×IQR)</li>
                </ul>
                <p>Violin plots show the complete distribution shape while box plots emphasize summary statistics.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.violin(
                    data,
                    x='Prediction',
                    y=feature,
                    color='Prediction',
                    box=True,
                    points="all",
                    title=f'Violin Plot of {feature} by Prediction'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Box Plot Analysis</h4>
                <p>Emphasizes statistical measures:</p>
                <ul>
                    <li><strong>Box:</strong> Middle 50% of values (25th-75th percentiles)</li>
                    <li><strong>Whiskers:</strong> Typical range (1.5×IQR from quartiles)</li>
                    <li><strong>Outliers:</strong> Values outside typical range shown as dots</li>
                    <li><strong>Median Line:</strong> Middle value of the distribution</li>
                </ul>
                <p>Box plots help quickly compare distributions between approved and rejected applications.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.box(
                    data,
                    x='Prediction',
                    y=feature,
                    color='Prediction',
                    notched=True,
                    title=f'Box Plot of {feature} by Prediction'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab4:
                st.markdown(f"""
                <div class="analysis-explanation">
                <h4>Scatter Matrix Analysis</h4>
                <p>Reveals relationships between features:</p>
                <ul>
                    <li><strong>Diagonal:</strong> Histograms of individual features</li>
                    <li><strong>Off-diagonal:</strong> Correlation patterns between features</li>
                    <li><strong>Color Coding:</strong> Approval status differentiation</li>
                    <li><strong>Clusters:</strong> Groups of similar applications</li>
                </ul>
                <p>Look for clear separation between approved/rejected points in the scatter plots.</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.scatter_matrix(
                    data,
                    dimensions=[feature, 'Probability', 'Total_Income'],
                    color='Prediction',
                    title=f'Scatter Matrix of {feature}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Data distribution error: {str(e)}")

def show_outlier_analysis(data, feature, threshold):
    """Enhanced outlier detection with explanatory analysis"""
    st.subheader("🔎 Outlier Detection Analysis")
    
    with st.expander("📘 Understanding Outliers", expanded=True):
        st.markdown(f"""
        <div class="analysis-explanation">
        <h4>Outlier Definition</h4>
        <p>Values outside {threshold}×IQR from quartiles:</p>
        <ul>
            <li><strong>IQR:</strong> Range between 25th-75th percentiles (middle 50% of data)</li>
            <li><strong>Threshold:</strong> {threshold}×IQR multiplier (adjustable in sidebar)</li>
            <li><strong>Impact:</strong> May indicate errors or special cases needing review</li>
        </ul>
        
        <h4>Recommendations:</h4>
        <ul>
            <li>Investigate extreme values for data entry errors</li>
            <li>Check if outliers represent valid special cases</li>
            <li>Consider capped processing for extreme values</li>
            <li>Review model sensitivity to outliers</li>
        </ul>
        
        <h4>Common Causes:</h4>
        <ul>
            <li>Data entry errors (extra zeros, misplaced decimals)</li>
            <li>Legitimate extreme cases (very high income, very small loans)</li>
            <li>System processing errors</li>
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
            color_discrete_map={'Approved': '#27ae60', 'Rejected': '#e74c3c'},
            hover_data=['Loan_ID', 'ApplicantIncome', 'LoanAmount'],
            title=f'Outlier Detection for {feature}'
        )
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="red")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="red")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        if not outliers.empty:
            st.subheader(f"📋 Outliers ({len(outliers)} found)")
            
            # Calculate percentage of approvals in outliers
            outlier_approval_rate = (outliers['Prediction'] == 'Approved').mean()
            st.markdown(f"""
            <div class="metric-box">
                <p><strong>Outlier Approval Rate:</strong> {outlier_approval_rate:.1%}</p>
                <p><strong>Overall Approval Rate:</strong> {(data['Prediction'] == 'Approved').mean():.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                outliers[['Loan_ID', feature, 'Prediction', 'Confidence']],
                use_container_width=True
            )
        else:
            st.success("No outliers detected with current threshold")
    else:
        st.error(f"Selected feature '{feature}' not found in data")

def show_correlation_analysis(data):
    """Enhanced correlation analysis with explanatory analysis"""
    st.subheader("📊 Correlation Analysis")
    
    with st.expander("📘 Understanding Correlations", expanded=True):
        st.markdown("""
        <div class="analysis-explanation">
        <h4>Correlation Interpretation</h4>
        <p>Measures how features move together (-1 to 1):</p>
        <ul>
            <li><strong>+1:</strong> Perfect positive relationship (both increase together)</li>
            <li><strong>0:</strong> No relationship</li>
            <li><strong>-1:</strong> Perfect negative relationship (one increases as other decreases)</li>
        </ul>
        
        <h4>Key Insights:</h4>
        <ul>
            <li><strong>Strong correlations (>0.7):</strong> May indicate redundant features</li>
            <li><strong>Unexpected correlations:</strong> Need investigation for hidden relationships</li>
            <li><strong>High target correlations:</strong> Guide feature selection for modeling</li>
            <li><strong>Negative correlations:</strong> Show inverse relationships between features</li>
        </ul>
        
        <h4>Color Scale:</h4>
        <ul>
            <li><strong>Blue:</strong> Positive correlation</li>
            <li><strong>Red:</strong> Negative correlation</li>
            <li><strong>White:</strong> No correlation</li>
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
            title="Feature Correlation Matrix"
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight strongest correlations
        corr_matrix = corr.abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Ignore diagonal
        strong_corrs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
        
        if len(strong_corrs) > 0:
            st.subheader("🔍 Strongest Correlations")
            top_corrs = strong_corrs.head(5)
            
            cols = st.columns(2)
            for i, (pair, value) in enumerate(top_corrs.items()):
                with cols[i % 2]:
                    st.metric(
                        label=f"{pair[0]} ↔ {pair[1]}",
                        value=f"{value:.2f}",
                        delta="Positive" if corr.loc[pair[0], pair[1]] > 0 else "Negative"
                    )
    else:
        st.error("No numeric features available for correlation analysis")

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
        """, unsafe_allow_html=True)
    
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

# Load model and data
model, metadata = load_model()
raw_data = load_data()

# Sidebar configuration
with st.sidebar:
    st.title("🔍 Analysis Controls")
    st.markdown("---")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["New Application", "Feature Importance", "Data Distribution", "Outlier Detection", "Correlation Analysis", "Data Explorer"]
    )
    
    if analysis_type == "Feature Importance":
        num_features = st.slider("Number of Features to Show", 5, 30, 15,
                               help="Adjust to focus on top N most important features")
    elif analysis_type == "Data Distribution":
        selected_feature = st.selectbox(
            "Select Feature",
            CONFIG["data"]["features"]["numeric"] if raw_data is not None else [],
            help="Choose which feature to analyze distributions for"
        )
    elif analysis_type == "Outlier Detection":
        outlier_feature = st.selectbox(
            "Select Feature for Outlier Analysis",
            ['LoanAmount', 'ApplicantIncome', 'Loan_to_Income_Ratio'],
            help="Choose which numeric feature to analyze for outliers"
        )
        outlier_threshold = st.slider("Outlier Threshold (IQR multiplier)", 1.0, 3.0, 1.5, step=0.1,
                                    help="Higher values detect fewer extreme outliers")
    elif analysis_type == "Data Explorer":
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

# Main content
st.markdown(f'<div class="header-text">{CONFIG["app"]["content"]["header_title"]}</div>', unsafe_allow_html=True)
st.markdown("---")

if model is not None:
    if analysis_type == "New Application":
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
            proba = model.predict_proba(X)
            
            data = processed_data.copy()
            data['Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
            data['Probability'] = [max(p) for p in proba]
            data['Confidence'] = [f"{max(p)*100:.1f}%" for p in proba]
            
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
            
            if analysis_type == "Feature Importance":
                show_feature_importance(
                    model, 
                    CONFIG["data"]["features"]["numeric"] + CONFIG["data"]["features"]["categorical"], 
                    num_features
                )
                
            elif analysis_type == "Data Distribution":
                show_data_distribution(data, selected_feature)
                
            elif analysis_type == "Outlier Detection":
                show_outlier_analysis(data, outlier_feature, outlier_threshold)
                    
            elif analysis_type == "Correlation Analysis":
                show_correlation_analysis(data)
                
            elif analysis_type == "Data Explorer":
                show_data_explorer(data, confidence_threshold, status_filter)
else:
    st.error("Unable to load model. Please check the configuration.")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #7f8c8d;'>{CONFIG['app']['content']['footer']['copyright']}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; color: #7f8c8d; font-size: 0.8em;'>{CONFIG['app']['content']['footer']['disclaimer']}</div>", unsafe_allow_html=True)