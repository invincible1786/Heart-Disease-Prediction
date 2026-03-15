"""Streamlit app for heart disease risk prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import sys

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from models.predict import predict_single_patient, load_artifacts
from models.train_model import compute_shap_values

@st.cache_resource
def load_model_artifacts():
    """Load model, preprocessor, and metadata once."""
    model_path = ROOT / 'results' / 'models' / 'best_model.joblib'
    preprocessor_path = ROOT / 'results' / 'models' / 'preprocessor.joblib'
    metadata_path = ROOT / 'results' / 'models' / 'metadata.json'
    
    model, preprocessor = load_artifacts(str(model_path), str(preprocessor_path))
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, preprocessor, metadata

@st.cache_resource
def load_model_comparison():
    """Load model comparison results."""
    csv_path = ROOT / 'results' / 'models' / 'model_comparison.csv'
    return pd.read_csv(csv_path)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        opacity: 0.9;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("🫀 Heart Disease Risk Predictor")
    st.markdown("**AI-Powered Clinical Decision Support**")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

with tab1:
    st.markdown("### Patient Information")
    
    model, preprocessor, metadata = load_model_artifacts()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Demographics")
        age = st.slider("Age (years)", 28, 77, 50, help="Patient age in years")
        sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex")
        
        st.markdown("#### Vital Signs")
        resting_bp = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130, 
                               help="Blood pressure at rest")
        max_hr = st.slider("Max Heart Rate (bpm)", 60, 202, 150, 
                          help="Maximum heart rate achieved during exercise")
        
        st.markdown("#### Blood Tests")
        cholesterol = st.slider("Cholesterol (mg/dL)", 0, 603, 200, 
                               help="Serum cholesterol level (0 = not measured)")
        fasting_bs = st.selectbox("Fasting Blood Sugar", 
                                 ["< 120 mg/dL", "> 120 mg/dL"],
                                 help="Fasting blood sugar > 120 mg/dL")
    
    with col_right:
        st.markdown("#### Symptoms & ECG")
        chest_pain = st.selectbox("Chest Pain Type",
                                 ["Typical Angina", "Atypical Angina", 
                                  "Non-anginal Pain", "Asymptomatic"],
                                 help="Type of chest pain experienced")
        exercise_angina = st.selectbox("Exercise-Induced Angina",
                                      ["No", "Yes"],
                                      help="Angina induced by physical activity")
        oldpeak = st.slider("ST Segment Depression (Oldpeak)", 0.0, 6.2, 1.5,
                           help="ST segment depression induced by exercise")
        st_slope = st.selectbox("ST Segment Slope",
                               ["Upsloping", "Flat", "Downsloping"],
                               help="Slope of ST segment during exercise")
        resting_ecg = st.selectbox("Resting ECG",
                                  ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
                                  help="Resting electrocardiogram result")
    
    patient_data = {
        'Age': age,
        'Sex': 'M' if sex == 'Male' else 'F',
        'ChestPainType': {
            'Typical Angina': 'TA',
            'Atypical Angina': 'ATA',
            'Non-anginal Pain': 'NAP',
            'Asymptomatic': 'ASY'
        }[chest_pain],
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == "> 120 mg/dL" else 0,
        'RestingECG': {
            'Normal': 'Normal',
            'ST-T Abnormality': 'ST',
            'Left Ventricular Hypertrophy': 'LVH'
        }[resting_ecg],
        'MaxHR': max_hr,
        'ExerciseAngina': 'Y' if exercise_angina == 'Yes' else 'N',
        'Oldpeak': oldpeak,
        'ST_Slope': {
            'Upsloping': 'Up',
            'Flat': 'Flat',
            'Downsloping': 'Down'
        }[st_slope]
    }
    
    if st.button("🔮 Predict Risk", key="predict_btn", use_container_width=True):
        try:
            result = predict_single_patient(patient_data)
            
            st.markdown("---")
            st.markdown("### 📋 Prediction Results")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Diagnosis",
                    value="⚠️ DISEASE" if result['prediction'] == 1 else "✅ HEALTHY",
                    delta="High Risk" if result['prediction'] == 1 else "Low Risk"
                )
            
            with metric_col2:
                st.metric(
                    label="Risk Probability",
                    value=f"{result['probability']:.1%}",
                    delta=f"{result['probability']-0.5:.1%}" if result['prediction'] == 1 else None
                )
            
            with metric_col3:
                st.metric(
                    label="Risk Level",
                    value=result['risk_level'],
                    delta="⬆️ Alert" if result['risk_level'] == 'High' else "Normal"
                )
            
            st.markdown("### Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability'] * 100,
                title={'text': "Disease Probability (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF6B6B" if result['prediction'] == 1 else "#4CAF50"},
                    'steps': [
                        {'range': [0, 33], 'color': "#E8F5E9"},
                        {'range': [33, 66], 'color': "#FFF9C4"},
                        {'range': [66, 100], 'color': "#FFEBEE"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            confidence = max(result['probability'], 1 - result['probability'])
            st.markdown(f"**Model Confidence:** `{confidence:.1%}`")
            
            st.markdown("### 💊 Clinical Recommendation")
            if result['prediction'] == 1:
                st.warning(f"""
                
                **Recommendation:** This patient shows signs of heart disease risk and requires:
                - Immediate medical consultation with a cardiologist
                - Comprehensive cardiac evaluation (ECG, stress test)
                - Blood biomarker assessment (troponin, BNP)
                - Risk stratification for preventive intervention
                """)
            else:
                st.success(f"""
                
                **Recommendation:** Continue routine preventive care:
                - Annual health screenings
                - Regular exercise (150 min/week moderate activity)
                - Heart-healthy diet (Mediterranean style)
                - Monitor vital signs regularly
                """)
            
            st.markdown("### 🔍 Key Risk Factors (From Model)")
            feature_importance = metadata.get('feature_importance', {})
            
            if feature_importance:
                top_features = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:5])
                
                fig_feat = go.Figure(data=[
                    go.Bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h',
                        marker=dict(
                            color=list(top_features.values()),
                            colorscale='Reds',
                            showscale=False
                        ),
                        text=[f"{v:.4f}" for v in top_features.values()],
                        textposition='auto'
                    )
                ])
                fig_feat.update_layout(
                    title="Top 5 Features Contributing to Predictions",
                    xaxis_title="SHAP Mean |Value|",
                    yaxis_title="Feature",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_feat, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

with tab2:
    st.markdown("### Model Comparison Results")
    
    try:
        results_df = load_model_comparison()
        
        st.dataframe(
            results_df.style.format({
                'accuracy': '{:.4f}',
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1': '{:.4f}',
                'roc_auc': '{:.4f}'
            }).highlight_max(subset=['f1', 'roc_auc'], color='lightgreen'),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_metrics = px.bar(
                results_df,
                x='Model',
                y=['accuracy', 'precision', 'recall', 'f1'],
                title="Classification Metrics Comparison",
                barmode='group',
                color_discrete_map={
                    'accuracy': '#667eea',
                    'precision': '#764ba2',
                    'recall': '#f093fb',
                    'f1': '#4facfe'
                }
            )
            fig_metrics.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            fig_auc = px.bar(
                results_df,
                x='Model',
                y='roc_auc',
                title="ROC-AUC Scores",
                color='roc_auc',
                color_continuous_scale='Viridis',
                text='roc_auc'
            )
            fig_auc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_auc.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_auc, use_container_width=True)
        
        best_model_name = results_df.loc[results_df['f1'].idxmax(), 'Model']
        best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
        
        st.markdown("### 🏆 Best Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", best_model_name)
        with col2:
            st.metric("F1-Score", f"{best_metrics['f1']:.4f}")
        with col3:
            st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
        with col4:
            st.metric("Sensitivity", f"{best_metrics['recall']:.4f}")
    
    except Exception as e:
        st.error(f"Error loading model comparison: {str(e)}")

with tab3:
    st.markdown("### 📚 About This Application")
    
    st.info("""
    
    This application uses machine learning to predict the risk of heart disease based on clinical features.
    The model is trained on 918 patient records with 12 clinical features.
    """)
    
    try:
        model, preprocessor, metadata = load_model_artifacts()
        
        st.markdown("### Model Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Base Model:** CatBoost Classifier
            
            **Hyperparameters:**
            """)
            if 'hyperparameters' in metadata:
                for key, value in metadata['hyperparameters'].items():
                    st.code(f"{key}: {value}")
        
        with col2:
            st.markdown("""
            **Preprocessing Pipeline:**
            - StandardScaler for numeric features
            - OneHotEncoder for categorical features
            - Stratified train/test split (80/20)
            
            **Optimization:**
            - Optuna hyperparameter tuning (50 trials)
            - Cross-validated metrics
            """)
        
        st.markdown("### ⚡ Top Features (SHAP)")
        feature_importance = metadata.get('feature_importance', {})
        if feature_importance:
            top_10 = dict(sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:10])
            for i, (feat, importance) in enumerate(top_10.items(), 1):
                st.write(f"{i}. **{feat}** — {importance:.4f}")
    
    except Exception as e:
        st.warning(f"Could not load metadata: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📖 Usage Instructions")
    st.markdown("""
    1. **Go to Predict Tab** — Enter patient clinical features
    2. **Click Predict** — Model computes heart disease risk
    3. **View Results** — Get probability, diagnosis, and clinical notes
    4. **Check Performance** — Review model metrics in Model Performance tab
    5. **Learn More** — Read feature importance and model details here
    
    This tool is for **educational and research purposes only**. 
    It is not a substitute for professional medical diagnosis or treatment.
    Always consult with qualified healthcare professionals.
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **ML Framework**
        - scikit-learn
        - CatBoost
        - Optuna
        """)
    
    with tech_col2:
        st.markdown("""
        **Explainability**
        - SHAP
        - Feature Importance
        - Waterfall Plots
        """)
    
    with tech_col3:
        st.markdown("""
        **Deployment**
        - Streamlit
        - Python 3.9+
        - Plotly Visualization
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ for Healthcare AI</p>
        <p><small>Heart Disease Prediction Pipeline | March 2026</small></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem; margin-top: 2rem;'>
    <p>This application is powered by CatBoost (ROC-AUC: 93.77%) 
    | <a href='https://github.com/invincible1786/Heart-Disease-Prediction'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
