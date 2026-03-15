# 🫀 Streamlit App - Heart Disease Risk Predictor

Beautiful web interface for heart disease risk prediction with interactive visualizations and SHAP explainability.

## ✨ Features

- **Patient Input Dashboard** — Sidebar form for entering 12 clinical features
- **Real-time Predictions** — CatBoost model with 93.77% ROC-AUC
- **Probability Gauge** — Interactive visualization of disease risk
- **Model Metrics Tab** — Comparative performance across 5 models
- **About Section** — Feature importance and model architecture
- **Cached Resources** — Fast loading with `@st.cache_resource`

## 🚀 Running the App

### Prerequisites
```bash
pip install -r ../requirements.txt
pip install streamlit plotly
```

### Launch
From the project root directory:
```bash
streamlit run app/streamlit_app.py
```

Or from the app directory:
```bash
cd app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📋 App Structure

### Three Main Tabs:

#### 1. 🔮 Predict Tab
- **Sidebar Input Form**
  - Demographics: Age, Sex
  - Vital Signs: Resting BP, Max Heart Rate
  - Blood Tests: Cholesterol, Fasting Blood Sugar
  - Symptoms: Chest Pain Type, Exercise Angina, ST Depression, ECG findings
- **Prediction Results**
  - Diagnosis badge (Disease/Healthy)
  - Risk probability gauge (0-100%)
  - Risk level indicator
  - Clinical recommendations
  - Top feature importance chart

#### 2. 📊 Model Performance Tab
- Model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- Bar charts for metrics comparison
- Best model highlight (CatBoost)
- ROC-AUC ranking

#### 3. ℹ️ About Tab
- Model architecture details
- Hyperparameters
- Top 10 SHAP features
- Technology stack
- Usage instructions
- Disclaimer

## 🛠️ Customization

### Adjust Feature Ranges
Edit the `st.slider()` and `st.selectbox()` calls in the Predict tab:
```python
age = st.slider("Age (years)", 28, 77, 50)  # Min, Max, Default
```

### Change Model Path
```python
model_path = Path('path/to/your/model.joblib')
```

### Modify Theme
```python
st.set_page_config(
    page_icon="🫀",
    layout="wide",
    theme="light"  # or "dark"
)
```

## 📊 Caching Strategy

The app uses `@st.cache_resource` to efficiently load:
- **Best model** (best_model.joblib) — ~50MB
- **Preprocessor** (preprocessor.joblib) — ~2KB
- **Metadata** (metadata.json) — Feature importance, hyperparameters
- **Model comparison CSV** — Results across 5 models

Caching eliminates reload on interaction, ensuring snappy performance.

## 🔗 Integration with Core Pipeline

The app imports from the core prediction module:
```python
from src.models.predict import predict_single_patient, load_artifacts
from src.models.train_model import compute_shap_values
```

Requires that the following are available:
- `src/models/predict.py` — `predict_single_patient()`, `load_artifacts()`
- `results/models/best_model.joblib` — Trained CatBoost model
- `results/models/preprocessor.joblib` — Scikit-learn preprocessing pipeline
- `results/models/metadata.json` — Feature importance, hyperparameters
- `results/models/model_comparison.csv` — Model metrics table

## 🎨 Styling

The app includes custom CSS for:
- Metric cards with gradient background
- Color-coded risk levels (Green = Low, Yellow = Medium, Red = High)
- Interactive Plotly visualizations
- Responsive layout (wide mode)

## 🚨 Known Limitations

1. **Requires Artifacts** — Ensure all model files exist in `results/models/`
2. **CPU-Only** — No GPU acceleration (CatBoost will use CPU)
3. **Offline** — App is fully local, no cloud integration
4. **Single Patient** — Predictions one patient at a time (no batch)

## ⚠️ Clinical Disclaimer

This application is for **educational and research purposes only**.
It is **not** a substitute for professional medical diagnosis or treatment.
Always consult qualified healthcare professionals for medical decisions.

## 📞 Support

For issues or questions:
1. Check app logs: `streamlit run app/streamlit_app.py --logger.level=debug`
2. Verify artifact paths: `ls results/models/`
3. Test prediction module: `python -c "from src.models.predict import load_artifacts"`

---

Built with Streamlit • Powered by CatBoost • Explained by SHAP
