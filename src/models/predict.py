"""Inference utilities for heart disease prediction."""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')



def load_artifacts(
    model_path: str = "results/models/best_model.joblib",
    preprocessor_path: str = "results/models/preprocessor.joblib"
) -> Tuple[Any, Any]:
    """Load model and preprocessor artifacts from disk."""
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Loaded model: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"✓ Loaded preprocessor: {preprocessor_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {str(e)}")
    
    return model, preprocessor



def preprocess_input(input_df: pd.DataFrame, preprocessor: Any) -> np.ndarray:
    """Transform input features with a fitted preprocessor."""
    try:
        processed = preprocessor.transform(input_df)
        return processed
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")



def predict(input_df: pd.DataFrame, model: Any = None, preprocessor: Any = None) -> np.ndarray:
    """Predict binary class labels for the input rows."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()
    
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame")
    
    if input_df.empty:
        raise ValueError("input_df cannot be empty")
    
    predictions = model.predict(input_df)
    return predictions



def predict_proba(input_df: pd.DataFrame, model: Any = None, preprocessor: Any = None) -> np.ndarray:
    """Predict class-1 probabilities for the input rows."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()
    
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame")
    
    if input_df.empty:
        raise ValueError("input_df cannot be empty")
    
    try:
        probabilities = model.predict_proba(input_df)[:, 1]
    except AttributeError:
        raise RuntimeError("Model does not support predict_proba")
    
    return probabilities



def predict_single_patient(
    patient_dict: Dict[str, Any],
    model: Any = None,
    preprocessor: Any = None
) -> Dict[str, Any]:
    """Predict risk for one patient and return label, score, and guidance."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()
    
    if not isinstance(patient_dict, dict):
        raise TypeError("patient_dict must be a dictionary")
    
    if not patient_dict:
        raise ValueError("patient_dict cannot be empty")
    
    try:
        patient_df = pd.DataFrame([patient_dict])
    except Exception as e:
        raise ValueError(f"Failed to create dataframe from patient_dict: {str(e)}")
    
    try:
        pred_class = model.predict(patient_df)[0]
        pred_proba = model.predict_proba(patient_df)[0, 1]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
    
    if pred_proba < 0.3:
        risk_level = "Low"
        recommendation = "Continue regular health checkups"
    elif pred_proba < 0.7:
        risk_level = "Medium"
        recommendation = "Consult with cardiologist for detailed evaluation"
    else:
        risk_level = "High"
        recommendation = "Urgent medical attention recommended"
    
    result = {
        'prediction': int(pred_class),
        'prediction_label': 'Disease' if pred_class == 1 else 'No Disease',
        'probability': float(pred_proba),
        'confidence': float(pred_proba * 100),
        'risk_level': risk_level,
        'recommendation': recommendation
    }
    
    return result



def predict_batch(
    input_df: pd.DataFrame,
    model: Any = None,
    preprocessor: Any = None,
    include_proba: bool = True
) -> pd.DataFrame:
    """Run batch predictions and optionally append probability outputs."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()
    
    result_df = input_df.copy()
    
    predictions = predict(input_df, model, preprocessor)
    result_df['prediction'] = predictions
    result_df['prediction_label'] = result_df['prediction'].map({0: 'No Disease', 1: 'Disease'})
    
    if include_proba:
        probabilities = predict_proba(input_df, model, preprocessor)
        result_df['probability'] = probabilities
        result_df['confidence'] = result_df['probability'] * 100
        
        result_df['risk_level'] = pd.cut(
            result_df['probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
    
    return result_df



def demo_prediction():
    """Run a quick CLI demo for a sample patient."""
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION - DEMO")
    print("="*80)
    
    patient = {
        'Age': 55,
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingBP': 150,
        'Cholesterol': 280,
        'FastingBS': 1,
        'RestingECG': 'LVH',
        'MaxHR': 120,
        'ExerciseAngina': 'Y',
        'Oldpeak': 2.5,
        'ST_Slope': 'Flat'
    }
    
    print("\n📋 PATIENT DATA:")
    print("-" * 80)
    for key, value in patient.items():
        print(f"  {key:20s}: {value}")
    
    print("\n🔄 LOADING ARTIFACTS...")
    print("-" * 80)
    try:
        model, preprocessor = load_artifacts()
    except FileNotFoundError as e:
        print(f"⚠ Error: {e}")
        print("Please run train_model.py first to generate artifacts")
        return
    
    print("\n🔮 MAKING PREDICTION...")
    print("-" * 80)
    try:
        result = predict_single_patient(patient, model, preprocessor)
        
        print(f"  Prediction:      {result['prediction_label']}")
        print(f"  Probability:     {result['probability']:.4f} ({result['confidence']:.1f}%)")
        print(f"  Risk Level:      {result['risk_level']}")
        print(f"  Recommendation:  {result['recommendation']}")
        
    except Exception as e:
        print(f"⚠ Prediction error: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_prediction()
