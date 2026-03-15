"""Training pipeline for heart disease classification."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def load_data(path: str) -> pd.DataFrame:
    """Load a dataset from CSV."""
    df = pd.read_csv(path)
    print(f"✓ Loaded {df.shape[0]} records, {df.shape[1]} features from {path}")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning to the dataset."""
    df = df.copy()
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"  ⚠ Removed {duplicates_removed} duplicate rows")
    else:
        print(f"  ✓ No duplicates found")
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  ⚠ Found {missing} missing values")
        df = df.dropna()
        print(f"  ✓ Dropped rows with missing values")
    else:
        print(f"  ✓ No missing values")
    
    categorical_indicators = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_indicators:
        if col in df.columns and df[col].dtype != 'object':
            df[col] = df[col].astype('object')
    
    print(f"  ✓ Cleaned dataset: {len(df)} records, {df.shape[1]} features")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "HeartDisease") -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"✓ Features: {X.shape[1]} | Target: {target_col}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build and fit the preprocessing pipeline."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"✓ Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"✓ Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        verbose=0
    )
    
    preprocessor.fit(X)
    
    print(f"✓ Preprocessor created and fitted")
    return preprocessor


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Train candidate models and return metrics plus fitted pipelines."""
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
    
    results = []
    fitted_models = {}
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    for model_name, model in models.items():
        try:
            print(f"\n▶ Training {model_name}...")
            
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            full_pipeline.fit(X_train, y_train)
            
            y_pred = full_pipeline.predict(X_test)
            y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            result_row = {'Model': model_name, **metrics}
            results.append(result_row)
            fitted_models[model_name] = full_pipeline
            
            print(f"  ✓ Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  ✓ Precision: {metrics['precision']:.4f}")
            print(f"  ✓ Recall:    {metrics['recall']:.4f}")
            print(f"  ✓ F1-Score:  {metrics['f1']:.4f}")
            print(f"  ✓ ROC-AUC:   {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Failed to train {model_name}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='f1', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df, fitted_models


def rank_models(results_df: pd.DataFrame, top_k: int = 3) -> Tuple[str, list]:
    """Rank models by F1 then ROC-AUC and return top candidates."""
    ranked = results_df.sort_values(
        by=['f1', 'roc_auc'],
        ascending=False
    ).reset_index(drop=True)
    
    best_model = ranked.iloc[0]['Model']
    best_f1 = ranked.iloc[0]['f1']
    best_roc = ranked.iloc[0]['roc_auc']
    
    top_models = ranked.iloc[:top_k]['Model'].tolist()
    
    print(f"\n✓ Best Model: {best_model}")
    print(f"  F1-Score: {best_f1:.4f}")
    print(f"  ROC-AUC:  {best_roc:.4f}")
    print(f"\n✓ Top {top_k} models for hyperparameter tuning:")
    for i, model in enumerate(top_models, 1):
        metrics = ranked[ranked['Model'] == model].iloc[0]
        print(f"  {i}. {model}: F1={metrics['f1']:.4f}")
    
    return best_model, top_models


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    top_models: list,
    fitted_models: Dict[str, Pipeline],
    n_trials: int = 50
) -> Tuple[str, Pipeline, Dict]:
    """Tune top models with Optuna and return the best tuned pipeline."""
    if not HAS_OPTUNA:
        print("⚠ Optuna not installed. Skipping hyperparameter tuning.")
        return top_models[0], fitted_models[top_models[0]], {}
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    
    tuned_results = {}
    
    for model_name in top_models:
        try:
            print(f"\n▶ Tuning {model_name} (n_trials={n_trials})...")
            
            def objective(trial):
                if model_name == 'LogisticRegression':
                    C = trial.suggest_float('C', 0.001, 100, log=True)
                    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
                    
                    model = LogisticRegression(
                        C=C, solver=solver, random_state=42, max_iter=1000
                    )
                
                elif model_name == 'RandomForest':
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    max_depth = trial.suggest_int('max_depth', 5, 30)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                
                elif model_name == 'XGBoost' and HAS_XGBOOST:
                    max_depth = trial.suggest_int('max_depth', 3, 10)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    subsample = trial.suggest_float('subsample', 0.5, 1.0)
                    
                    model = xgb.XGBClassifier(
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                
                elif model_name == 'LightGBM' and HAS_LIGHTGBM:
                    num_leaves = trial.suggest_int('num_leaves', 20, 100)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    
                    model = lgb.LGBMClassifier(
                        num_leaves=num_leaves,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    )
                
                elif model_name == 'CatBoost' and HAS_CATBOOST:
                    depth = trial.suggest_int('depth', 4, 10)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                    iterations = trial.suggest_int('iterations', 50, 300)
                    
                    model = CatBoostClassifier(
                        depth=depth,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        random_state=42,
                        verbose=0
                    )
                
                else:
                    return 0.0
                
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                return f1
            
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_trial = study.best_trial
            best_f1 = best_trial.value
            
            if model_name == 'LogisticRegression':
                best_model = LogisticRegression(
                    C=best_trial.params['C'],
                    solver=best_trial.params['solver'],
                    random_state=42,
                    max_iter=1000
                )
            elif model_name == 'RandomForest':
                best_model = RandomForestClassifier(
                    n_estimators=best_trial.params['n_estimators'],
                    max_depth=best_trial.params['max_depth'],
                    min_samples_split=best_trial.params['min_samples_split'],
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'XGBoost':
                best_model = xgb.XGBClassifier(
                    max_depth=best_trial.params['max_depth'],
                    learning_rate=best_trial.params['learning_rate'],
                    n_estimators=best_trial.params['n_estimators'],
                    subsample=best_trial.params['subsample'],
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            elif model_name == 'LightGBM':
                best_model = lgb.LGBMClassifier(
                    num_leaves=best_trial.params['num_leaves'],
                    learning_rate=best_trial.params['learning_rate'],
                    n_estimators=best_trial.params['n_estimators'],
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            elif model_name == 'CatBoost':
                best_model = CatBoostClassifier(
                    depth=best_trial.params['depth'],
                    learning_rate=best_trial.params['learning_rate'],
                    iterations=best_trial.params['iterations'],
                    random_state=42,
                    verbose=0
                )
            
            best_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', best_model)
            ])
            
            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)
            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
            
            metrics = {
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            tuned_results[model_name] = {
                'pipeline': best_pipeline,
                'params': best_trial.params,
                'metrics': metrics
            }
            
            print(f"  ✓ Best F1: {metrics['f1']:.4f}")
            print(f"  ✓ Best hyperparameters from trial {best_trial.number}:")
            for key, val in best_trial.params.items():
                print(f"    - {key}: {val}")
            
        except Exception as e:
            print(f"  ⚠ Failed to tune {model_name}: {str(e)}")
    
    if tuned_results:
        best_tuned_model_name = max(
            tuned_results.keys(),
            key=lambda x: tuned_results[x]['metrics']['f1']
        )
        best_tuned_pipeline = tuned_results[best_tuned_model_name]['pipeline']
        best_tuned_params = tuned_results[best_tuned_model_name]['params']
        
        print(f"\n✓ Best Tuned Model: {best_tuned_model_name}")
        print(f"  F1-Score: {tuned_results[best_tuned_model_name]['metrics']['f1']:.4f}")
        
        return best_tuned_model_name, best_tuned_pipeline, best_tuned_params
    else:
        return top_models[0], fitted_models[top_models[0]], {}


def compute_shap_values(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    path: str = "results/figures"
) -> Tuple[np.ndarray, list]:
    """Compute SHAP values and save a summary plot."""
    if not HAS_SHAP:
        print("⚠ SHAP not installed. Skipping SHAP analysis.")
        return None, []
    
    print("\n" + "="*80)
    print("COMPUTING SHAP VALUES")
    print("="*80)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['model']
        
        X_test_processed = preprocessor.transform(X_test)
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
            feature_names = numeric_cols + list(cat_feature_names)
        else:
            feature_names = numeric_cols
        
        print(f"✓ Feature names: {len(feature_names)} features")
        
        sample_size = min(len(X_train), 100)
        X_train_processed = preprocessor.transform(X_train)
        X_train_sample = X_train_processed[:sample_size]
        
        explainer = shap.TreeExplainer(clf) if hasattr(clf, 'tree_') or hasattr(clf, 'booster_') \
                    else shap.KernelExplainer(clf.predict, X_train_sample)
        
        print(f"✓ Created {type(explainer).__name__}")
        
        shap_values = explainer.shap_values(X_test_processed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        print(f"✓ Computed SHAP values: {shap_values.shape}")
        
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
            
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_dir / 'shap_summary.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved SHAP summary plot: {plot_path}")
        except Exception as e:
            print(f"  ⚠ Could not create SHAP plot: {str(e)}")
        
        return shap_values, feature_names
        
    except Exception as e:
        print(f"⚠ SHAP computation failed: {str(e)}")
        return None, []


def compute_feature_importance(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    shap_values: np.ndarray = None,
    feature_names: list = None
) -> Dict[str, float]:
    """Build a feature-importance map from model and SHAP outputs."""
    importance_dict = {}
    
    try:
        clf = model.named_steps['model']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            
            preprocessor = model.named_steps['preprocessor']
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
                feature_names_model = numeric_cols + list(cat_feature_names)
            else:
                feature_names_model = numeric_cols
            
            for name, imp in zip(feature_names_model, importances):
                importance_dict[name] = float(imp)
    except:
        pass
    
    if shap_values is not None and feature_names is not None:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for name, imp in zip(feature_names, mean_abs_shap):
            importance_dict[name] = float(imp)
    
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict


def save_artifacts(
    best_model: Pipeline,
    preprocessor: ColumnTransformer,
    metrics: Dict[str, float],
    path: str = "results/models",
    tuned_params: Dict = None,
    feature_importance: Dict = None,
    shap_values: np.ndarray = None
) -> None:
    """Save model artifacts and training metadata to disk."""
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*80)
    print("SAVING ARTIFACTS")
    print("="*80)
    
    model_path = save_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    preprocessor_path = save_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✓ Saved preprocessor: {preprocessor_path}")
    
    if shap_values is not None:
        shap_path = save_dir / "shap_values.joblib"
        joblib.dump(shap_values, shap_path)
        print(f"✓ Saved SHAP values: {shap_path}")
    
    metadata = {
        'model_name': best_model.named_steps['model'].__class__.__name__,
        'metrics': metrics,
        'hyperparameters': tuned_params if tuned_params else {},
        'feature_importance': feature_importance if feature_importance else {},
        'timestamp': pd.Timestamp.now().isoformat(),
        'random_state': 42
    }
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")


def save_preprocessed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    path: str = "data/processed"
) -> None:
    """Transform and save processed train/test datasets."""
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
    
    all_columns = numeric_cols + list(cat_feature_names)
    
    train_df = pd.DataFrame(X_train_processed, columns=all_columns)
    train_df['target'] = y_train.values
    
    test_df = pd.DataFrame(X_test_processed, columns=all_columns)
    test_df['target'] = y_test.values
    
    train_path = save_dir / "train_processed.csv"
    test_path = save_dir / "test_processed.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Saved training data: {train_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"✓ Saved test data: {test_path}")
    print(f"  Shape: {test_df.shape}")


def run_training_pipeline(
    data_path: str = "data/raw/heart.csv",
    target_col: str = "HeartDisease",
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Run end-to-end training, evaluation, and artifact export."""
    
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION - TRAINING PIPELINE")
    print("="*80)
    
    print("\n[STEP 1] LOADING DATA")
    print("-" * 80)
    df = load_data(data_path)
    
    print("\n[STEP 2] BASIC CLEANING")
    print("-" * 80)
    df = basic_clean(df)
    
    print("\n[STEP 3] FEATURE-TARGET SPLIT")
    print("-" * 80)
    X, y = split_features_target(df, target_col)
    
    print("\n[STEP 4] TRAIN-TEST SPLIT")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    print(f"✓ Train set: {X_train.shape[0]} records")
    print(f"✓ Test set: {X_test.shape[0]} records")
    print(f"✓ Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"✓ Test class distribution: {y_test.value_counts().to_dict()}")
    
    print("\n[STEP 5] CREATE PREPROCESSOR")
    print("-" * 80)
    preprocessor = make_preprocessor(X_train)
    
    print("\n[STEP 6] TRAIN AND EVALUATE MODELS")
    print("-" * 80)
    results_df, fitted_models = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    print("\n[STEP 7] RANK MODELS")
    print("-" * 80)
    best_model_name = rank_models(results_df)
    best_model = fitted_models[best_model_name]
    best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0].to_dict()
    
    print("\n[STEP 8] SAVE ARTIFACTS")
    print("-" * 80)
    save_artifacts(best_model, preprocessor, best_metrics)
    
    print("\n[STEP 9] SAVE PREPROCESSED DATA")
    print("-" * 80)
    save_preprocessed_data(X_train, y_train, X_test, y_test, preprocessor)
    
    summary = {
        'total_records': len(df),
        'features': X.shape[1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'best_model': best_model_name,
        'metrics': best_metrics,
        'all_results': results_df.to_dict(orient='records')
    }
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED")
    print("="*80)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ F1-Score: {best_metrics['f1']:.4f}")
    print(f"✓ ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"✓ Accuracy: {best_metrics['accuracy']:.4f}\n")
    
    return summary


if __name__ == "__main__":
    summary = run_training_pipeline()
