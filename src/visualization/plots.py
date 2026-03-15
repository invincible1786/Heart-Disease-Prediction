"""
Comprehensive visualization module for EDA and model evaluation.
Includes 15+ EDA functions and 4 reusable evaluation functions with timestamped 300 DPI outputs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_fig(name, timestamp=False, dpi=300, bbox_inches='tight', facecolor='white'):
    """Save figure with optional timestamp and 300 DPI.
    
    Args:
        name (str): Figure filename (without extension)
        timestamp (bool): Add YYYYMMDD_HHMMSS prefix if True
        dpi (int): Resolution in dots per inch (default 300)
        bbox_inches (str): Trim whitespace ('tight' recommended)
        facecolor (str): Background color
    """
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/figures/{ts}_{name}.png'
    else:
        filename = f'results/figures/{name}.png'
    
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
    print(f"✓ Saved: {filename}")
    plt.close()


# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) - 15+ FUNCTIONS
# ============================================================================

def plot_missing_values(df):
    """Visualize missing data patterns."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values detected")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    missing[missing > 0].plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Missing Count')
    ax.set_title('Missing Values Distribution')
    save_fig('01_missing_values')


def plot_target_distribution(df, target_col='HeartDisease'):
    """Plot target variable distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    counts = df[target_col].value_counts()
    axes[0].bar(counts.index, counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{target_col} Distribution')
    axes[0].set_xticklabels(['No Disease', 'Disease'])
    
    # Percentage
    pct = df[target_col].value_counts(normalize=True) * 100
    axes[1].pie(pct.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    
    save_fig('02_target_distribution')


def plot_feature_distributions(df, features=None, n_cols=3):
    """Plot distributions for numeric features."""
    if features is None:
        features = df.select_dtypes(include=['int64', 'float64']).columns
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
    axes = axes.flatten()
    
    for idx, col in enumerate(features):
        axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_fig('03_feature_distributions')


def plot_numeric_summary_stats(df):
    """Display summary statistics as heatmap."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    stats_df = df[numeric_cols].describe().T
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(stats_df[['mean', 'std', 'min', 'max']], annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Numeric Features Summary Statistics')
    save_fig('04_summary_stats')


def plot_correlation_matrix(df, figsize=(12, 10)):
    """Heatmap of feature correlations."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix')
    save_fig('05_correlation_matrix')


def plot_categorical_features(df):
    """Distribution of categorical variables."""
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if len(cat_cols) == 0:
        print("No categorical features found")
        return
    
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(4*len(cat_cols), 4))
    
    for idx, col in enumerate(cat_cols):
        if len(cat_cols) > 1:
            ax = axes[idx]
        else:
            ax = axes
        
        df[col].value_counts().plot(kind='bar', ax=ax, color='teal')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_fig('06_categorical_features')


def plot_outliers_boxplot(df):
    """Detect outliers with boxplots."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:6]):
        axes[idx].boxplot(df[col], vert=True)
        axes[idx].set_title(f'{col} (Outliers)')
        axes[idx].set_ylabel(col)
    
    for idx in range(len(numeric_cols[:6]), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_fig('07_outliers_boxplot')


def plot_skewness_kurtosis(df):
    """Analyze skewness and kurtosis of features."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    skewness = df[numeric_cols].skew().sort_values()
    kurtosis = df[numeric_cols].kurtosis().sort_values()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].barh(range(len(skewness)), skewness.values, color='skyblue')
    axes[0].set_yticks(range(len(skewness)))
    axes[0].set_yticklabels(skewness.index)
    axes[0].set_xlabel('Skewness')
    axes[0].set_title('Feature Skewness (Symmetry)')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    
    axes[1].barh(range(len(kurtosis)), kurtosis.values, color='salmon')
    axes[1].set_yticks(range(len(kurtosis)))
    axes[1].set_yticklabels(kurtosis.index)
    axes[1].set_xlabel('Kurtosis')
    axes[1].set_title('Feature Kurtosis (Tail Heaviness)')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    save_fig('08_skewness_kurtosis')


def plot_feature_vs_target(df, target_col='HeartDisease'):
    """Box plots for numeric features vs target."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:6]):
        df.boxplot(column=col, by=target_col, ax=axes[idx])
        axes[idx].set_title(f'{col} vs {target_col}')
    
    for idx in range(len(numeric_cols[:6]), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    save_fig('09_feature_vs_target')


def plot_age_analysis(df, target_col='HeartDisease'):
    """Age group analysis."""
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Count by age group
    age_counts = df['AgeGroup'].value_counts().sort_index()
    axes[0].bar(range(len(age_counts)), age_counts.values, color='steelblue')
    axes[0].set_xticks(range(len(age_counts)))
    axes[0].set_xticklabels(age_counts.index, rotation=45)
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Age Group Distribution')
    
    # Disease rate by age group
    disease_by_age = df.groupby('AgeGroup')[target_col].mean() * 100
    axes[1].bar(range(len(disease_by_age)), disease_by_age.values, color='coral')
    axes[1].set_xticks(range(len(disease_by_age)))
    axes[1].set_xticklabels(disease_by_age.index, rotation=45)
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Disease Rate (%)')
    axes[1].set_title(f'{target_col} Rate by Age Group')
    
    save_fig('10_age_analysis')


def plot_data_quality_report(df):
    """Comprehensive data quality assessment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Missing data
    missing = df.isnull().sum()
    axes[0, 0].barh(missing[missing > 0].index, missing[missing > 0].values, color='coral')
    axes[0, 0].set_xlabel('Missing Count')
    axes[0, 0].set_title('Missing Values')
    
    # Duplicates
    n_duplicates = df.duplicated().sum()
    axes[0, 1].text(0.5, 0.5, f'Duplicates: {n_duplicates}\nShape: {df.shape}',
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Data Shape & Duplicates')
    
    # Data types
    dtypes = df.dtypes.value_counts()
    axes[1, 0].pie(dtypes.values, labels=dtypes.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Data Type Distribution')
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    axes[1, 1].text(0.5, 0.5, f'Memory: {memory_usage:.2f} MB\nRows: {len(df)}\nCols: {len(df.columns)}',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Memory & Dimensions')
    
    plt.tight_layout()
    save_fig('11_data_quality_report')


def plot_sex_distribution(df, target_col='HeartDisease'):
    """Sex-wise analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Sex count
    sex_counts = df['Sex'].value_counts()
    axes[0].bar(sex_counts.index, sex_counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sex Distribution')
    
    # Disease by sex
    disease_by_sex = df.groupby('Sex')[target_col].mean() * 100
    axes[1].bar(disease_by_sex.index, disease_by_sex.values, color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Disease Rate (%)')
    axes[1].set_title(f'{target_col} Rate by Sex')
    
    save_fig('12_sex_distribution')


def plot_pairplot_sample(df, target_col='HeartDisease', sample_size=500):
    """Pairplot for feature relationships (sampled for performance)."""
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    numeric_cols = df_sample.select_dtypes(include=['int64', 'float64']).columns[:5]
    df_plot = df_sample[list(numeric_cols) + [target_col]]
    
    g = sns.pairplot(df_plot, hue=target_col, diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pairplot: Feature Relationships (Sample)', y=1.00)
    
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    g.savefig('results/figures/13_pairplot_sample.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/figures/13_pairplot_sample.png")
    plt.close()


# ============================================================================
# MODEL EVALUATION - 4 REUSABLE FUNCTIONS WITH TIMESTAMPED OUTPUTS
# ============================================================================

def plot_model_comparison(results_df, save_name='model_comparison', timestamp=True):
    """6-panel metrics comparison across models.
    
    Args:
        results_df (pd.DataFrame): Model results with columns: Model, accuracy, precision, recall, f1, roc_auc
        save_name (str): Base filename for output
        timestamp (bool): Add timestamp to filename
    """
    if results_df is None or len(results_df) == 0:
        print("⚠ Empty results dataframe")
        return
    
    required_cols = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    if not all(col in results_df.columns for col in required_cols):
        print(f"⚠ Missing required columns. Expected: {required_cols}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold', y=1.00)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        bars = ax.bar(results_df['Model'].astype(str), results_df[metric], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Summary statistics panel
    ax_summary = axes.flatten()[5]
    ax_summary.axis('off')
    best_f1_idx = results_df['f1'].idxmax()
    best_auc_idx = results_df['roc_auc'].idxmax()
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Best F1-Score: {results_df.loc[best_f1_idx, 'Model']}
                   ({results_df.loc[best_f1_idx, 'f1']:.4f})
    
    Best ROC-AUC:  {results_df.loc[best_auc_idx, 'Model']}
                   ({results_df.loc[best_auc_idx, 'roc_auc']:.4f})
    
    Models Compared: {len(results_df)}
    """
    ax_summary.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    save_fig(save_name, timestamp=timestamp, dpi=300)


def plot_shap_summary(model, X_train, X_test, feature_names, save_name='shap_summary', timestamp=True):
    """SHAP feature importance with auto-explainer selection.
    
    Args:
        model: Trained model (CatBoost/LightGBM/XGBoost/RandomForest)
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        feature_names (list): Column names
        save_name (str): Base filename
        timestamp (bool): Add timestamp to filename
    """
    try:
        import shap
    except ImportError:
        print("⚠ SHAP not installed. Skipping SHAP plot.")
        return
    
    try:
        # Auto-select explainer based on model type
        model_type = type(model).__name__
        
        if model_type in ['CatBoostClassifier', 'LGBMClassifier', 'XGBClassifier']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 for binary
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_train.iloc[:100])
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                         plot_type="bar", show=False)
        
        plt.title('SHAP Feature Importance (Mean |SHAP Value|)', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mean |SHAP Value|', fontsize=11)
        plt.tight_layout()
        
        save_fig(save_name, timestamp=timestamp, dpi=300)
    except Exception as e:
        print(f"⚠ SHAP computation failed: {str(e)}")


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_name='confusion_matrix', timestamp=True):
    """Confusion matrix with sensitivity/specificity metrics.
    
    Args:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        model_name (str): Name for title
        save_name (str): Base filename
        timestamp (bool): Add timestamp to filename
    """
    from sklearn.metrics import confusion_matrix, recall_score
    
    cm = confusion_matrix(y_true, y_pred)
    # Sensitivity = Recall = TP / (TP + FN)
    sensitivity = recall_score(y_true, y_pred, average='binary')
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
               xticklabels=['No Disease', 'Disease'],
               yticklabels=['No Disease', 'Disease'],
               annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add metrics text
    metrics_text = f'Sensitivity: {sensitivity:.4f}\nSpecificity: {specificity:.4f}'
    ax.text(1.5, -0.5, metrics_text, fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    save_fig(save_name, timestamp=timestamp, dpi=300)


def plot_roc_curves(models_dict, X_test, y_test, preprocessor, save_name='roc_curves', timestamp=True):
    """Multi-model ROC curves with AUC labels.
    
    Args:
        models_dict (dict): {model_name: model_object}
        X_test (pd.DataFrame): Test features
        y_test (array): Test labels
        preprocessor: sklearn Pipeline for preprocessing
        save_name (str): Base filename
        timestamp (bool): Add timestamp to filename
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_dict)))
    
    for (model_name, model), color in zip(models_dict.items(), colors):
        try:
            # Preprocess test data
            if hasattr(preprocessor, 'transform'):
                X_test_processed = preprocessor.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_processed)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test_processed)
            else:
                print(f"⚠ {model_name} has no predict_proba or decision_function")
                continue
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.4f})',
                   color=color, linewidth=2.5)
        except Exception as e:
            print(f"⚠ Error plotting {model_name}: {str(e)}")
            continue
    
    # Baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC=0.5000)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_name, timestamp=timestamp, dpi=300)
