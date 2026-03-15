# New Reusable Plotting Functions - Documentation

## Overview
Four new reusable plotting functions have been added to `src/visualization/plots.py` for comprehensive model evaluation and visualization. All functions:
- ✓ Save plots with **timestamped filenames** (format: `name_YYYYMMDD_HHMMSS.png`)
- ✓ Use **300 DPI** for publication-quality outputs
- ✓ Return the path to the saved file
- ✓ Support both pipeline and standalone models

---

## Function Details

### 1. `plot_model_comparison(results_df, save_name='model_comparison')`

Creates a comprehensive 6-panel visualization comparing all trained models.

**Parameters:**
- `results_df`: DataFrame with columns: `['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']`
- `save_name`: Base filename for saved plot (default: `'model_comparison'`)

**Output:**
- 5 metric comparison bar charts (accuracy, precision, recall, F1, ROC-AUC)
- 1 summary panel with best models by each metric
- Filename: `model_comparison_20260315_143022.png`

**Example Usage:**
```python
from visualization.plots import plot_model_comparison

comparison_plot = plot_model_comparison(results_df, save_name='model_comparison')
print(f"Saved: {comparison_plot}")
```

---

### 2. `plot_shap_summary(model, X_train, X_test, feature_names=None, save_name='shap_summary')`

Generates SHAP summary plot showing feature importance using SHAP values.

**Parameters:**
- `model`: Trained scikit-learn model (supports pipelines)
- `X_train`: Training features (used as SHAP background)
- `X_test`: Test features (for SHAP value computation)
- `feature_names`: Optional list of feature names
- `save_name`: Base filename for saved plot

**Features:**
- Automatically selects TreeExplainer for tree-based models
- Falls back to KernelExplainer for other models
- Handles both list and array SHAP value formats
- Shows top 15 features by impact

**Example Usage:**
```python
from visualization.plots import plot_shap_summary

shap_plot = plot_shap_summary(
    model=best_model,
    X_train=X_train,
    X_test=X_test,
    save_name='shap_summary'
)
```

**Requirements:**
```bash
pip install shap
```

---

### 3. `plot_confusion_matrix(y_true, y_pred, model_name='Model', save_name='confusion_matrix')`

Creates a confusion matrix heatmap with sensitivity and specificity metrics.

**Parameters:**
- `y_true`: True labels (binary: 0 or 1)
- `y_pred`: Predicted labels (binary: 0 or 1)
- `model_name`: Name of the model (for title)
- `save_name`: Base filename for saved plot

**Output:**
- Confusion matrix heatmap (4 quadrants)
- Automatically computes and displays:
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)

**Example Usage:**
```python
from visualization.plots import plot_confusion_matrix
from sklearn.pipeline import Pipeline

# Get predictions from best model
y_pred = best_model.predict(X_test)

# Create confusion matrix plot
cm_plot = plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    model_name='RandomForest',
    save_name='confusion_matrix'
)
```

---

### 4. `plot_roc_curves(models_dict, X_test, y_test, preprocessor=None, save_name='roc_curves')`

Creates ROC curves for multiple models on the same plot.

**Parameters:**
- `models_dict`: Dictionary mapping model names to fitted models: `{'Model1': model1, 'Model2': model2, ...}`
- `X_test`: Test features
- `y_test`: Test labels (binary)
- `preprocessor`: Optional preprocessor (if models aren't pipelines)
- `save_name`: Base filename for saved plot

**Output:**
- Overlaid ROC curves for all models
- Each curve labeled with model name and AUC score
- Includes random classifier baseline line (diagonal)

**Example Usage:**
```python
from visualization.plots import plot_roc_curves

# Create ROC curves for all trained models
roc_plot = plot_roc_curves(
    models_dict=fitted_models,
    X_test=X_test,
    y_test=y_test,
    preprocessor=preprocessor,
    save_name='roc_curves'
)
```

---

## Integration in Notebook

The notebook `02_Full_Model_Comparison_and_Evaluation.ipynb` has been updated to use these functions:

### Setup Cell (Cell 2)
```python
from visualization.plots import (
    plot_model_comparison,
    plot_shap_summary,
    plot_confusion_matrix,
    plot_roc_curves
)
```

### Visualization Cells
**Cell: Model Comparison**
```python
comparison_plot = plot_model_comparison(results_df, save_name='model_comparison')
```

**Cell: ROC Curves**
```python
roc_plot = plot_roc_curves(
    models_dict=fitted_models,
    X_test=X_test,
    y_test=y_test,
    preprocessor=preprocessor,
    save_name='roc_curves'
)
```

**Cell: Confusion Matrix & SHAP**
```python
# Confusion Matrix
y_pred_best = best_model.predict(X_test)
cm_plot = plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred_best,
    model_name=best_model_name,
    save_name='confusion_matrix'
)

# SHAP Summary
shap_plot = plot_shap_summary(
    model=best_model,
    X_train=X_train,
    X_test=X_test,
    save_name='shap_summary'
)
```

---

## Output File Organization

All plots are saved to `results/figures/` with timestamped names:

```
results/figures/
├── model_comparison_20260315_143022.png    # Metrics for all models
├── roc_curves_20260315_143023.png          # ROC curves comparison
├── confusion_matrix_20260315_143024.png    # Best model confusion matrix
└── shap_summary_20260315_143025.png        # SHAP feature importance
```

---

## Key Features

✓ **Timestamped filenames** prevent accidental overwrites
✓ **300 DPI** for publication-quality images
✓ **Automatic model detection** for pipeline vs standalone models
✓ **Error handling** with graceful fallbacks (e.g., SHAP unavailable)
✓ **Reusable** across different projects
✓ **Well-documented** with docstrings for IDE autocomplete
✓ **Returns paths** for programmatic access to saved files

---

## Dependencies

**Required:**
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn

**Optional (for SHAP):**
```bash
pip install shap
```

---

## Next Steps

1. Run the notebook: `02_Full_Model_Comparison_and_Evaluation.ipynb`
2. Check `results/figures/` for timestamped output plots
3. Review SHAP plot for model interpretability
4. Use confusion matrix and ROC curves to assess model performance
5. Use `plot_model_comparison()` for presentations showing multi-model comparison
