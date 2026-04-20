# CFRP Composite Materials Analysis - Explainable AI

This repository contains a post-hoc explainability and querying framework that extracts temporal symbolic rules from a trained Transformer-based Remaining Useful Life (RUL) predictor for Carbon Fiber Reinforced Polymer (CFRP) composite structures under fatigue loading.

## Key Features

- **Transformer RUL Predictor**: Deep learning model that captures complex temporal dependencies in multi-sensor time series (PZT, acoustic emission, strain gauges, optical sensors).
- **Three-Tier Rule Extraction Architecture**:
  - *Standard RuleFit*: Distills predictions into static, interpretable conjunctive rules using a gradient-boosted tree ensemble and Lasso regression.
  - *Temporal RuleFit-AF*: Augments input representation with sequence-derived statistics (slope, delta, volatility) to capture temporal dynamics.
  - *STL RuleFit*: Employs Signal Temporal Logic (STL), using min/max robust signatures that map to Globally ($G$) and Finally ($F$) temporal operators.
- **Interactive Multi-Query Engine**: Supports 7 structured query types (Point, Why, Why-Not, What-If, Counterfactual, Global Feature Influence, Cohort Pattern). Automatically generates explanations in three ways: programmable rule format, formal STL notation, and natural English.
- **Quantitative XAI Benchmarking**: Systematically compares XAI methods using Faithfulness, Stability, Compactness, and Concordance (vs SHAP) metrics.

## Repository Structure

- `notebooks/modules/rulefit.py`: Base rule extraction logic using tree ensembles and Lasso.
- `notebooks/modules/temporal_rules.py`: Temporal feature augmentation pipeline.
- `notebooks/modules/stl_rules.py`: Signal Temporal Logic feature extraction and translation logic.
- `notebooks/modules/query_engine.py`: The interactive query engine to reason about model predictions.
- `notebooks/modules/xai_benchmark.py`: Benchmarking suite for the four primary evaluation metrics.
- `BTP report 8th sem/`: LaTeX files containing the detailed methodology, results, and structural validation of the thesis.

## Domain Validation
The rules extracted by this system have been validated against the established three-stage CFRP fatigue degradation model (Matrix Cracking, Delamination, Fiber Breakage), confirming 100% physical consistency.

## Usage
The framework operates by fitting the RuleFit surrogates to your trained Transformer's responses:

```python
from modules import STLRuleFit, STLRuleQueryEngine

# Fit the STL Rule Surrogate
stl_rulefit = STLRuleFit(
    model=transformer_model,
    feature_names=feature_names,
    augment_features=["avg_scatter_energy", "stiffness_degradation"],
    seq_len=10,
)
stl_rulefit.fit(X_train, y_train_pred, stage_labels)

# Instantiate the Query Engine
query_engine = STLRuleQueryEngine(stl_rulefit, stage_boundaries)

# Perform a 'Why' query with formal STL translations
why_explanation = query_engine.stl_why_query(sample)
print(why_explanation)
```
