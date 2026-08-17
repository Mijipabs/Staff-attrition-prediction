# Staff Attrition Prediction

Predicting employee attrition using the IBM HR Analytics dataset — hypothesis-driven EDA, a three-model comparison, and a deployed Streamlit app for real-time risk prediction.

🔗 **Live app:** [staff-attrition-olamiji.streamlit.app](https://staff-attrition-predictor.streamlit.app/)
📓 **Full analysis notebook:** [code.ipynb](./code.ipynb)

---

## Overview

This project predicts whether an employee is likely to leave a company, using the IBM HR Analytics Attrition dataset (1,470 employees, 16.1% attrition rate). Three hypotheses were formed from HR domain experience — around overtime, commute distance, and pay equity — and tested directly against the data, with each finding refined to reflect what the analysis actually revealed rather than the initial assumption.

## Key Findings

| What Was Investigated | Finding |
|---|---|
| **Overtime and income** — does overtime drive attrition on its own, regardless of pay? | Income strongly moderates overtime's impact: low earners doing overtime attrite at **53%**, vs. **17%** for high earners doing the same overtime. Pay buffers the effect, but doesn't eliminate it — a true interaction, not two independent factors. |
| **Distance from home and income** — does a long commute hurt low earners more than high earners? | Distance adds a roughly constant **~7-8 point** attrition penalty at *every* income level — an additive effect, not an interaction. Income remains the dominant driver overall. |
| **Relative income vs. promotion gap** — which matters more, pay equity or time since last promotion? | Being paid below your job level's average carries a **~9-point** attrition gap — more than double the ~4-point gap from promotion timing. Promotion gap was also counterintuitively **non-monotonic**: long-tenured, non-promoted staff attrite *less* than recently promoted staff. |

**Additional findings**, checked for confounding before being reported as genuine:
- **Sales** shows independently elevated attrition — not explained by overtime rate or age, suggesting role-specific factors (commission pressure, client-facing burnout).
- **Single employees** attrite at ~2x the rate of married employees — only partly explained by age, largely an independent effect.

## Modeling

Three models were compared: **Logistic Regression**, **Random Forest**, and **XGBoost**.

Because EDA revealed a genuine interaction effect (Overtime × Income), Logistic Regression was given an explicitly engineered interaction term to capture it — something tree models are normally expected to learn natively, but didn't reliably pick up on a dataset this size.

| Model | ROC-AUC | Recall (Left) | Precision (Left) |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.80** | **0.68** | 0.37 |
| Random Forest | 0.76 | 0.09 | 0.44 |
| XGBoost | 0.77 | 0.32 | 0.56 |

**Logistic Regression was selected as the final model.** Recall on the "left" class was prioritized over raw accuracy, since missing an employee who's actually about to leave is costlier to a business than a false alarm. The engineered interaction term gave it a clear, explainable edge over both tree-based models.

*Ensembling (Voting Classifier) was considered but not adopted — with this dataset size, the interpretability trade-off wasn't worth a marginal, uncertain gain, and a single well-justified model better suits an HR use case where explaining "why" matters.*

## The App

The deployed Streamlit app takes a single employee's details (personal info, job info, compensation, satisfaction scores, tenure) through an expandable-section form and returns a real-time attrition risk prediction, along with the probability and a short explanation of the key drivers found in the analysis.

## Tech Stack

- **Analysis:** Python, pandas, seaborn, matplotlib
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Deployment:** Streamlit, Streamlit Community Cloud

## Repository Structure

```
├── code.ipynb                              # Full EDA, hypothesis testing, and modeling notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Source dataset (IBM HR Analytics)
└── Deployment/
    ├── app.py                              # Streamlit app
    ├── requirements.txt
    ├── attrition_model.pkl                 # Trained Logistic Regression model
    ├── scaler.pkl                          # Fitted StandardScaler
    └── model_columns.pkl                   # Expected column order for inference
```

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — a fictional dataset created by IBM data scientists, commonly used for attrition/retention modeling.
