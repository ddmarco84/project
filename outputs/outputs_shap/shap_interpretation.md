# SHAP Interpretation

Selected model: XGBoost.
Selection metric: roc_auc. The model was selected from the main predictive metrics, not from expected-loss analysis.

The SHAP values were computed for the transformed feature matrix used by the fitted XGBoost model. For binary XGBoost with the default TreeExplainer output, SHAP values are in raw-margin/log-odds space. Positive SHAP values push the prediction toward higher default risk; negative values push it toward lower default risk.

## Test Metrics

|   threshold |   accuracy |   precision_BAD_1 |   recall_BAD_1 |   f1_BAD_1 |   roc_auc |   average_precision |
|------------:|-----------:|------------------:|---------------:|-----------:|----------:|--------------------:|
|         0.5 |     0.9211 |             0.814 |         0.7843 |     0.7989 |    0.9641 |              0.8971 |

## Most Important Features

|   rank | feature         |   mean_abs_shap |   mean_shap |
|-------:|:----------------|----------------:|------------:|
|      1 | DEBTINC         |          0.9187 |     -0.5146 |
|      2 | DEBTINC_missing |          0.7798 |     -0.2883 |
|      3 | CLAGE           |          0.7225 |     -0.4154 |
|      4 | DELINQ          |          0.7168 |     -0.2153 |
|      5 | LOAN            |          0.4735 |     -0.3299 |
|      6 | MORTDUE         |          0.4195 |     -0.2511 |
|      7 | CLNO            |          0.3964 |     -0.2825 |
|      8 | DEROG           |          0.3897 |     -0.1113 |

The highest-ranked features are dominated by debt burden, missing debt-to-income information, credit-history variables, and collateral or balance-sheet variables. This is consistent with the original notebook, where SHAP highlighted DEBTINC, DEBTINC_missing, DELINQ, CLAGE, VALUE, and related missingness signals.

One-hot encoded categorical variables such as JOB_Sales or REASON_HomeImp should be interpreted relative to the omitted baseline category. Missing-category indicators and one-hot features are predictive associations, not causal explanations.

Base value in log-odds space: 0.6222.