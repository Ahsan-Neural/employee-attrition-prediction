# Employee Attrition Prediction


### Project Overview
End-to-end machine learning pipeline to predict employee attrition using the
IBM HR Analytics dataset (1,470 records, 35 features).

### Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Key Results

| Model | Recall | F1-Score | Pipeline |
|---|---|---|---|
| Logistic Regression | 0.6170 | 0.4793 | Engineered |
| Random Forest (tuned threshold) | 0.7234 | 0.5231 | Engineered |
| XGBoost | 0.2979 | 0.4058 | Engineered |
| Decision Tree | 0.3191 | 0.3226 | Engineered |

### Techniques Used
- Exploratory Data Analysis (EDA)
- Label Encoding and One-Hot Encoding
- StandardScaler normalization
- SMOTE oversampling (applied to training data only)
- IQR-based outlier capping
- Composite feature engineering (IncomePerYear, SatisfactionIndex, PromotionLag, YearsWithManagerRatio)
- 5-Fold Stratified Cross-Validation
- GridSearchCV Hyperparameter Tuning
- Probability threshold optimization
- SHAP explainability

### Top Findings
- OverTime is the single strongest predictor with SHAP value of 1.30
- StockOptionLevel and MonthlyIncome are the top financial retention signals
- Feature engineering improved Logistic Regression Recall from 0.5957 to 0.6170
- Random Forest at threshold 0.30 achieves Recall of 0.7234 after tuning
- ROC-AUC for top 3 models: 0.79 (Logistic Regression, Random Forest, XGBoost)

### Dataset
IBM HR Analytics Employee Attrition dataset
Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
### Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Matplotlib, Seaborn
Platform: Kaggle Notebook (CPU)
