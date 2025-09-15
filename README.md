# Maternal Health Risk Prediction

https://maternal-risk-guardian.lovable.app/

This project applies supervised machine learning techniques to predict maternal health risk levels (Low, Mid, High) based on clinical features.  
The dataset was collected from hospitals, community clinics, and maternal healthcare centers  

## 1. Problem Statement

Maternal health complications are a significant global challenge, especially in rural and resource-limited areas.  
The goal of this project is to build a supervised machine learning model that predicts maternal health risk levels to support early detection and timely interventions.  

- **Problem**: Predict maternal risk levels based on clinical features.  
- **Importance**: Early detection reduces maternal mortality and improves pregnancy outcomes.  
- **Beneficiaries**: Doctors, community health workers, and public health policymakers.  
- **ML Task**: Multiclass classification.  

## 2. Dataset
https://archive.ics.uci.edu/dataset/863/maternal+health+risk
- **Name**: Maternal Health Risk  
- **Donated**: August 14, 2023  
- **Instances**: 1,013  
- **Features**: 6 (clinical and demographic)  
- **Target**: RiskLevel (Categorical: Low, Mid, High)  
- **Missing Values**: None  

### Features
| Variable     | Type    | Description                                                                 | Units  |
|--------------|---------|-----------------------------------------------------------------------------|--------|
| Age          | Integer | Age of the pregnant woman                                                  | years  |
| SystolicBP   | Integer | Upper value of blood pressure                                              | mmHg   |
| DiastolicBP  | Integer | Lower value of blood pressure                                              | mmHg   |
| BS           | Integer | Blood sugar level                                                          | mmol/L |
| BodyTemp     | Integer | Body temperature                                                           | °F     |
| HeartRate    | Integer | Resting heart rate                                                         | bpm    |
| RiskLevel    | Target  | Risk intensity level (Low, Mid, High)                                       | -      |



## 3. Data Collection & Understanding

- Data collected via IoT-based maternal health monitoring systems in Bangladesh.  
- Dataset contains only numerical features and a categorical target.  
- No missing values.  
- Exploratory Data Analysis (EDA) includes:  
  - Distribution of each feature  
  - Correlation heatmap  
  - Risk level distribution (class balance check)  



## 4. Data Preprocessing

- Handle outliers in features such as blood pressure and blood sugar.  
- Normalize/standardize numerical features.  
- Encode target variable (Low, Mid, High → 0, 1, 2).  
- Train-test split (e.g., 80/20).  
- Apply resampling if class imbalance is detected.  



## 5. Modeling

Baseline and advanced models considered:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

Hyperparameter tuning is performed using GridSearchCV or RandomizedSearchCV.  
Final model is chosen based on best generalization performance.  



## 6. Evaluation

Metrics used:  
- Accuracy  
- Precision  
- Recall  
- F1-score  

Visualizations:  
- Confusion matrix  
- Feature importance (tree-based models)  
- Validation curves (hyperparameter analysis)  
- Learning curves (bias vs variance analysis)  



## 7. Error Analysis

- Identify misclassified cases in confusion matrix.  
- Analyze whether errors occur more in specific risk levels.  
- Discuss possible causes: limited data, overlapping feature ranges, model complexity.  
- Suggest improvements: collect more data, engineer new features, test additional algorithms.  



## 8. Model Interpretation

- Tree-based models provide feature importance ranking.  
- Example insights:  
  - Abnormal blood pressure strongly contributes to high-risk predictions.  
  - Elevated blood sugar and body temperature increase risk level.  



## 9. Deployment

- Deploy model with **Streamlit** for interactive prediction.  
- Application allows healthcare workers to input patient data and receive real-time risk assessment.  

