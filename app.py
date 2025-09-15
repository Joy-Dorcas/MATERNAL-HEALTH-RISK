import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Maternal Health Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .low-risk {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .mid-risk {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
    }
    .high-risk {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the actual dataset"""
    try:
        # Load the actual dataset
        df = pd.read_csv("Maternal Health Risk Data Set.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file 'Maternal Health Risk Data Set.csv' not found. Please ensure the file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def train_model():
    """Train and return the best model"""
    df = load_and_prepare_data()
    
    # Prepare data
    le = LabelEncoder()
    df['RiskLevel_encoded'] = le.fit_transform(df['RiskLevel'])
    
    X = df.drop(['RiskLevel', 'RiskLevel_encoded'], axis=1)
    y = df['RiskLevel_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create the best pipeline (based on your results)
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            colsample_bytree=0.9,
            learning_rate=0.2,
            max_depth=6,
            n_estimators=300,
            subsample=1.0,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    # Train the model
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, le, X_test, y_test, df

def predict_risk(model, le, age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    """Make prediction for given input"""
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Convert back to original labels
    risk_labels = le.inverse_transform([0, 1, 2])
    predicted_label = le.inverse_transform([prediction])[0]
    
    return predicted_label, probability, risk_labels

def main():
    st.markdown('<h1 class="main-header">Maternal Health Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Performance", "ℹ️ About"])
    
    # Load model and data
    model, le, X_test, y_test, df = train_model()
    
    if page == "Prediction":
        prediction_page(model, le)
    elif page == "Data Analysis":
        data_analysis_page(df)
    elif page == "Model Performance":
        model_performance_page(model, X_test, y_test, le)
    elif page == "ℹ️ About":
        about_page()

def prediction_page(model, le):
    st.markdown('<h2 class="sub-header">Enter Patient Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", min_value=10, max_value=70, value=25, help="Patient's age in years")
        systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=70, max_value=160, value=120, help="Upper blood pressure reading")
        diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=49, max_value=100, value=80, help="Lower blood pressure reading")
    
    with col2:
        bs = st.slider("Blood Sugar (mmol/L)", min_value=6.0, max_value=19.0, value=7.5, help="Blood glucose level")
        body_temp = st.slider("Body Temperature (°F)", min_value=98.0, max_value=103.0, value=98.6, help="Body temperature in Fahrenheit")
        heart_rate = st.slider("Heart Rate (bpm)", min_value=7, max_value=90, value=75, help="Heart rate in beats per minute")
    
    # Input validation
    if systolic_bp <= diastolic_bp:
        st.warning("⚠️ Systolic BP should be higher than Diastolic BP")
        return
    
    if st.button("🔍 Predict Risk Level", type="primary", use_container_width=True):
        try:
            predicted_label, probabilities, risk_labels = predict_risk(
                model, le, age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate
            )
            
            # Display prediction
            risk_class = predicted_label.replace(' ', '-')
            st.markdown(f'''
            <div class="prediction-box {risk_class}">
                <h2>Prediction Result</h2>
                <h3>Risk Level: {predicted_label.upper()}</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown('<h3 class="sub-header">Probability Breakdown</h3>', unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Risk Level': risk_labels,
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)
            
            fig = px.bar(prob_df, x='Probability', y='Risk Level', orientation='h',
                        color='Probability', color_continuous_scale='RdYlGn_r',
                        title="Risk Level Probabilities")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            st.markdown('<h3 class="sub-header">Risk Interpretation</h3>', unsafe_allow_html=True)
            
            if predicted_label == 'low risk':
                st.success("✅ **Low Risk**: The patient shows normal vital signs and low risk for complications. Continue regular prenatal care.")
            elif predicted_label == 'mid risk':
                st.warning("⚠️ **Medium Risk**: The patient may need additional monitoring. Consult with healthcare provider for personalized care plan.")
            else:
                st.error("🚨 **High Risk**: The patient requires immediate medical attention and close monitoring. Contact healthcare provider immediately.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def data_analysis_page(df):
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Risk Categories", df['RiskLevel'].nunique())
    with col4:
        st.metric("Avg Age", f"{df['Age'].mean():.1f}")
    
    # Risk distribution
    st.markdown('<h3 class="sub-header">Risk Level Distribution</h3>', unsafe_allow_html=True)
    
    risk_counts = df['RiskLevel'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                title="Distribution of Risk Levels",
                color_discrete_map={'low risk': '#2ecc71', 'mid risk': '#f39c12', 'high risk': '#e74c3c'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown('<h3 class="sub-header">Feature Distributions</h3>', unsafe_allow_html=True)
    
    numeric_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox("Select Feature for Distribution", numeric_cols)
        
        fig = px.histogram(df, x=selected_feature, color='RiskLevel', 
                          marginal="box", hover_data=df.columns,
                          color_discrete_map={'low risk': '#2ecc71', 'mid risk': '#f39c12', 'high risk': '#e74c3c'},
                          title=f"Distribution of {selected_feature} by Risk Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.markdown('<h3 class="sub-header">Feature Comparison by Risk Level</h3>', unsafe_allow_html=True)
    
    selected_features = st.multiselect("Select Features to Compare", numeric_cols, default=['SystolicBP', 'DiastolicBP'])
    
    if selected_features:
        fig = px.box(df.melt(id_vars='RiskLevel', value_vars=selected_features), 
                    x='RiskLevel', y='value', color='RiskLevel', facet_col='variable',
                    facet_col_wrap=2,
                    color_discrete_map={'low risk': '#2ecc71', 'mid risk': '#f39c12', 'high risk': '#e74c3c'})
        fig.update_yaxes(matches=None)
        st.plotly_chart(fig, use_container_width=True)

def model_performance_page(model, X_test, y_test, le):
    st.markdown('<h2 class="sub-header">Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Model Type", "XGBoost")
    with col3:
        st.metric("Test Samples", len(X_test))
    
    # Confusion Matrix
    st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
    
    cm = confusion_matrix(y_test, y_pred)
    risk_labels = le.inverse_transform([0, 1, 2])
    
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                   x=risk_labels, y=risk_labels,
                   labels=dict(x="Predicted", y="Actual"),
                   title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown('<h3 class="sub-header">Classification Report</h3>', unsafe_allow_html=True)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    report_df = pd.DataFrame({
        'Risk Level': risk_labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    st.dataframe(report_df, use_container_width=True)
    
    # Feature Importance
    st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
    
    if hasattr(model.named_steps['xgb'], 'feature_importances_'):
        importance = model.named_steps['xgb'].feature_importances_
        feature_names = X_test.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance (XGBoost)")
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Purpose
    This application predicts maternal health risk levels based on various health indicators to assist healthcare providers in early risk assessment and intervention.
    
    ### Model Details
    - **Algorithm**: XGBoost Classifier (Gradient Boosting)
    - **Accuracy**: ~85.7% on test data
    - **Features**: 6 health indicators
    - **Classes**: Low Risk, Mid Risk, High Risk
    - **Deployment Platform**: Streamlit
    
    ### Input Features
    1. **Age**: Patient's age in years (Normal: 18-35 years)
    2. **Systolic BP**: Upper blood pressure reading (Normal: 90-120 mmHg)
    3. **Diastolic BP**: Lower blood pressure reading (Normal: 60-80 mmHg)
    4. **Blood Sugar**: Blood glucose level (Normal: 6.1-7.8 mmol/L)
    5. **Body Temperature**: Body temperature (Normal: 98.0-99.5°F)
    6. **Heart Rate**: Heart rate (Normal: 60-100 bpm)
    
    ### Important Disclaimers
    - This tool is for **educational and screening purposes only**
    - **Not a substitute for professional medical advice**
    - Always consult healthcare professionals for medical decisions
    - Results should be interpreted by qualified medical personnel
    
    ### Model Performance
    - **Training Accuracy**: ~92.9%
    - **Cross-Validation Score**: ~83.0% ± 1.6%
    - **Low Risk Precision**: 92.6%
    - **Mid Risk Precision**: 91.6%
    - **High Risk Precision**: 75.6%
    - **High Risk Recall**: 88.1% (Critical for patient safety)
    
    ### Data Source
    **Dataset**: Maternal Health Risk Data
    - **Source**: UCI Machine Learning Repository
    - **URL**: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
    - **Size**: 1,014 maternal health records
    - **Features**: 6 key maternal health indicators
    - **Target Classes**: Low Risk (40.0%), Mid Risk (33.1%), High Risk (26.8%)
    
    ### Technical Implementation
    - **Framework**: Streamlit (Python web application framework)
    - **Machine Learning**: scikit-learn and XGBoost libraries
    - **Visualizations**: Interactive charts with Plotly
    - **Data Processing**: Pandas and NumPy for data manipulation
    - **Model Pipeline**: StandardScaler + XGBoost with optimized hyperparameters
    - **Deployment**: Web-based application with responsive design
    
    ### Key Findings from Analysis
    - **Most Important Features**: Systolic Blood Pressure and Age
    - **Feature Correlations**: Strong correlation between systolic and diastolic BP (0.67)
    - **High-Risk Indicators**: Elevated BP (>140/90 mmHg), advanced age (>35 years), high blood sugar
    - **Model Behavior**: Tends to err on the side of caution (Mid Risk → High Risk predictions)
    - **Clinical Validation**: Feature importance aligns with established medical knowledge
    
    ### Clinical Applications
    - **Risk Screening**: Automated preliminary risk assessment
    - **Resource Allocation**: Prioritize high-risk cases for immediate attention
    - **Decision Support**: Assist healthcare providers with data-driven insights
    - **Preventive Care**: Early identification enables timely interventions
    - **Workflow Optimization**: Streamline patient triage processes
    
    ### Comprehensive Project Report
    This application is based on a comprehensive machine learning project that included:
    - **Problem Definition**: Addressing maternal mortality through early risk prediction
    - **Data Exploration**: Thorough EDA revealing key risk patterns
    - **Model Development**: Comparison of 4 algorithms (Random Forest, XGBoost, SVM, Logistic Regression)
    - **Hyperparameter Tuning**: Grid search optimization with 324 parameter combinations
    - **Validation**: 5-fold cross-validation ensuring model reliability
    - **Feature Analysis**: XGBoost feature importance revealing clinical insights
    - **Error Analysis**: Detailed misclassification patterns and edge cases
    - **Performance Metrics**: Comprehensive evaluation across all risk categories
    
    ---
    
    **Developed for healthcare analytics and machine learning applications with focus on maternal health improvement.**
    """)

if __name__ == "__main__":
    main()