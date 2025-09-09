# MATERNAL-HEALTH-RISK
Maternal health complications are a major concern in global healthcare, especially in rural and resource-limited areas. The goal of this project is to apply supervised machine learning methods to predict maternal health risk levels (low, mid, high) using clinical and demographic features.  

# Maternal Health Risk Prediction

This project applies supervised machine learning techniques to predict maternal health risk levels based on clinical features.  
The dataset was collected from hospitals, community clinics, and maternal healthcare centers in rural Bangladesh through an IoT-based monitoring system.  



## Dataset

- **Name**: Maternal Health Risk  
- **Donated**: August 14, 2023  
- **Instances**: 1,013  
- **Features**: 6 (all clinical and demographic)  
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

**Introductory Paper**:  
Ahmed, Marzia, M. A. Kashem, Mostafijur Rahman, and S. Khatun.  
*"Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT)"*.  
Lecture Notes in Electrical Engineering, vol 632, 2020.



## Project Structure

maternal-health-risk/
│
├── data/
│ └── maternal_health_risk.csv # Raw dataset
│
├── src/
│ ├── preprocess.py # Data preprocessing
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│
├── app.py # Streamlit app for interactive demo
│
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code



## Installation

Clone this repository and install dependencies:

Dataset collected through IoT-based monitoring systems in Bangladesh.

Special thanks to the researchers: Marzia Ahmed, M. A. Kashem, Mostafijur Rahman, and S. Khatun.
