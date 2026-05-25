import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.preprocessing import preprocess_data
from feature_engineering.feature_engineering import feature_engineering
from model.train_model import train_model
from prediction.predict import make_prediction
from evaluation.evaluation import evaluate_model
from utils.helper import prepare_input


# Streamlit Page Configuration
st.set_page_config(
    page_title='Insurance Cost Prediction',
    layout='wide'
)

# Title
st.title('Insurance Cost Prediction using Linear Regression')

# Load Dataset
df = preprocess_data('data/insurance.csv')

# Dataset Preview
st.subheader('Dataset Preview')
st.dataframe(df.head())

# =========================================
# EDA SECTION
# =========================================

st.subheader('Exploratory Data Analysis (EDA)')

# -------------------------------
# Correlation Heatmap
# -------------------------------

st.write('### Correlation Heatmap')

fig1, ax1 = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax1
)

st.pyplot(fig1)

# -------------------------------
# Charges Distribution
# -------------------------------

st.write('### Charges Distribution')

fig2, ax2 = plt.subplots(figsize=(8, 5))

sns.histplot(
    df['charges'],
    kde=True,
    ax=ax2
)

ax2.set_title('Distribution of Insurance Charges')

st.pyplot(fig2)

# -------------------------------
# BMI vs Charges
# -------------------------------

st.write('### BMI vs Charges')

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    x='bmi',
    y='charges',
    data=df,
    ax=ax3
)

ax3.set_title('BMI vs Charges')

st.pyplot(fig3)

# -------------------------------
# Smoker vs Charges
# -------------------------------

st.write('### Smoker vs Charges')

fig4, ax4 = plt.subplots(figsize=(8, 5))

sns.boxplot(
    x='smoker',
    y='charges',
    data=df,
    ax=ax4
)

ax4.set_title('Smoker vs Charges')

st.pyplot(fig4)

# =========================================
# FEATURE ENGINEERING
# =========================================

X, y = feature_engineering(df)

# =========================================
# TRAIN MODEL
# =========================================

model, X_test, y_test = train_model(X, y)

# Predictions
y_pred = model.predict(X_test)

# =========================================
# EVALUATION METRICS
# =========================================

mae, mse, rmse, r2 = evaluate_model(
    y_test,
    y_pred
)

st.subheader('Model Evaluation Metrics')

col1, col2 = st.columns(2)

with col1:
    st.metric('MAE', f'{mae:.2f}')
    st.metric('MSE', f'{mse:.2f}')

with col2:
    st.metric('RMSE', f'{rmse:.2f}')
    st.metric('R2 Score', f'{r2:.2f}')

# =========================================
# USER INPUT SECTION
# =========================================

st.subheader('Enter Customer Details')

age = st.slider(
    'Age',
    18,
    100,
    25
)

sex = st.selectbox(
    'Sex',
    ['male', 'female']
)

bmi = st.slider(
    'BMI',
    10.0,
    50.0,
    25.0
)

children = st.slider(
    'Children',
    0,
    5,
    0
)

smoker = st.selectbox(
    'Smoker',
    ['yes', 'no']
)

region = st.selectbox(
    'Region',
    [
        'northeast',
        'northwest',
        'southeast',
        'southwest'
    ]
)

# =========================================
# PREDICTION
# =========================================

if st.button('Predict Insurance Charges'):

    input_data = prepare_input(
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    )

    prediction = make_prediction(
        model,
        input_data
    )

    st.success(
        f'Predicted Insurance Charges: ${prediction[0]:,.2f}'
    )