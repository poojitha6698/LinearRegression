import streamlit as st
import pickle

from preprocessing.preprocessing import load_and_preprocess_data
from model.train_model import train_model
from utils.helper import prepare_input

# Streamlit Configuration
st.set_page_config(page_title='Insurance Prediction App')

st.title('Insurance Cost Prediction')

# Load Dataset
_df = load_and_preprocess_data('insurance.csv')

st.subheader('Dataset Preview')
st.dataframe(_df.head())

# Train Model
model = train_model(_df)

# User Inputs
st.subheader('Enter Customer Details')

age = st.slider('Age', 18, 100, 25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
children = st.slider('Children', 0, 5, 0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox(
    'Region',
    ['southwest', 'southeast', 'northwest', 'northeast']
)

if st.button('Predict'):

    input_data = prepare_input(
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    )

    prediction = model.predict(input_data)

    st.success(f'Predicted Charges: ${prediction[0]:,.2f}')