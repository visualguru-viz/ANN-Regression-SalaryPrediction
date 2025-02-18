import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# load the model
model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encode_geo.pkl', 'rb') as file:
    onehot_encode_geo = pickle.load(file)
    
with open('sscaler.pkl', 'rb') as file:
    sscaler = pickle.load(file)


## Streamlit app

st.title("Salary Estimation")

geography = st.selectbox('Geography', onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 99)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0,10)
numOfProducts = st.slider('Number of Proucts', 1,4)
hasCrCard = st.selectbox('Has Credit Card', [0,1])
exited = st.selectbox('Exited', [0,1])
isActiveMember = st.selectbox('Is Active Member∂', [0,1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numOfProducts],
    'HasCrCard': [hasCrCard],
    'IsActiveMember': [isActiveMember],
    'Exited': [exited]
}
)
# one-hot encoding - Geo

geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=('Geography_France', 'Geography_Germany', 'Geography_Spain'))

# st.write(geo_encoded_df)

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)


input_scaled = sscaler.transform(input_data)

prediction = model.predict(input_scaled)

sal_predicted = prediction[0][0]
# if predict_prob <=0.5:
#     pred = "The customer is not likely to churn."
# else:
#     pred = "The customer is likely to churn!"


st.write(sal_predicted)