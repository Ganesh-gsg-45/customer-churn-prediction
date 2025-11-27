import streamlit as st 
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import numpy as np
model=tf.keras.models.load_model('model.h5')
with open('label.pkl','rb') as f:
    label_encoder=pickle.load(f)
with open('onehot.pkl','rb') as f:
    one_hot=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scale_encode=pickle.load(f)
st.title("customer churn pridiction")
geography=st.selectbox('Geography',one_hot.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    
})
geo_encoded = one_hot.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scale_encode.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

