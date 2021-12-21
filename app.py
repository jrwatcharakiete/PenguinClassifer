import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Classification App

This app predicts the **Penguin** species!
""")

st.sidebar.header('Features')

uploaded_file = st.sidebar.file_uploader('Upload your input csv file here', type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_l_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_d_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_l_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        bodymass = st.sidebar.slider('Body mass (grams)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'sex': sex,
                'bill_length_mm': bill_l_mm,
                'bill_depth_mm': bill_d_mm,
                'flipper_length_mm': flipper_l_mm,
                'body_mass_g': bodymass
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins = pd.read_csv('penguins_cleaned.csv')
penguins = penguins.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

print(df.head())

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write(df)

clf = pickle.load(open('penguin_clf.pkl', 'rb'))

pred = clf.predict(df)
pred_prob = clf.predict_proba(df)

st.subheader('Prediction')
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguin_species[pred])

st.subheader('Prediction Probability')
st.write(pred_prob)


