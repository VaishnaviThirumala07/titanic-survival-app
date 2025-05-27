# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:38:08 2025

@author: hi
"""

import numpy as np
import pandas as pd
import streamlit as st
import pickle
pickle_in =open("classifier.pkl","rb")
classifier =pickle.load(pickle_in)
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict if they would have survived.")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
Sex = st.selectbox("Sex", ['male', 'female'])
Age = st.slider("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)
Embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

sex_encoded =0 if Sex=="male" else 1
embarked_encoded ={'S':0,'C':1,'Q':2}[Embarked]

if st.button("Predict"):
    # Prepare input
    prediction = classifier.predict([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]])
    result = "🎉 Survived!" if prediction == 1 else "❌ Did Not Survive"
    st.subheader("Prediction Result:")
    st.success(result)

