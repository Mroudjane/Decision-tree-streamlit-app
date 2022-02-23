#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:10:54 2022

@author: massivaroudjane
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




header= st.container()
dataset= st.container()
features=st.container()
model_training= st.container()

@st.cache()
def get_data(filename):
    taxi_data= pd.read_csv(filename)
    
    return taxi_data

with header:
    st.title("Welcome to this projet app")
    st.text("In this project we will work on the transactions in NYC taxi dataset.")
    
with dataset:
    st.header("NYC taxi dataset")
    st.text("This section will contain all the data used to create the model. The datset source is : url")
    
    st.subheader("Pick-up location ID distribution.")
    taxi_data= get_data("/Users/massivaroudjane/Desktop/Decision tree streamlit app/Data/data.csv")
    st.write(taxi_data.head())
    pulocation_dist= taxi_data['PULocationID'].value_counts().head(50)
    st.bar_chart(pulocation_dist)
    
with features:
    st.header("New features created")
    st.text("I will explain in detail the features I have come up with and discuss why are they relevant to this project.")
    st.markdown('* **First feature**: This is why I created this feature and this is the logic behind it')
    st.markdown('* **second feature** : This is why I created this feature and this is the logic behind it')


with model_training:
    st.header("Model training step")
    st.text("In this section you will be able to change the hyperparameters of the model and see there impact on the model performance." )
    
    sel_col, disp_col= st.columns(2)
    
    max_depth= sel_col.slider("Max depth of the model", min_value=10, max_value=100, value= 20, step=10)
    n_estimators= sel_col.selectbox("Number of trees", options=[100, 200, 300, 'No limits'], index=0)
    
    sel_col.text("Here is a list of features to choose from :")
    sel_col.write(taxi_data.columns)
    input_feature= sel_col.text_input("Which feature should be used in the model ?", 'PULocationID')
    
    #model
    
    if n_estimators=='No limits':
        regressor= RandomForestRegressor(max_depth= max_depth)
    
    else:
         regressor= RandomForestRegressor(max_depth= max_depth, n_estimators= n_estimators)
         
         
    X= taxi_data[[input_feature]]
    y= taxi_data[["trip_distance"]]
    
    regressor.fit(X,y)
    prediction= regressor.predict(y)
    
    
    disp_col.subheader("Mean absolute error of the model is :")
    disp_col.write(mean_absolute_error(y, prediction))
    
    disp_col.subheader("Mean squared error of the model is :")
    disp_col.write(mean_squared_error (y, prediction))
    
    disp_col.subheader("R suqared score of the model is :")
    disp_col.write(r2_score(y, prediction))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    