# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:31:11 2023

@author: harin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open("C:/Users/harin/Pictures/ML 03/crop_prediction.pkl","rb"))

def crop_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = loaded_model.predict(id_reshaped)
    
    return prediction[0]
   
    
def main():
    
    st.title("Crop Prediction App")

    STATE = st.number_input("Enter the State:", min_value=0, max_value=25)
    
    st.title("Crop Prediction App")
 
    N_SOIL = st.number_input("Enter the Nitrogen content of Soil:")
    P_SOIL = st.number_input("Enter the Phosphorous content of Soil:")
    K_SOIL = st.number_input("Enter the Pottasium content of Soil:")
    ph = st.number_input("Enter the pH of Soil:", min_value=0, max_value=14)
    TEMPERATURE = st.number_input("Enter the Temperature of the State:", min_value=0, max_value=75)
    HUMIDITY = st.number_input("Enter the Relative humidity (%):", min_value=0, max_value=100)
    RAINFALL = st.number_input("Enter the Average rainfall (in mm):", min_value=0, max_value=1000)
    CROP_PRICE = st.number_input("Enter the Price of the crop:", min_value=0)
    
    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = crop_prediction([STATE, N_SOIL, P_SOIL, K_SOIL, ph, TEMPERATURE, HUMIDITY, RAINFALL, CROP_PRICE])
        st.write("The predicted crop is:", diagnosis)
        
if __name__=='__main__':
    main()
     