import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

# Loading the trained model
loaded_model = pickle.load(open("C:/Users/jeyasri/Downloads/CROP/crop_prediction.pkl","rb"))

def crop_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = loaded_model.predict(id_reshaped)
    
    return prediction[0]
   
    
def main():
    image = Image.open("C:/Users/jeyasri/Downloads/CROP/crop_image.jpg")
    st.image(image, width=700)
    
    st.title("Crop Prediction App")

    STATE = st.number_input("Enter the State:", min_value=0, max_value=25)
    st.write(pd.DataFrame({
   'STATE': ['Andaman and Nicobar','Tamil Nadu','Andhra Pradesh','Assam','Chattisgarh','Goa','Gujarat','Haryana','Himachal Pradesh','Jammu and Kashmir','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Nagaland','Odisha','Pondicherry','Punjab','Rajasthan','Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttrakhand','West Bengal'],
   'EQUIVALENT VALUE': [0,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
   }))
    N_SOIL = st.slider('Nitrogen content of Soil', 0, 150)
    st.write(N_SOIL)
    P_SOIL = st.slider('Phosphorous content of Soil', 0, 150)
    st.write(P_SOIL)
    K_SOIL = st.slider('Pottasium content of Soil', 0, 150)
    st.write(K_SOIL)
    ph = st.slider('pH of Soil', 0, 14)
    st.write(ph)
    TEMPERATURE = st.slider('Temperature of the State', 0, 75)
    st.write(TEMPERATURE)
    HUMIDITY = st.slider('Relative humidity (%)', 0, 100)
    st.write(HUMIDITY)
    RAINFALL = st.slider('Enter the Average rainfall (in mm)', 0, 1000)
    st.write(RAINFALL)
    CROP_PRICE = st.number_input("Enter the Price of the crop:", min_value=0)
    
    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = crop_prediction([STATE, N_SOIL, P_SOIL, K_SOIL, ph, TEMPERATURE, HUMIDITY, RAINFALL, CROP_PRICE])
        st.write("The predicted crop is:", diagnosis)
        
if __name__=='__main__':
    main()
     
