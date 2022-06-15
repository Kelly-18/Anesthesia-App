# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:37:47 2022

@author: KELLY OWINO
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/KELLY OWINO/Desktop/Project2/saved_model.sv', 'rb'))




def main():
    st.title('Risk Prediction App')
    html_demo = """
    <div style="background_color:tomato; padding=10px;">
    <h3 style = "color:black; text_align:center;"background_color:tomato;">Depth and risk prediction App for anesthesia</h3>
    </div>
    
    """
    st.markdown(html_demo, unsafe_allow_html=True)
    
    #Input variables
    with st.form(key= 'form', clear_on_submit=True):
           Gender = st.text_input('Enter 1 for male, 0 for female')
           Weight = st.text_input('Weight')
           Age = st.text_input('Age')
           SBP = st.text_input('Systolic Blood Pressure(mm/Hg)')
           DBP = st.text_input('Diastolic Blood Pressure(mm/Hg)')
           HeartRate = st.text_input('Heart Rate(bpm)')
           MAP = st.text_input('Mean Arteriole Pressure(mm/Hg)')
           Oxygen_saturation = st.text_input('Oxygen Saturation (%)')
           Timestamp = st.text_input('Timestamp')
           submit_button = st.form_submit_button('Submit')
           
           #Prediction code
           if submit_button:
               prediction = loaded_model.predict([[Gender,Weight,Age,SBP,DBP,HeartRate,MAP,Oxygen_saturation,Timestamp]])
               result = round(prediction[0],2)
               st.success('Predicted output {} '.format(result))
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
         main()
        