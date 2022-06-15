# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# from app import App1
# from apps import App1
# =============================================================================


import streamlit as st
import sqlite3
dataset = pd.read_excel('dataset.csv.xlsx')
dataset

dataset.isnull().sum()

dataset.describe()

dummy = pd.get_dummies(dataset['Gender'])
dummy.head()

df=pd.concat((dataset,dummy), axis = 1)
df.head()

df=df.drop(['Gender'],axis=1)
df

df=df.drop(['female'],axis=1)
df.head()

first_column=df.pop('male')

df.insert(0, 'male', first_column)

df.head()

df.rename(columns={'male':'Gender'})

X = df.iloc[:,0:9].values
print(X)

y = df.iloc[:,-1].values
print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

df.isnull().sum()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
print(y_train_predict)


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score

y_true = [3, -0.5, 2, 7]
y_train_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_train_pred) 

print(r2_score(y_train,y_train_predict))

print(mean_absolute_error(y_train,y_train_predict))

import numpy as np
print(np.sqrt(mean_squared_error(y_train,y_train_predict)))

print(explained_variance_score(y_train, y_train_predict))

ytest_pred = model.predict(X_test)
print(ytest_pred)

print(r2_score(y_test, ytest_pred))

#plotting the observed and the predicted remifentanil flow
import matplotlib.pyplot as plt
#setting the boundaries and the parameters
plt.rcParams['figure.figsize']=(16,6)
x_ax = range(len(X_test))
#plotting
plt.plot(x_ax, y_test,label ='Observed', color ='k', linestyle ='-')
plt.plot(x_ax,ytest_pred,label = 'Predicted', color ='k', linestyle ='--')
plt.ylabel('Remifentanil Flow')
plt.xlabel('Testing sample data')
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'lower center', ncol =2, frameon = False)
plt.show()

from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()

visualizer.predict([[0,65.0,33,114,80,95,74,88,1]])

visualizer.predict([[1,74.5,35,112,75,110,62,80,1]])
visualizer.predict([[0,68.6,46,147,92,103,75,74,1]])

import pickle
filename = 'saved_model.sv'
pickle.dump(visualizer, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open('saved_model.sv', 'rb'))


loaded_model = pickle.load(open('C:/Users/KELLY OWINO/Desktop/Project2/saved_model.sv', 'rb'))

con = sqlite3.connect('data.db')
c = con.cursor()

def create_usertable():
    c.execute("""CREATE TABLE IF NOT EXISTS userstable
              (username TEXT(50), 
               password TEXT(50));""")
    con.commit()
     
    
def add_userdata(username,password):
    c.execute("""INSERT INTO userstable 
              
              VALUES (?,?)""",  (username, password))
    con.commit()
    
def login_user(username, password):
    c.execute("""SELECT * FROM userstable WHERE
              username = ? AND
              password = ?""", (username, password) )
    data = c.fetchall()
    return data

def view_all_uesrs():
    c.execute("""SELECT * FROM userstable""")
    data = c.fetchall()
    return data

def createData():
    c.execute("""CREATE TABLE IF NOT EXISTS predicted_data
              (Gender TEXT(50),
               Weight TEXT(50),
               Age (50),
               SBP (50),
               DBP (50),
               HeartRate (50),
               MAP (50),
               Oxygen_saturation (50),
               Timestamp (50));""")
    con.commit()
def addData(a,b,c,d,e,f,g,h,i):
    
           
    c.execute("""INSERT INTO predicted_data VALUES (?,?,?,?,?,?,?,?,?)""",(a,b,c,d,e,f,g,h,i))
    con.commit()
    con.close()
        
# =============================================================================
# import pyrebase
# from datatime import datetime
# from Cryptodome.Cipher import AES
# from pyrebase import initialize_app
# =============================================================================




def main():
    
    st.title('Anesthesia prediction App')
    
    Menu = ['Home', 'Login', 'Sign Up']
    choice = st.sidebar.selectbox('Menu', Menu)
    
    if choice == 'Home':
        st.subheader('Home')
        
    elif choice == 'Login':
        st.subheader('Login section')
        
        username = st.sidebar.text_input('User Name')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.checkbox('Login'):
            #if password == '1234567':
            create_usertable()
            result = login_user(username, password)
            if result:
                
                st.success('Logged in as {}'.format(username))
                
                task = st.selectbox('Task', ['Model','Profiles'])
                
                if task == 'Model':
                    
                    html_demo = """
                    <div style="background_color:tomato; padding=10px;">
                    <h3 style = "color:black; text_align:center;"background_color:tomato;">Enter Records</h3>
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
                          # if submit_button == True:
                               #createData()
                               #addData(Gender,Weight,Age,SBP,DBP,HeartRate,MAP,Oxygen_saturation,Timestamp)
                
                # if task == 'Analytics':
                #     st.subheader('Analytics')
                elif task == 'Profiles':
                    st.subheader('User Profiles')
                    user_result = view_all_uesrs()
                    clean_db = pd.DataFrame(user_result, columns = ["Username", "Password"])
                    st.dataframe(clean_db)
            else:
                st.warning('Incorrect Username or Password')
            
        
        
         
        
    elif choice == 'Sign Up':
        st.subheader('Create New Account')
        
        new_user = st.text_input('Username')
        new_password = st.text_input('Password', type='password')
        
        if st.button('SignUp'):
            create_usertable()
            add_userdata(new_user, new_password)
            st.success('You have successfully created an account')
            st.info('Go to Menu to Log in')


 # Configuraion key
# =============================================================================
#  firebaseConfig = {
#   'apiKey': "AIzaSyCjWYym3ZWAI5sl0pl2d1QjGVNYeaM5ZDU",
#   'authDomain': "final-project-streamlit.firebaseapp.com",
#   'projectId': "final-project-streamlit",
#   'databaseURL': "https://final-project-streamlit-default-rtdb.firebaseio.com/",
#   'storageBucket': "final-project-streamlit.appspot.com",
#   'messagingSenderId': "744204469180",
#   'appId': "1:744204469180:web:b1c7d707b0117122cc61b9",
#   'measurementId': "G-B5CT35NGMP"
# }
#  
#  #Firebase Authentication
#  firebase = pyrebase.initialize_app(firebaseConfig)
#  auth = firebase.auth()    
#     
# #Database
#  db = firebase.database()
#  storage = firebase.storage()
# 
# 
# st.sidebar.title('Depth and Risk prediction for Anesthesia')
# 
# #Aunthetication
# choice = st.sidebar('Login/signUp', ['Login', 'Sign Up'])
# Email = st.sidebar('Enter your email')
# Password = st.sidebar('Enter password')
# =============================================================================






    
    
    
    
    
    
        
    
    
    
    
    











if __name__ == "__main__":
    main()
    
   


