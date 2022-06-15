#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_excel('dataset.csv.xlsx')
dataset


# In[3]:


dataset.isnull().sum()


# In[4]:


dataset.describe()


# In[5]:


dummy = pd.get_dummies(dataset['Gender'])
dummy.head()


# In[6]:


df=pd.concat((dataset,dummy), axis = 1)
df.head()


# In[7]:


df=df.drop(['Gender'],axis=1)
df


# In[8]:


df=df.drop(['female'],axis=1)
df.head()


# In[9]:


first_column=df.pop('male')


# In[10]:


df.insert(0, 'male', first_column)


# In[11]:


df.head()


# In[12]:


df.rename(columns={'male':'Gender'})


# In[13]:


X = df.iloc[:,0:9].values
print(X)


# In[14]:


y = df.iloc[:,-1].values
print(y)


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


# In[16]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[17]:


df.isnull().sum()


# In[18]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[19]:


y_train_predict = model.predict(X_train)
print(y_train_predict)


# In[20]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score


# In[21]:


y_true = [3, -0.5, 2, 7]
y_train_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_train_pred) 


# In[22]:


print(r2_score(y_train,y_train_predict))


# In[23]:


print(mean_absolute_error(y_train,y_train_predict))


# In[24]:


import numpy as np
print(np.sqrt(mean_squared_error(y_train,y_train_predict)))


# In[25]:


print(explained_variance_score(y_train, y_train_predict))


# In[26]:


ytest_pred = model.predict(X_test)
print(ytest_pred)


# In[27]:


print(r2_score(y_test, ytest_pred))


# In[28]:


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


# In[29]:


from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()


# In[30]:


from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()


# In[31]:


visualizer.predict([[0,65.0,33,114,80,95,74,88,1]])


# In[32]:


visualizer.predict([[1,74.5,35,112,75,110,62,80,1]])


# In[33]:


visualizer.predict([[0,68.6,46,147,92,103,75,74,1]])


# In[34]:


visualizer.predict([[1,80,56,134,90,68,70,67,1]])


# In[35]:


visualizer.predict([[1,100,76,154,110,88,90,87,1]])


# In[36]:


visualizer.predict([[0,80,56,134,90,68,70,67,1]])


# In[37]:


visualizer.predict([[1,80,56,134,90,68,70,67,2]])


# In[38]:


visualizer.predict([[1,80,56,134,90,68,70,67,3]])


# In[39]:


visualizer.predict([[1,80,56,134,90,68,70,67,0]])


# In[40]:


visualizer.predict([[1,60,36,114,70,48,60,67,1]])


# In[41]:


visualizer.predict([[1,80,56,134,90,68,70,67,0]])


# In[42]:


visualizer.predict([[1,80,56,134,90,68,70,67,0.5]])


# In[43]:


visualizer.predict([[1,80,56,134,90,68,70,67,1]])


# In[44]:


visualizer.predict([[1,72,34,104,78,75,75,87,1]])


# In[45]:


visualizer.predict([[1,40,16,114,81,90,66,87,1]])


# In[46]:


visualizer.predict([[1,68.6,46,147,92,103,75,74,1]])


# In[47]:


visualizer.predict([[0,50.0,32,145,90,93,75,87,2]])


# In[48]:


visualizer.predict([[1,70.0,52,165,110,113,95,107,2]])


# In[49]:


visualizer.predict([[0,70.0,52,165,110,113,95,107,2]])


# In[50]:


visualizer.predict([[0,70.0,52,165,110,113,95,107,1]])


# In[51]:


visualizer.predict([[0,70.0,52,165,110,113,95,107,3]])


# In[74]:


import pickle
filename = 'saved_model.sv'
pickle.dump(visualizer, open(filename, 'wb'))


# In[75]:


#loading the saved model
loaded_model = pickle.load(open('saved_model.sv', 'rb'))


# In[52]:


#Decision Tree algorithm


# In[53]:


from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)


# In[57]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score


# In[58]:


y_train_pred = model2.predict(X_train)


# In[59]:


print(r2_score(y_train,y_train_pred))


# In[60]:


print(mean_absolute_error(y_train,y_train_pred))


# In[61]:


print(mean_squared_error(y_train, y_train_pred))


# In[62]:


y_test_pred = model2.predict(X_test)


# In[63]:


print(r2_score(y_test, y_test_pred))


# In[65]:


#plotting the observed and the predicted Remifentanil Flow
import matplotlib.pyplot as plt
#setting the boundaries and the parameters
plt.rcParams['figure.figsize']=(16,6)
x_ax = range(len(X_test))
#plotting
plt.plot(x_ax, y_test,label ='Observed', color ='k', linestyle ='-')
plt.plot(x_ax,y_test_pred,label = 'Predicted', color ='k', linestyle ='--')
plt.ylabel('Remifentanil Flow')
plt.xlabel('Testing sample data')
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'lower center', ncol =2, frameon = False)
plt.show()


# In[66]:


from yellowbrick.regressor import PredictionError
visualizer2 = PredictionError(model2)
visualizer2.fit(X_train, y_train)
visualizer2.score(X_test, y_test)
visualizer2.poof()


# In[67]:


from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model2)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()


# In[68]:


visualizer2.predict([[0,70.0,52,165,110,113,95,107,3]])


# In[69]:


visualizer.predict([[0,50.0,32,146,89,94,75,88,2]])


# In[70]:


visualizer2.predict([[0,50.0,32,145,90,93,75,87,2]])


# In[71]:


visualizer2.predict([[1,50.0,32,144,89,94,75,90,1]])


# In[72]:


visualizer2.predict([[1,68.6,46,147,92,103,75,74,1]])


# In[73]:


visualizer2.predict([[1,88.6,66,167,112,123,95,94,1]])


# In[ ]:




