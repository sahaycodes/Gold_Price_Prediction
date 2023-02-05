#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 


# In[20]:


# data collection and processing 
#loading the data on a Pandas Dataframe
gold_data=pd.read_csv(r"D:Users\User\Downloads\gld_price.csv")


# In[21]:


# print first 5 rows of the Dataframe 
gold_data.head()


# In[22]:


#print last 5 rows 
gold_data.tail()


# In[23]:


gold_data.shape


# In[24]:


#getting some basic info about the data
gold_data.info()


# In[25]:


# CHECKING THE NO.OF MISSING VALUES 
gold_data.isnull().sum()


# In[26]:


# getting statistical measures of the data 
gold_data.describe()


# In[27]:


# positive correlation and # negative correlation 
correlation = gold_data.corr()


# In[29]:


# constructing a heatmap to understand the correlation 
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar = True, square = True , fmt = '.1f' , annot = True , annot_kws={'size:8'},cmap='Blues')


# In[30]:


#correlation values of GLD 
print(correlation['GLD'])


# In[31]:


# checking the distribution of the GLD price
sns.distplot(gold_data['GLD'],color = 'green')


# In[32]:


# splitting the features and Target 
X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']


# In[33]:


print(X)


# In[34]:


print(Y)


# In[35]:


#Splitting into Training and Testing Data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 2)


# In[36]:


# Model Training 
# Random Forest Regressor 
regressor = RandomForestRegressor(n_estimators=100)


# In[37]:


#training the model 
regressor.fit(X_train,Y_train)


# In[38]:


#Model evaluation # prediction on Test Data
test_data_prediction = regressor.predict(X_test)


# In[39]:


print(test_data_prediction)


# In[40]:


# R square error 
error_score = metrics.r2_score(Y_test,test_data_prediction)
print(" R squared error : ",error_score)


# In[41]:


#Compare the actual and predicted values in a plot 
Y_test = list(Y_test)


# In[42]:


plt.plot(Y_test,color='blue',label='Actual Value')
plt.plot(test_data_prediction,color='green',label='Predicted Value')
plt.title('Actual Price vs Predicted Price ')
plt.xlabel('no.of values')
plt.ylabel('GLD price')
plt.legend()
plt.show()


# In[ ]:




