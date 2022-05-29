#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Dhaka Rent.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


plt.scatter(df.area , df.rent, color='red')


# In[8]:


plt.scatter(df.area , df.rent, color='red',marker='+') # https://matplotlib.org/stable/api/markers_api.html


# In[9]:


plt.scatter(df.area , df.rent, color='red',marker='+') 
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')


# In[10]:


plt.scatter(df.area , df.rent, color='red',marker='+') 
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')
plt.title('Rent in Bangladesh')


# In[11]:


df.area.mean()


# In[12]:


df.rent.mean()


# In[13]:


plt.scatter(df['area'] , df['rent'],marker='+') 
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')
plt.title('Rent in Bangladesh')


# In[14]:


plt.figure(figsize=(12,8))
plt.scatter(df['area'] , df['rent'],marker='+') 
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')
plt.title('Rent in Bangladesh')


# In[15]:


len(df.area)


# In[16]:


df.shape


# In[17]:


x = df[['area']]  # x = df.drop('rent', axis=1) # ind var
y = df['rent'] #dep variable


# # Split the dataset 

# In[18]:


df.head(10)


# In[19]:


from sklearn.model_selection import train_test_split as tts


# In[20]:


xtrain , xtest , ytrain, ytest = tts(x,y,train_size=.70,random_state=1) #test_size=.30
#train, test = tts(df,train_size=.70,random_state=1) #test_size=.30


# In[21]:


xtrain.shape


# In[22]:


xtrain.head() #52, 17


# In[23]:


ytrain.head()


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lr = LinearRegression() # creating object for linear model


# In[26]:


lr.fit(xtrain ,ytrain) #train the ML model ; xtrain=x ; ytrain=y


# In[27]:


xtrain.shape


# In[28]:


lr.coef_


# In[29]:


lr.intercept_


# In[30]:


m = lr.coef_
c = lr.intercept_


# In[31]:


x = 1500
y = (m * x) + c


# In[32]:


y


# In[34]:


lr.predict([[1500]])


# In[35]:


df


# In[37]:


lr.predict(df[['area']])


# In[38]:


df['Predicted Rent'] = lr.predict(df[['area']]) # y=mx+c


# In[40]:


df.head()


# In[43]:


testing = lr.predict(xtest) # ytest


# In[44]:


testing


# In[45]:


ytest


# In[46]:


xtest['pred'] = lr.predict(xtest)


# In[48]:


xtest.head()


# In[51]:


xtest = xtest[['area']]


# In[52]:


from sklearn.metrics import mean_squared_error # mse
pred = lr.predict(xtest)
mse = mean_squared_error(ytest , pred) #testing error


# In[53]:


mse


# In[54]:


from sklearn.metrics import mean_absolute_error


# In[55]:


mean_absolute_error(ytest, pred) # MAE


# In[57]:


lr.predict([[3000]])


# In[58]:


lr.predict([[30000]])


# In[64]:


plt.figure(figsize=(12,8))
plt.scatter(df['area'] , df['rent'],marker='+') 
plt.plot(xtest , lr.predict(xtest), color='red')
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')
plt.title('Rent in Bangladesh')


# In[63]:


lr.predict([[3500]])


# In[65]:


plt.figure(figsize=(12,8)) 
plt.scatter(df['area'] , df['rent'],marker='+') 
plt.plot(df[['area']] , lr.predict(df[['area']]), color='red')
plt.xlabel('Area in SqFt')
plt.ylabel('Rent in BDT')
plt.title('Rent in Bangladesh')


# In[66]:


lr.score(xtest, ytest) # R Squared Value


# In[68]:


lr.score(df[['area']], df.rent) # R Squared Value


# In[69]:


lr.score(xtrain, ytrain) # R Squared Value


# In[ ]:




