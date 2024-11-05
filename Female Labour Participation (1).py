#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


labour  = pd.read_csv('C:\\Users\\mayan\\Downloads\\flabourdataV.csv')


# # 1. Display Top 5 Rows of The Dataset

# In[5]:


labour.head()


# # 2. Check Last 5 Rows of The Dataset

# In[6]:


labour.tail()


# In[5]:


labour.shape


# In[6]:


import seaborn as sns
import numpy as np
from matplotlib  import pyplot as plt


# In[7]:


sns.set(rc = {'figure.figsize':(15,10)})
sns.barplot(x = "Year",y ="India",data = labour)


# In[8]:


sns.lmplot(x = "Year",y ="India",data = labour)


# In[9]:


print("Number of Rows",labour.shape[0])
print("Number of Columns",labour.shape[1])


# In[10]:


labour.info()


# In[11]:


labour.isnull().sum()


# # Get Overall Statistics About The Dataset

# In[12]:


labour.describe(include='all')


# In[ ]:





# In[13]:


#Using Linear Regression Algorithm


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


y = labour[['India']]


# In[16]:


x = labour[['Year']]


# In[17]:


# test size = 0.20 we are storing 20% data in test size 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) 


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


# In[215]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[216]:


x_train.head()


# In[217]:


x_test.head()


# In[218]:


y_train.head()


# In[219]:


y_test.head()


# # PREDICTION BY LINEAR REGRESSION

# In[220]:


from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm


# In[221]:


x_train,x_test,y_train,y_Test = train_test_split(x,y,test_size=0.3)


# In[222]:


lr = LinearRegression() 


# In[223]:


lr.fit(x_train,y_train)


# In[224]:


lr.predict(x_test)


# In[225]:


y_test.head()


# In[226]:


y_pred = lr.predict(x_test)


# In[227]:


y_pred[0:5]


# In[228]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[229]:


model = LinearRegression()


# In[230]:


mean_squared_error(y_test,y_pred)


# In[231]:


print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))


# In[151]:


# DONT IMPLEMENT THIS
import seaborn as sns
labour.head()

sns.lmplot(x ='Year', y ='India', data = labour)
# putting labels


# In[54]:


import statsmodels.api as sm


# In[55]:


x = sm.add_constant(x)


# In[56]:


model = sm.OLS(y, x)


# In[57]:


results = model.fit()


# In[58]:


print(results.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # END OF LINEAR REGRESSION

# # Using decison tree Algorithm 

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_Test = train_test_split(x,y,test_size=0.3)


# In[21]:


from sklearn.tree import DecisionTreeRegressor


# In[22]:


dtr = DecisionTreeRegressor()


# In[23]:


dtr.fit(x_train,y_train)


# In[24]:


y_predc =dtr.predict(x_test)


# In[25]:


y_test.head()


# In[26]:


y_predc[0:5]


# In[27]:


mean_squared_error(y_test,y_predc)


# In[169]:


print(dtr.score(y_test, y_pred))


# In[ ]:





# In[ ]:





# # Using Random Forest Algorithm

# In[170]:


y = labour[['India']]


# In[171]:


x = labour[['Year']]


# In[172]:


from sklearn.model_selection import train_test_split


# In[173]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[174]:


from sklearn.ensemble import RandomForestRegressor


# In[175]:


rfg  = RandomForestRegressor()


# In[176]:


rfg.fit(x_train,y_train)


# In[177]:


y_pred = rfg.predict(x_test)


# In[178]:


y_test.head(),y_pred[0:5]


# In[179]:


mean_squared_error(y_test,y_pred)


# In[188]:


from sklearn import metrics


# In[189]:


rfg.score(y_test, y_pred)


# In[ ]:





# In[6]:


pip install pandoc


# In[ ]:




