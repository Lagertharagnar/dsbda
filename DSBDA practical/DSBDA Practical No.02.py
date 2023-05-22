#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


student = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/StudentsPerformance.csv")


# In[7]:


student.info()


# In[8]:


student.isnull().sum()


# In[9]:


#filling missing value by mean
student['math_score'].fillna(int(student['math_score'].mean()), inplace=True)


# In[10]:


student.isnull().sum()


# In[11]:


# filling a missing value with previous ones
student['reading_score'].fillna(method ='pad',inplace=True)


# In[12]:


student.isnull().sum()


# In[13]:


#filling missing value by median
student['writing_score'].fillna(int(student['writing_score'].median()), inplace=True)


# In[14]:


student.isnull().sum()


# In[15]:


#Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques to deal with them.


# In[18]:


from numpy.random import seed 
from numpy.random import randn 
from numpy import mean
from numpy import std
seed(1)
#univariate dataset- single variable/ attribute #multivariate detaset-muliple variables/attributes
data=5*randn(10000)+50
print('mean=%.3f stdv=%.3f' %(mean(data), std(data)))


# In[19]:


#Standard Deviation Method 
data_mean = mean(data)
data_std = std(data)
cut_off = data_std * 3
lower = data_mean - cut_off
upper = data_mean + cut_off


# In[21]:


outliers=[x for x in data if x<lower or x > upper]
outliers


# In[22]:


import matplotlib.pyplot as plt
plt.plot(data)


# In[23]:


outliers_removed=[x for x in data if x>=lower and x<=upper] 
plt.plot(outliers_removed)


# In[24]:


#Interquartile Range Method
from numpy.lib.function_base import percentile 
q25=percentile(data,25) 
q75=percentile(data,75)
IQR=q75-q25
cut_off_IQR= IQR * 2
lower=q25-cut_off_IQR
upper= q75 +cut_off_IQR


# In[25]:


outliers_IQR = [x for x in data if x < lower or x > upper] 
outliers_IQR


# In[26]:


outliers_removed=[x for x in data if x>=lower and x<=upper] 
plt.plot(outliers_removed)


# In[27]:


#Apply data transformations on at least one of the variables.The purpose of this transformation should be one of the
#following reasons: to change the scale for better understanding of the variable, to convert a non-linear relation
#into a linear one, or to decrease the skewness and convert the distribution into a normal distribution.


# In[29]:


from sklearn.preprocessing import MinMaxScaler


# In[30]:


mms = MinMaxScaler()


# In[32]:


student[['math_score','reading_score','writing_score']] = mms.fit_transform(student[['math_score','reading_score','writing_score']])


# In[33]:


student.head()

