#!/usr/bin/env python
# coding: utf-8

# In[65]:


#1. Import all the required Python Librarie


# In[3]:


import pandas as pd


# In[4]:


import numpy as np 


# In[5]:


import matplotlib.pyplot as plt


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
#so that we can view the graphs inside the notebook


# In[7]:


s1 = pd.Series(range(1,10,1))


# In[8]:


s1


# In[9]:


s3 = pd.Series({1:21, 2:13,3:45})


# In[10]:


s3


# In[11]:


s2 = pd.Series([1, 2, 3, 4], index=['p', 'q', 'r','s'], name='one')


# In[12]:


s2


# In[13]:


df1 = pd.DataFrame(s2)


# In[14]:


df1


# In[70]:


#Load the Dataset into pandas data frame


# In[15]:


df2 = pd.read_csv("/Users/janhvikarki/Desktop/Dataset/employees.csv")


# In[16]:


df2.head(10)


# In[17]:


df2.tail(3)


# In[18]:


df2.to_json('data1.json')


# In[21]:


len(df2['Team'])


# In[22]:


df2['Team'].count()


# In[24]:


df2['Salary'].mean()


# In[25]:


df2['Salary'].sum()


# In[26]:


df2['Salary'].median()


# In[27]:


df2['Salary'].std()


# In[28]:


df2['Salary'].min()


# In[29]:


df2['Salary'].describe()


# In[30]:


df2['Salary'].cumsum()


# In[64]:


# When you give the whole dataframe, then all numerical columns will be analysis
df2.mean()


# In[32]:


df2.describe()


# In[33]:


# DATA PREPROCESSING


# In[41]:


#importing pandas as pd
import pandas as pd

#making data frame from csv file
df2 = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/employees.csv")

df2.head(10)


# In[42]:


df2.describe()


# In[43]:


df2.isnull()


# In[44]:


df2.notnull()


# In[45]:


df2.isnull().sum()


# In[47]:


#Filling a null values using fillna()


# In[48]:


df2["Gender"].fillna("No Gender", inplace = True)


# In[49]:


df2.isnull().sum()


# In[50]:


# will replace  Nan value in dataframe with value -99


# In[51]:


import numpy as np
df2.replace(to_replace = np.nan, value = -99)


# In[52]:


# filling a missing value with previous ones
df2.fillna(method ='pad')


# In[53]:


df2['Salary'].fillna(int(df2['Salary'].mean()), inplace=True)


# In[54]:


#Dropping missing values using dropna()


# In[55]:


df2.dropna(axis=1)


# In[56]:


# importing pandas as pd
import pandas as pd
# Creating the dataframe
df = pd.DataFrame({"A":[12, 4, 5, None, 1],
                   "B":[None, 2, 54, 3, None],
                   "C":[20, 16, None, 3, 8],
                   "D":[14, 3, None, None, 6]})
# Print the dataframe
df


# In[58]:


df.interpolate(method = 'linear', limit_direction ='forward')


# In[59]:


#Data Formatting and Data Normalization


# In[60]:


#remove white space everywhere
text="today is Monday"
#df['Col Name'] = df['Col Name'].str.replace(‘ ‘, ‘’) 
text.replace(' ','')


# In[61]:


text=' Today'
text.lstrip()


# In[62]:


text='Today '
text.rstrip()


# In[63]:


text=' Today '
text.strip()

