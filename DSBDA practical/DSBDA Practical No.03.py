#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


student = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/StudentsPerformance.csv")


# In[7]:


student.info()


# In[8]:


student.describe()


# In[9]:


student.head()


# In[10]:


male_female = student.groupby('gender')['gender'].count()
print(male_female)


# In[11]:


student.test_preparation_course.unique()


# In[12]:


mean_math = student.groupby('gender').math_score.mean()


# In[13]:


print(mean_math)


# In[14]:


mean_math_test_preparation = student.groupby(['gender','test_preparation_course']).math_score.mean()


# In[15]:


print(mean_math_test_preparation)


# In[16]:


mean_math_test_preparation = student.groupby(['gender','race/ethnicity']).math_score.mean()


# In[17]:


print(mean_math_test_preparation)


# In[18]:


student.math_score.unique()


# In[19]:


#Group by of a Single Column and Apply the describe() Method on a Single Column


# In[20]:


print(student.groupby('gender').math_score.describe())


# In[21]:


groups = pd.cut(student['math_score'],bins=4)
groups


# In[22]:


groups = pd.cut(student['math_score'],bins=5)
groups


# In[23]:


student.groupby(groups)['math_score'].count()


# In[24]:


pd.crosstab(groups, student['gender'])


# In[25]:


#Write a Python program to display some basic statistical details like percentile, mean, standard deviation etc. 
#of the species of ‘Iris-setosa’, ‘Iris- versicolor’ and 'Iris-versicolor’ of iris.csv dataset.


# In[26]:


import statistics as st


# In[27]:


data = [1,2,3,4,5,6]


# In[28]:


st.mean(data)


# In[29]:


st.median(data)


# In[30]:


#Will show error as data is having no unique modal value
st.mode(data)


# In[31]:


data1 = [1,2,7,5,4,7,8,2,1,7]
st.mode(data1)


# In[32]:


#Variance
st.variance(data1)


# In[33]:


import pandas as pd
df = pd.DataFrame(data1)


# In[34]:


df.mean()


# In[35]:


df.mode()


# In[36]:


df.median()


# In[37]:


#using California housing csv file
df1 = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/housing.csv")
df1


# In[38]:


df1.mean()


# In[39]:


df1["households"].mean()


# In[40]:


df1["households"].median()


# In[41]:


df1["households"].mode()


# In[42]:


df1["households"].var()


# In[43]:


st.stdev(df1["households"])


# In[44]:


#Descriptive Statistics on IRIS dataset


# In[45]:


import pandas as pd
data = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/iris.csv")
print('Iris-setosa')


# In[46]:


setosa = data['species'] == 'Iris-setosa' 
print(data[setosa].describe())


# In[47]:


print('\nIris-versicolor')
setosa = data['species'] == 'Iris-versicolor' 
print(data[setosa].describe())


# In[48]:


print('\nIris-virginica')
setosa = data['species'] == 'Iris-virginica' 
print(data[setosa].describe())

