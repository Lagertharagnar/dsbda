#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
iris = sns.load_dataset("iris")


# In[3]:


iris


# In[4]:


iris.info()


# In[5]:


iris.describe()


# In[6]:


type(iris.sepal_length)


# In[7]:


iris.sepal_length.dtype


# In[8]:


iris.sepal_width.dtype


# In[9]:


iris.petal_length.dtype


# In[10]:


iris.petal_width.dtype


# In[11]:


iris.species.dtype


# In[12]:


import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,2,figsize=(10,6)) 
sns.histplot(iris["sepal_length"],ax=axes[0,0]) 
sns.histplot(iris["sepal_width"],ax=axes[0,1]) 
sns.histplot(iris["petal_length"],ax=axes[1,0]) 
sns.histplot(iris["petal_width"],ax=axes[1,1])


# In[13]:


#For boxplot
fig,axes = plt.subplots(2,2,figsize=(16,10))
sns.boxplot(x="species",y="sepal_length",data=iris,ax=axes[0,0])
sns.boxplot(x="species",y="sepal_width",data=iris,ax=axes[0,1])
sns.boxplot(x="species",y="petal_length",data=iris,ax=axes[1,0])
sns.boxplot(x="species",y="petal_width",data=iris,ax=axes[1,1])

