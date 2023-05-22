#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns
import pandas as pd


# In[4]:


titanic = sns.load_dataset("titanic")


# In[5]:


titanic


# In[6]:


titanic.info()


# In[7]:


x = titanic["fare"]
x


# In[8]:


#titanic.iloc[:,"fare"]


# In[9]:


titanic.describe()


# In[10]:


#First Part
#Data Cleanup
#inform us about empty fileds in column
titanic.info()


# In[11]:


#Dropping the not required columns
titanic_cleaned = titanic.drop(['pclass','embarked','deck','embark_town'],axis=1)
titanic_cleaned.head(15)


# In[12]:


titanic_cleaned.info()


# In[13]:


titanic_cleaned.isnull().sum()


# In[14]:


titanic_cleaned.corr(method='pearson')


# In[15]:


sns.histplot(data=titanic,x="fare",bins=8)


# In[16]:


sns.histplot(data=titanic,x="fare",binwidth=10)


# In[17]:


sns.histplot(data=titanic,x="fare",bins=20,binwidth=10)


# In[18]:


sns.histplot(data=titanic,x="fare",binwidth=20)


# In[19]:


sns.histplot(data=titanic,x="fare",binwidth=1)


# In[20]:


sns.histplot(data=titanic,x="fare", bins=20,binwidth=50)

