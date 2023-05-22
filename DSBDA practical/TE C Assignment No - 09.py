#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
titanic = sns.load_dataset("titanic")


# In[3]:


titanic


# In[4]:


titanic.head()


# In[5]:


titanic.info()


# In[6]:


titanic.describe()


# In[7]:


#Custom Columns with all rows
titanic.loc[:,["survived","alive"]]


# In[8]:


#Now Plot boxplot
sns.boxplot(x="sex",y="age",data=titanic)


# In[9]:


sns.boxplot(x="sex",y="age",data=titanic,hue="survived")

