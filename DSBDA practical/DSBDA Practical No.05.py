#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


# In[4]:


df = pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/Social_Network_Ads.csv")


# In[5]:


df.head(10)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


X = df.iloc[:,[2,3]].values
y = df.iloc[:,4].values


# In[10]:


X


# In[11]:


y


# In[12]:


#Split the dataset into train and test


# In[13]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.25,random_state=0)


# In[14]:


#Preprocessing
#Standard Scalar


# In[15]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


X_train


# In[17]:


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[18]:


#Prediction


# In[19]:


y_pred = classifier.predict(X_test)


# In[20]:


y_pred


# In[21]:


#Confusion Matrix


# In[22]:


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test , y_pred)


# In[23]:


cm


# In[24]:


c1_report = classification_report(y_test,y_pred)


# In[25]:


c1_report


# In[26]:


tp , fn ,fp , tn = confusion_matrix(y_test,y_pred,labels=[0,1]).reshape(-1)
print('Outcome values : \n' , tp , fn , fp ,tn)


# In[27]:


accuracy_cm = (tp+tn)/(tp+fp+tn+fn)
precision_cm = tp/(tp+fp)
recall_cm = tp/(tp+fn)
f1_score = 2/((1/recall_cm)+(1/precision_cm))


# In[28]:


print("Accuracy : ",accuracy_cm)
print("Precision : ",precision_cm) 
print("Recall : ",recall_cm) 
print("F1-Score : ",f1_score)

