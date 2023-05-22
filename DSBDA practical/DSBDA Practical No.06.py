#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv("/Users/shreyaspeherkar/Desktop/Dataset/iris.csv")
df.head(10)


# In[4]:


X=df.iloc[:,0:4]
y=df.iloc[:,-1]
y  


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)
X_test


# In[7]:


from sklearn.preprocessing import LabelEncoder 
la_object = LabelEncoder()
y = la_object.fit_transform(y)
y


# In[8]:


from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()
model.fit(X_train, y_train)


# In[9]:


y_predicted = model.predict(X_test)
y_predicted


# In[10]:


model.score(X_test,y_test)


# In[11]:


from sklearn.metrics import confusion_matrix,classification_report 
cm = confusion_matrix(y_test, y_predicted)


# In[12]:


cm


# In[14]:


# classification report for precision, recall f1-score and accuracy
cl_report=classification_report(y_test,y_predicted)
cl_report


# In[15]:


# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for pl otting.
cm_df = pd.DataFrame(cm,
                     index = ['SETOSA','VERSICOLR','VIRGINICA'],
                     columns = ['SETOSA','VERSICOLR','VIRGINICA'])


# In[18]:


#Plotting the confusion matrix
import seaborn as sns 
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True) 
plt.title('Confusion Matrix') 
plt.ylabel('Actal Values') 
plt.xlabel('Predicted Values')
plt.show()


# In[20]:


def accuracy_cm(tp,fn,fp,tn): 
    return (tp+tn)/(tp+fp+tn+fn)

def precision_cm(tp,fn,fp,tn): 
    return tp/(tp+fp)

def recall_cm(tp,fn,fp,tn): 
    return tp/(tp+fn)

def f1_score(tp,fn,fp,tn):
    return (2/((1/recall_cm(tp,fn,fp,tn))+precision_cm(tp,fn,fp,tn)))

def error_rate_cm(tp,fn,fp,tn): 
    return 1-accuracy_cm(tp,fn,fp,tn)


# In[22]:


#For Virginica
tp = cm[2][2]
fn = cm[2][0]+cm[2][1]
fp = cm[0][2]+cm[1][2]
tn = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
print("For Virginica \n")
print("Accuracy: ",accuracy_cm(tp,fn,fp,tn))
print("Precision: ",precision_cm(tp,fn,fp,tn))
print("Recall: ",recall_cm(tp,fn,fp,tn))
print("F1-Score: ",f1_score(tp,fn,fp,tn))
print("Error rate : ",error_rate_cm(tp,fn,fp,tn))

