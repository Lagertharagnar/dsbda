#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing Necessary Libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[4]:


# Reading Data
data = pd.read_csv('/Users/shreyaspeherkar/Desktop/Dataset/HousingData.csv')
print(data.shape)
data.head()


# In[5]:


# Collecting X and Y
X = data['DIS'].values
Y = data['MEDV'].values


# In[7]:


Y
#Y=mX+b m= difference in y coordinate/difference in x coordinate b= y-intercept


# In[8]:


# Calculating coefficient
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
# Total number of values
n = len(X)


# In[9]:


n


# In[10]:


# Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x) 
# m(b1) and c(bo)
# Printing coefficients 
print("Coefficients")
print("m=",b1)
print("c=",b0)


# In[11]:


# Plotting Values and Regression Line
max_x = np.max(X)
min_x = np.min(X)

# Calculating line values x and y

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Ploting Line
#plt.plot(x, y, color='#58b970', label='Regression Line')
plt.plot(x, y, color='green', label='Regression Line')
# Ploting Scatter Points
#plt.scatter(X, Y, c='#ef5423', label='Scatter Plot') 
plt.scatter(X, Y, c='red', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()


# In[12]:


# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score")
print(r2)


# In[13]:


#using scikit-learn


# In[14]:


# Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[16]:


data=pd.read_csv('/Users/shreyaspeherkar/Desktop/Dataset/HousingData.csv')
X = data.iloc[:,7].values.reshape(-1,1) #converts it into numpy array
Y = data.iloc[:,13].values.reshape(-1,1) 
linear_regressor=LinearRegression() # create obect for class 
linear_regressor.fit(X,Y) # perform linear regression 
y_pred=linear_regressor.predict(X) # make prediction


# In[17]:


plt.scatter(X,Y)
plt.plot(X,y_pred, color='red')


# In[18]:


# The coefficients
print("Coefficients: \n", linear_regressor.coef_)


# In[19]:


from sklearn.metrics import mean_squared_error, r2_score 
print("Coefficient of determination: %.2f" % r2_score(Y, y_pred))

