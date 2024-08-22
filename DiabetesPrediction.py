#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# # Explonatory Data Analysis

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("D:\project\diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.sample(5)


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.info()


# df.describe().T

# In[10]:


df.dtypes


# In[11]:


df.isna().sum()


# In[12]:


df['Outcome'].value_counts()


# # selecting independent and dependent variable

# In[13]:


x=df.iloc[:,0:8]
x


# In[14]:


y=df.iloc[:,-1]
y


# # splting data set

# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)


# In[16]:


x_train


# # standardisation 

# In[17]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[18]:


x_train


# # Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[20]:


y_pred=lr.predict(x_test)
y_pred


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


lr_score=accuracy_score(y_test,y_pred)


# In[23]:


lr_score


# # Random Forest

# In[24]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[25]:


y_pred=rf.predict(x_test)
y_pred


# In[26]:


rf_score=accuracy_score(y_test,y_pred)
rf_score


# # KNN

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[28]:


y_pred=knn.predict(x_test)
y_pred


# In[29]:


knn_score=accuracy_score(y_test,y_pred)
knn_score


# 
# # comparison of 3 algorithms

# In[30]:


plt.bar(['Logistic Regression','Random Forest','KNN'],[lr_score,rf_score,knn_score],color=['pink','orange','skyblue'])
plt.xlabel("algorithms")
plt.ylabel("accuracy score")
plt.show()


# # final prediction

# we are taking logistic regression for prediction

# In[31]:


df.sample(5)


# In[32]:


import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


# In[33]:


sample_data=[[1,111,86,19,0,30.1,0.143,23]]
sample_data_scaled=sc.transform(sample_data)
prediction=lr.predict(sample_data_scaled)
print("final prediction : ",prediction)


# In[34]:


sample_data=[[3,173,78,39,185,33.8,0.970,31]]
sample_data_scaled=sc.transform(sample_data)
prediction=lr.predict(sample_data_scaled)
print("final prediction :",prediction)
from joblib import dump
dump(lr,'diabetes_model.pkl')

