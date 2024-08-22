#!/usr/bin/env python
# coding: utf-8

# # Heart disease prediction
# datset information:
#
# sex: 1= male, 0= female
# cp:chest pain type(vaues 0,1,2,3)
# trestbps:resting blood pressure
# chol:serum cholestoral in mg/dl
# fbs:fasting blood sugar > 120 mg/dl
# restecg:resting electrocardiographic results (values 0,1,2)
# thalach: maximum heart rate achieved
# exang:exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# slope:the slope of the peak exercise ST segment
# ca:	number of major vessels (0-3) colored by flourosopy
# thal:(values:0,1,23)
# # # Explonatory Data Analysis

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("D:\project\heart.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# In[8]:


df.describe().T


# In[9]:


df.dtypes


# In[10]:


df['target'].value_counts()


# # selecting dependent and independent variable

# In[11]:


x=df.iloc[:,0:13]
x


# In[12]:


y=df.iloc[:,-1]
y


# # splitting dataset into training & testing data set

# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=101)


# In[14]:


x_train


# # Standardisation

# In[15]:


import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# In[16]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[17]:


x_train


# # Random Forest

# In[18]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[19]:


y_pred=rf.predict(x_test)
y_pred


# In[20]:


from sklearn.metrics import accuracy_score
rf_score=accuracy_score(y_test,y_pred)
rf_score


# # Logistic Regression

# In[21]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[22]:


y_pred=lr.predict(x_test)
y_pred


# In[23]:


lr_score=accuracy_score(y_test,y_pred)
lr_score


# # KNN

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[25]:


y_pred=knn.predict(x_test)
y_pred


# In[26]:


knn_score=accuracy_score(y_test,y_pred)
knn_score


# # comparison of the 3 algorithm

# In[27]:


plt.bar(['Random Forest','Logistic Regression','KNN'],[rf_score,lr_score,knn_score],color=['pink','orange','skyblue'])
plt.xlabel('Algorithms')
plt.ylabel('accuracy_score')
plt.show()


# # final prediction

# random forest has more accuracy compared to logistic regression and knn.so we use random forest for final prediction

# In[28]:


df.sample()


# In[31]:


sample_data=[[64,0,0,130,303,0,1,122,0,2.0,1,2,2]]
sample_data_scaled=sc.transform(sample_data)
prediction=rf.predict(sample_data_scaled)
print("final prediction : ",prediction)


# In[32]:


sample_data=[[66,1,1,160,246,0,1,120,1,0.0,1,3,1]]
sample_data_scaled=sc.transform(sample_data)
prediction=rf.predict(sample_data_scaled)
print("final prediction : ",prediction)
from joblib import dump
dump(rf,'heart_model.pkl')

