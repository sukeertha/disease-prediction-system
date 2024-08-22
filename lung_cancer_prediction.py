#!/usr/bin/env python
# coding: utf-8

# # LUNG CANCER PREDICTION
# dataset information
#
# Gender: M(male), F(female)
# Age: Age of the patient
# Smoking: YES=2 , NO=1.
# Yellow fingers: YES=2 , NO=1.
# Anxiety: YES=2 , NO=1.
# Peer_pressure: YES=2 , NO=1.
# Chronic Disease: YES=2 , NO=1.
# Fatigue: YES=2 , NO=1.
# Allergy: YES=2 , NO=1.
# Wheezing: YES=2 , NO=1.
# Alcohol consuming : YES=2 , NO=1.
# Coughing: YES=2 , NO=1.
# Shortness of Breath: YES=2 , NO=1.
# Swallowing Difficulty: YES=2 , NO=1.
# Chest pain: YES=2 , NO=1.
# Lung Cancer: YES , NO.importing libraries
# In[26]:


import pandas as pd
import matplotlib.pyplot as plt


# # Explonatory Data Analysis

# In[27]:


df=pd.read_csv("survey lung cancer.csv")


# In[28]:


df.head()


# In[29]:


df.tail()


# In[30]:


df.shape


# In[31]:


df.info()


# In[32]:


df.describe().T


# In[33]:


df.isnull().sum()




# # # Label Encoding
# GENDER: male=1,female=0
# LUNG_CANCER:yes=1,No=0
# # In[35]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])


# In[36]:


df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])


# In[38]:


df.dtypes


# # Selecting x and y

# In[39]:


x=df.iloc[:,0:15]
x


# In[40]:


y=df.iloc[:,-1]
y


# # Splitting the dataset

# In[41]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)


# In[42]:


x_train


# # Standardization

# In[43]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[44]:


x_train


# # Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[46]:


y_pred=lr.predict(x_test)
y_pred


# In[47]:


from sklearn.metrics import accuracy_score
lr_score=accuracy_score(y_test,y_pred)
lr_score


# # KNN

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[49]:


y_pred=lr.predict(x_test)
y_pred


# In[50]:


knn_score=accuracy_score(y_test,y_pred)
knn_score


# # Random Forest

# In[51]:


from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()
rc.fit(x_train,y_train)
from joblib import dump
dump(lr,'lung_mode.pkl')
# In[52]:


y_pred=rc.predict(x_test)
y_pred


# In[53]:


rc_score=accuracy_score(y_test,y_pred)
rc_score


# # Comparing 3 algorithms

# In[54]:


plt.figure(figsize=(8, 6))
plt.bar(['Random Forest','KNN','Logistic Regression'],[rc_score,knn_score,lr_score],color=['pink','orange','skyblue'])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy_Score")
plt.show()


# # Final prediction

# we are using logistic regression for final prediction

# In[55]:


df.sample(5)


# In[56]:


sample_data = [[0,67,2,2,2,2,1,2,1,1,1,1,1,1,1]]  # Sample data for prediction
sample_data_scaled = sc.transform(sample_data)
prediction = lr.predict(sample_data_scaled)
print("Final Prediction:",prediction)


# In[57]:


sample_data = [[0,63,1,1,1,1,2,2,1,1,1,1,2,1,1]]  # Sample data for prediction
sample_data_scaled = sc.transform(sample_data)
prediction = lr.predict(sample_data_scaled)
print("Final Prediction:",prediction)

