#!/usr/bin/env python
# coding: utf-8

# # kidney disease prediction
# data set information
#
# id: Identification or patient ID.
# age: Age of the patient.
# bp: Blood pressure of the patient.
# sg: Specific gravity of urine.
# al: Albumin level in urine.
# su: Sugar level in urine.
# rbc: Red blood cell count.
# pc: Pus cell count.
# pcc: Pus cell clumps.
# ba: Bacteria present in urine.
# bgr: Blood glucose random.
# bu: Blood urea.
# sc: Serum creatinine.
# sod: Sodium level in blood.
# pot: Potassium level in blood.
# hemo: Hemoglobin level.
# pcv: Packed cell volume.
# wc: White blood cell count.
# rc: Red blood cell count.
# htn: Hypertension (Yes/No).
# dm: Diabetes mellitus (Yes/No).
# cad: Coronary artery disease (Yes/No).
# appet: Appetite (Good/Poor).
# pe: Pedal edema (Yes/No).
# ane: Anemia (Yes/No).
# classification: The target variable indicating the presence or absence of kidney disease.
# # importing libraries

# In[254]:


import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# In[255]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Explonatory Data Analysis

# In[256]:


df=pd.read_csv("D:\project\kidney_disease.csv")


# In[257]:


df.head()


# In[258]:


df.tail()


# In[259]:


df.sample(5)


# In[260]:


df.shape


# In[261]:


df.info()


# In[262]:


df.describe().T


# In[263]:


df.dtypes


# In[264]:


# There is some ambugity present in the columns dm and cad we have to remove that.


# In[265]:


df['dm'].value_counts()


# In[266]:


df['cad'].value_counts()


# In[267]:


df['classification'].value_counts()


# In[268]:


# replace incorrect values


# In[269]:


df['dm'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
df['cad'] = df['cad'].replace(to_replace = '\tno', value='no')
df['classification'] = df['classification'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})


# In[270]:


df.isna().sum()


# In[271]:


df.drop(['id','rbc','sod','pot','wc','rc'],axis=1,inplace=True)


# In[272]:


df.columns


# In[273]:


df.dtypes


# In[ ]:





# In[274]:


for i in df.columns:
    if df[i].dtype!='object':
        sns.histplot(x=df[i],kde=True)
        plt.title(f'Histogram for {i}')
        plt.show()


# In[275]:


# age,bp,sg,al,su,bgr,bu,sc,hemo have skewness in its histogram
# so we  took its median values to fill their null values


# In[276]:


m=['age','bp','sg','al','su','bgr','bu','sc','hemo']
for i in m:
    df[i]=df[i].fillna(df[i].median())


# In[277]:


# pc,pcc,ba,pcv,htn,dm,cad,appet,pe,ane are object columns
# so it is treated with its mode values


# In[278]:


n=['pc','pcc','ba','pcv','htn','dm','cad','appet','pe','ane']
for i in n:
    df[i]=df[i].fillna(df[i].mode()[0])


# In[279]:


df.isna().sum()

#
# # # Label Encoding
# pc:normal=1,abnormal=0
# pcc:notpresent=0,present=1
# ba:notpresent=0,present=1
# htn:yes=1,no=0
# dm:yes=1, No=0
# cad:yes=1,No=0
# appet:good=0,poor=1
# pe:no=0,yes=1
# ane:no=0,yes=1
# classification:1=ckd,0=notckd
# In[280]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[281]:


m=['pc','pcc','ba','pcv','htn','dm','cad','appet','pe','ane','classification']
for i in m:
    df[i]=le.fit_transform(df[i])


# In[282]:


df.dtypes


# In[283]:


df['classification'].value_counts()


# In[284]:


df.dtypes


# In[285]:


sns.heatmap(df.corr())


# # selecting X and Y

# In[286]:


x=df.iloc[:,0:19]
x


# In[287]:


y=df.iloc[:,-1]
y


# # splitting the data set into train set and test set

# In[288]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=101,test_size=0.2)


# In[289]:


x_train


# # Multinomial Naive bayes

# In[290]:


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(x_train,y_train)


# In[291]:


y_pred=clf.predict(x_test)
y_pred


# In[292]:


from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score,f1_score
m_score=accuracy_score(y_test,y_pred)
m_score


# In[293]:


print('precision_score = ',precision_score(y_test,y_pred,average='macro'))
print('recall_score =',recall_score(y_test,y_pred,average='macro'))
print('f1_score = ',f1_score(y_test,y_pred,average='macro'))
print('classification_report \n',classification_report(y_test,y_pred))


# # Gaussian Naive Bayes

# In[294]:


from sklearn.naive_bayes import GaussianNB
g_nb=GaussianNB()
g_nb.fit(x_train,y_train)


# In[295]:


y_pred=g_nb.predict(x_test)
y_pred


# In[296]:


g_score=accuracy_score(y_test,y_pred)
g_score


# In[297]:


print('precision_score = ',precision_score(y_test,y_pred,average='macro'))
print('recall_score =',recall_score(y_test,y_pred,average='macro'))
print('f1_score =',f1_score(y_test,y_pred,average='macro'))
print('classification_report \n ',classification_report(y_test,y_pred))


# # Bernoulli Naive Bayes

# In[298]:


from sklearn.naive_bayes import BernoulliNB
b_nb=BernoulliNB()
b_nb.fit(x_train,y_train)


# In[299]:


y_pred=b_nb.predict(x_test)
y_pred


# In[300]:


b_score=accuracy_score(y_test,y_pred)
b_score


# In[301]:


print('precision_score = ',precision_score(y_test,y_pred,average='macro'))
print('recall_score =',recall_score(y_test,y_pred,average='macro'))
print('f1_score = ',f1_score(y_test,y_pred,average='macro'))
print('classification_report \n',classification_report(y_test,y_pred))


# In[302]:


plt.bar(['Multinomial NB','Gaussian NB','Bernoulli NB'],[m_score,g_score,b_score],color=['pink','orange','skyblue'])
plt.xlabel('naive bayes')
plt.ylabel('accuracy_score')
plt.show()


# # final prediction

# we take gaussion naive bayes for prediction

# In[303]:


df.sample(5)


# In[304]:

sample_data=[[55.0,70.0,1.010,0.0,2.0,1,0,0,220.0,68.0,2.8,8.70,15,1,1,0,0,0,1]]
prediction =g_nb.predict(sample_data)
print("Final Prediction :",prediction)


# In[305]:

sample_data=[[34.0,60.0,1.020,0.0,0.0,1,0,0,91.0,49.0,1.2,13.50,36,0,0,0,0,0,0]]
prediction1 =g_nb.predict(sample_data)
print("final Prediction",prediction1)

from joblib import dump
dump(g_nb,'kidney_model.pkl')


