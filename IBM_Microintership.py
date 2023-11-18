#!/usr/bin/env python
# coding: utf-8

# # #Disease prediction from various symptoms 

# BY Nikhil Mahajan 
# 

# In[145]:


#import necessary libries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[ ]:





# In[146]:


#import dataset into df variavble for processing 
df=pd.read_csv("Training.csv")


# In[147]:


df.head(100)


# In[ ]:





# In[148]:


df.tail(20)


# In[149]:


#to view datasize 
df.shape


# In[150]:


# to view statical overview of datset 
df.describe()


# In[151]:


df.info()


# In[152]:


df.columns


# In[153]:


df.corr()


# In[154]:


#cheak missing values  # as per obervation dataset is clean 
df.isnull().sum()


# In[ ]:





# In[155]:


df['itching'].plot.hist()


# In[156]:


#to cheak how mant varible are present inside progniosis 
df['prognosis'].value_counts()


# In[ ]:





# In[ ]:





# In[157]:


df['prognosis'].value_counts().plot(kind= 'bar')


# In[158]:


df.dtypes


# In[159]:


df.plot.scatter('itching','prognosis')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[160]:


y=df['prognosis']


# In[161]:


x = df.drop("prognosis", axis= "columns")


# In[162]:


# Step  : train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=2529)


# In[163]:


# check shape of train and test sample
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[176]:


from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

%%time
classifier_rf.fit(x_train, y_train)


# In[ ]:





# In[175]:


classifier_rf.predict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




