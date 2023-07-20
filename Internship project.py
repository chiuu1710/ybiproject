#!/usr/bin/env python
# coding: utf-8

# # Hill and Valley Prediction with Logistic Regression

# Each dataset represent 100 points on 2d graph. 
# 1-100 : Labeled "V##". Floating point values, X-values.
# 101 : Labeled "Class". Binary {0,1} representing {valley,hill}.

# In[1]:


#Import Library
import pandas as pd


# In[2]:


import numpy as np


# In[6]:


#Import CSV as DataFrame
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Hill%20Valley%20Dataset.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


print(df.columns.tolist())


# In[12]:


df.shape


# In[13]:


df['Class'].value_counts()


# In[14]:


df.groupby('Class').mean()


# In[16]:


y = df['Class']
y.shape


# In[17]:


y


# In[19]:


x = df.drop('Class', axis=1)
x.shape


# In[20]:


x


# In[21]:


import matplotlib.pyplot as plt


# In[23]:


plt.plot(x.iloc[0,:])
plt.title('Valley');


# In[25]:


plt.plot(x.iloc[1,:])
plt.title('Hill');


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


ss = StandardScaler()


# In[28]:


x = ss.fit_transform(x)
x


# In[29]:


x.shape


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, stratify= y, random_state=2529)


# In[32]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


lr = LogisticRegression()


# In[35]:


lr.fit(x_train, y_train)


# In[36]:


y_pred = lr.predict(x_test)


# In[40]:


y_pred.shape


# In[41]:


y_pred


# In[42]:


lr.predict_proba(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix, classification_report


# In[44]:


print(confusion_matrix(y_test, y_pred))


# In[45]:


print(classification_report(y_test,y_pred))


# In[ ]:


#Future Predictions


# In[46]:


x_new = df.sample(1)


# In[47]:


x_new


# In[48]:


x_new.shape


# In[49]:


x_new = x_new.drop('Class', axis = 1)


# In[51]:


x_new


# In[53]:


x_new.shape


# In[54]:


x_new = ss.fit_transform(x_new)


# In[55]:


y_pred_new = lr.predict(x_new)


# In[56]:


y_pred_new


# In[57]:


lr.predict_proba(x_new)


# In[ ]:




