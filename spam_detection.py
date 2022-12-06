#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\umar1\Desktop\jupyterproject\spam.csv")
df


# In[3]:


check_nan=df['Category'].isnull().values.any()


# In[4]:


check_nan


# In[34]:


df['spam']=df.Category.apply(lambda x: 1 if x=='spam' else 0)


# In[35]:


df


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df.Message,df.spam,test_size=0.2)


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_train_new = cv.fit_transform(X_train.values)


# In[38]:


X_train_new.toarray()[:2]


# In[39]:


from sklearn.naive_bayes import MultinomialNB
model =MultinomialNB()
model.fit(X_train_new,Y_train)


# In[40]:


email=['Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!']
email_new =cv.transform(email)
email_new


# In[41]:


model.predict(email_new)


# In[44]:


X_test_count = cv.transform(X_test)
model.score(X_test_count, Y_test)


# In[ ]:




