#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()


# In[3]:


loan = pd.read_csv("C:\\Users\\singh\\Downloads\\train_u6lujuX_CVtuZ9i (1).csv")


# In[4]:


loan.head()


# In[5]:


loan.tail()


# In[6]:


loan.info()


# In[10]:


loan.shape


# In[11]:


type(loan)


# In[8]:


loan.isnull().sum()


# In[12]:


df1_loan = pd.get_dummies(loan,drop_first=True)
df1_loan.head()


# In[13]:


X = df1_loan.drop(columns='Loan_Status_Y')
y = df1_loan['Loan_Status_Y']

################# Splitting into Train -Test Data #######
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify =y,random_state =3)
############### Handling/Imputing Missing values #############
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test_imp = imp_train.transform(X_test)


# In[14]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)


# In[15]:


y_pred = knn.predict(X_test_imp)


# In[16]:


confusion_matrix(y_test, y_pred)


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


accuracy_score(y_test, y_pred)


# In[19]:


from sklearn.metrics import precision_recall_fscore_support


# In[20]:


precision_recall_fscore_support(y_test, y_pred)


# In[21]:


from sklearn.metrics import precision_score


# In[22]:


precision_score(y_test, y_pred)


# In[23]:


from sklearn.metrics import recall_score


# In[24]:


recall_score(y_test, y_pred)


# In[25]:


from sklearn.metrics import f1_score


# In[26]:


f1_score(y_test, y_pred)


# In[28]:


error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test_imp)
 #print (pred_i)
 #print (1-accuracy_score(y_test, pred_i))
 error_rate.append(1-accuracy_score(y_test, pred_i))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='green', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate))+1)


# In[29]:


knn = KNeighborsClassifier(n_neighbors=29, metric='euclidean')
knn.fit(X_train, y_train)


# In[30]:


y_pred = knn.predict(X_test_imp)


# In[31]:


accuracy_score(y_test, y_pred)


# In[ ]:




