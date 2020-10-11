#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
df1 = pandas.read_csv("C:\\Users\\mac\\Desktop\\test_set.csv")
dataset = df1.values
data=dataset[:,5:21]
label=dataset[:,4]
df2 = pandas.read_csv("C:\\Users\\mac\\Desktop\\new_data.csv")
dataset2 = df2.values
data2=dataset2[:,5:21]
label2=dataset2[:,4]
train_set,test_set,train_label,test_label=train_test_split(data, label,test_size=0.2)
start1 = time.clock()
from sklearn.tree import DecisionTreeClassifier
cls = DecisionTreeClassifier(max_depth=20)
cls.fit(train_set, train_label)
from sklearn.metrics import accuracy_score
pred=cls.predict(data2)
print('The predicted accuracy of the decision tree only:',accuracy_score(label2,pred))
end1 = time.clock()
print('Running time of decision tree only:',end1 - start1)


# In[70]:


start2 = time.clock()
from sklearn.neighbors import KNeighborsClassifier
cls2= KNeighborsClassifier(n_neighbors=5)
cls2.fit(train_set, train_label)
pred2=cls2.predict(data2)
print('The predicted accuracy of the k-NN only:',accuracy_score(label2,pred2))
end2 = time.clock()
print('Running time of decision tree with bagging:',end2 - start2)


# In[68]:


start3 = time.clock()
clfb1 = BaggingClassifier(base_estimator=DecisionTreeClassifier()).fit(train_set, train_label)
predictb = clfb.predict(data2)
print('The predicted accuracy of the decision tree with bagging',accuracy_score(label2,predictb))
end3 = time.clock()
print('Running time of decision tree with bagging:',end3 - start3)


# In[69]:


start4 = time.clock()
clfb2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier()).fit(train_set, train_label)
y_2 = clfb2.predict(data2)
print('The predicted accuracy of the decision tree with Adaboosting',accuracy_score(label2,y_2))
end4 = time.clock()
print('Running time of decision tree with Adaboosting:',end4 - start4)


# In[71]:


clfb2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier()).fit(train_set, train_label)


# In[74]:


clfb1 = BaggingClassifier(base_estimator=DecisionTreeClassifier())
clfb1.fit(train_set, train_label)


# In[75]:


clfb2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
clfb2.fit(train_set, train_label)


# In[ ]:




