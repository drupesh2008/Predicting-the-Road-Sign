
# coding: utf-8

# In[2]:

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[5]:

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[6]:

train.head()


# In[7]:

test.head()


# In[8]:

train['DetectedCamera'].value_counts()


# In[9]:

#encode as integer
mapping = {'Front':0, 'Right':1, 'Left':2, 'Rear':3}
train = train.replace({'DetectedCamera':mapping})
test = test.replace({'DetectedCamera':mapping})


# In[10]:

#renaming column
train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)


# In[11]:

#encode Target Variable based on sample submission file
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'Target':mapping})


# In[12]:

#target variable
y_train = train['Target']
test_id = test['Id']


# In[13]:

#drop columns
train.drop(['Target','Id'], inplace=True, axis=1)
test.drop('Id',inplace=True,axis=1)


# In[14]:

#train model
clf = RandomForestClassifier(n_estimators=500,max_features=3,min_samples_split=5,oob_score=True)
clf.fit(train, y_train)


# In[15]:

#predict on test data
pred = clf.predict_proba(test)


# In[18]:

#write submission file and submit
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_rf.csv", index=False) #99.8XXX





