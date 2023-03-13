#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing= pd.read_csv("Book1.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50,figsize=(20,15))


# In[10]:


#learning purpose
def split_train_test(data,test_ratio):
    np.random.seed(42)
    suffled=np.random.permutation(len(data))
    print(suffled)
    test_set_size=int(len(data))*test_ratio
    test_indices=suffled[:int(test_set_size)]
    train_indices=suffled[int(test_set_size):]
    return data.iloc[train_indices],data.iloc[test_indices]
    


# In[11]:


train_set,test_set=split_train_test(housing,0.2)
print(f'rows in tain set :{len(train_set)}\nrows in train set:{len(test_set)}')


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set =train_test_split(housing, test_size=0.2, random_state=42)
print(f'rows in tain set :{len(train_set)}\nrows in train set:{len(test_set)}')


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set= housing.loc[train_index]
    strat_test_set= housing.loc[test_index]


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


housing=strat_train_set.copy()


# # #looking for insights
# 

# In[17]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[19]:


housing.plot(kind='scatter', x='RM',y='MEDV',alpha=0.8)


# In[20]:


housing.plot(kind="scatter",x='LSTAT',y='MEDV',alpha=0.8)


# # tryingout attributes combination

# In[21]:


housing['TAXRM']=housing["TAX"]/housing["RM"]
housing.head()


# In[22]:


housing.plot(kind='scatter', x='TAXRM',y='MEDV',alpha=0.8)


# In[23]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing=strat_train_set.drop('MEDV',axis=1)
housing_label=strat_train_set['MEDV'].copy()


# ## missing attributes

# In[25]:


#to solve this problem:
   # 1.get rid of missing data points
        #housing.dropna(subset['RM'])
   # 2.get rid of whole attribute
        #housing.drop('RM',axis=1)
    #3.set the value to some values(0 or mean or median)
       # median=housing['RM'].median()
       # housing['RM'].fillna(median)
#note thata the original datatfram reamin unchanged 
    #for doing option3 we have sklearn classes
#             from sklearn.impute import SimpleImputer
#             imputer=SimpleImputer(strategy='median')
#             imputer.fit(housing)
#               imputer.statistics_
#               x=imputer.transform(housing)  it gives numpy arryto convert it
#               housing_tr=pd.DataFrame(x,housing.columns)  


# # scikit learn design

# In[26]:


# Primarily, three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. 
#     It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). 
#     It also has a convenience function called fit_transform () which fits and then transforms.
# 3. Predictors - LinearRegression model is an example of predictor. 
#     fit() and predict () are two common functions. It also gives score() function which will evaluate the predictions.


# ## Feature scaling

# In[27]:


# Primarily, two types of feature scaling methods: 
#     1. Min-max scaling (Normalization)
#         (value - min)/(max - min)

#     Sklearn provides a class called MinMaxScaler for this

# 2. Standardization

# (value - mean)/std

# Sklearn provides a class called StandardScaler for this


# ## Creating pipeline

# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    #..........add as many as you wnat
    ('std_scaler',StandardScaler()),
])


# In[29]:


housing_num_tr=my_pipeline.fit_transform(housing) 


# In[30]:


housing_num_tr.shape


# ## Selecting model

# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_label)


# In[32]:


some_data=housing.iloc[:5]
some_label=housing_label.iloc[:5]
preparedata=my_pipeline.transform(some_data)
model.predict(preparedata)


# In[33]:


list(some_label)


# ## evaluating model

# In[34]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_label,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print(lin_mse)


# ## using better evaluation techniques: cross validation

# In[35]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_label,cv=10,scoring="neg_mean_squared_error")
rmse_score=np.sqrt(-scores)


# In[36]:


rmse_score


# In[37]:


def print_scores(score):
    print("score:",score)
    print("mean:",score.mean())
    print("Standard Devialtion:",score.std())


# In[38]:


print_scores(rmse_score)


# ## saving the model

# In[44]:


from joblib import dump,load
dump(model,'Dragon.joblib') 


# ## Testing the model

# In[49]:


x_test=strat_test_set.drop(["MEDV"],axis=1)
y_test=strat_test_set["MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_prediction=model.predict(x_test_prepared)
final_mes=mean_squared_error(y_test,final_prediction)
final_rmse=np.sqrt(final_mes)
print(final_prediction,list(y_test))


# In[47]:


final_rmse


# In[51]:


preparedata[0]


# In[ ]:




