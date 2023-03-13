#!/usr/bin/env python
# coding: utf-8

# In[3]:


from joblib import dump,load 
import numpy as np
import pandas as pd ;


# In[4]:


model=load("Dragon.joblib")


# In[5]:


features1=[-0.43942006,  0.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]

features2=np.array(features1)
features=np.reshape(features2,(-1,1))



model.predict(features)


# In[ ]:




