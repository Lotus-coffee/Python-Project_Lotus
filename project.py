#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this is the project 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#part I: Explore the data set
train = pd.read_csv('train.csv',index_col= 0, low_memory= False)
test = pd.read_csv('test.csv',index_col= 0, low_memory= False)
train.head()


# In[3]:


#check the number of the train dataset
len(train)


# In[4]:


#explore the features
print(train.columns)
len(train.columns)


# In[5]:


#check the none value
train.isna().sum()


# In[6]:


#explore the target(loan approval),loan_status = 0 means 
plt.hist(data = train, x ='loan_status')


# In[7]:


# Check how many applicants can get loan approval in the training data
loan_approval_test = train.groupby('loan_status').size()
la_percent_test = loan_approval_test.to_frame(name='number_of_loan').reset_index()

# Calculate total approved and not approved loans
total_test = la_percent_test.number_of_loan.sum()

# Calculate the percentage for each loan status
la_percent_test['Percentage'] = la_percent_test['number_of_loan'] / total_test * 100

print(la_percent_test)


# In[8]:


# Ensure `id` is a column in both train and test DataFrames
train = train.reset_index() if 'id' not in train.columns else train
test = test.reset_index() if 'id' not in test.columns else test


# In[9]:


# Separate features and target variable in the training set
X = train.drop(['loan_status', 'id'], axis=1, errors='ignore')
y = train['loan_status']
X_test = test.drop(['id'], axis=1, errors='ignore')


# In[10]:


# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# In[11]:


# Preprocessing pipeline
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[12]:


# Create a pipeline with the preprocessor and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[13]:


# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Train the model
model.fit(X_train, y_train)


# In[15]:


# Validate the model (evaluate with AUC-ROC)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Get the probability of the positive class
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print("Validation AUC-ROC Score:", auc_score)


# In[16]:


# Train on full training data and predict on the test set
model.fit(X, y)
y_test_pred = model.predict(X_test)


# In[17]:


# Prepare the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'loan_status': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")


# In[18]:


# Check how many applicants can get loan approval in the test data
loan_approval_test = submission.groupby('loan_status').size()
la_percent_test = loan_approval_test.to_frame(name='number_of_loan').reset_index()

# Calculate total approved and not approved loans
total_test = la_percent_test.number_of_loan.sum()

# Calculate the percentage for each loan status
la_percent_test['Percentage'] = la_percent_test['number_of_loan'] / total_test * 100

print(la_percent_test)


# In[ ]:





# In[ ]:





# In[ ]:




