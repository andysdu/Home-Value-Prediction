### This is the python code to train machine learning models for zillow's home value data ###

# Import modules
import os
import gc
import numpy as np
import pandas as pd 

# Load data
train = pd.read_csv('./input/train_2016.csv')
prop = pd.read_csv('./input/properties_2016.csv')

# Merge data
train_xy = train.merge(prop, on = 'parcelid', how = 'left')

# Impute missing values (Many machine learning algorithms will deal with missing values automatically, so optional sometimes)
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', copy=True, axis=0, verbose=0)
imp.fit(train_xy)
train_data = imp.transform(train_xy)
train_data = pd.DataFrame(train_data, columns = train_xy.columns)

# Convert string/text variable into numbers, and convert dtypes into np.float32
# For categorical variables, one can use several different ways to encode them, e.g. Label encoding, One hot encoding
# , Label binarizer etc. Depend on the number of categories and real situation, choose appropriate methods.

for c, dtype in zip(train_data.columns, train_data.dtypes):
	if dtype == object:
		train_data[c] = (train_data[c] == True)
		train_data[c] = train_data[c].astype(np.float32)
	else:
		train_data[c] = train_data[c].astype(np.float32)

X = train_data.drop(labels=['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis = 1)
y = train_data['logerror'].values

# train using random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
model_rf = RandomForestRegressor(n_estimator=100, max_depth=10, criterion='mae', oob_score=True, n_jobs=-1)
model_rf.fit(X, y)







