# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:55 2016

@author: bit_hammer
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy()
new_data.drop('Milk', axis = 1, inplace = True)
y = data[data.columns[1]][:,1]

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(new_data, y, test_size=0.25, random_state=42)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn import tree
regressor =  tree.DecisionTreeClassifier()
regressor = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)


# TODO: Report the score of the prediction using the testing set
from sklearn.metrics import mean_squared_error
print mean_squared_error(regressor.predict(X_train), y_train.loc['Milk'])