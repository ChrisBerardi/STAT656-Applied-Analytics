#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to Week 2 Assignment for Stat656 Spring 2018
Uses Python to preprocess and clean data
"""


import pandas as pd
import numpy  as np
from sklearn import preprocessing

file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Data/'
credit = pd.read_excel(file_path+"credithistory_HW2.xlsx")
# Place the number of observations in 'n_obs'
n_obs      = credit.shape[0]
# Identify Outliers and Set to Missing
# Age should be between 1 and 120
# Amount should be between 0 and 20,000
# The categorical attributes should only contain the values in the dictionary
# Check:  'savings', 'employed' and 'marital'
# Recode:  Nominal and Ordinal values
# Scale:  Interval values
# Print the mean of all interval variables and the mode frequency for 
# each nominal or ordinal variable

initial_missing = credit.isnull().sum()
feature_names = np.array(credit.columns.values)
for feature in feature_names:
    if initial_missing[feature]>(n_obs/2):
        print(feature+":\n\t%i missing: Drop this attribute." \
                  %initial_missing[feature])

# Category Values for Nominal and Binary Attributes
n_interval = 3
n_binary   = 4
n_nominal  = 20-n_interval-n_binary
n_cat      = n_binary+n_nominal
# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=Binary, 2=Nominal
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
attribute_map = {
    'age'     :[0,(1, 120),[0,0]],
    'amount'  :[0,(0, 20000),[0,0]],
    'duration':[0,(0,240),[0,0]],
    'checking':[2,(1, 2, 3, 4),[0,0]],
    'coapp'   :[2,(1,2,3),[0,0]],
    'depends' :[1,(1,2),[0,0]], 
    'employed':[2,(1,2,3,4,5),[0,0]], 
    'existcr' :[2,(1,2,3,4),[0,0]],
    'foreign' :[1,(1,2),[0,0]],
    'good_bad':[1,('good','bad'),[0,0]],
    'history' :[2,(0,1,2,3,4),[0,0]],
    'housing' :[2,(1,2,3),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job'     :[2,(1,2,3,4),[0,0]],
    'marital' :[2,(1,2,3,4),[0,0]],
    'other'   :[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings' :[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]]
}
# Initialize number missing in attribute_map
for k,v in attribute_map.items():
    for feature in feature_names:
        if feature==k:
            v[2][0] = initial_missing[feature]
            break

# Scan for outliers among interval attributes
nan_map = credit.isnull()
print(nan_map.shape)
for i in range(n_obs):
    # Check for outliers in interval attributes
    for k, v in attribute_map.items():
        if nan_map.loc[i,k]==True:
            continue
        if v[0]==0: # Interval Attribute
            l_limit = v[1][0]
            u_limit = v[1][1]
            if credit.loc[i, k]>u_limit or credit.loc[i,k]<l_limit:
                v[2][1] += 1
                credit.loc[i,k] = None
        else: # Categorical Attribute
            in_cat = False
            for cat in v[1]:
                if credit.loc[i,k]==cat:
                    in_cat=True
            if in_cat==False:
                credit.loc[i,k] = None
                v[2][1] += 1
    
print("\nNumber of missing values and outliers by attribute:")
feature_names = np.array(credit.columns.values)
for k,v in attribute_map.items():
    print(k+":\t%i missing" %v[2][0]+ "  %i outlier(s)" %v[2][1])
    
interval_attributes = []
nominal_attributes  = []
binary_attributes   = []
onehot_attributes   = []
for k,v in attribute_map.items():
    if v[0]==0:
        interval_attributes.append(k)
    else:
        if v[0]==1:
            binary_attributes.append(k)
        else:
            nominal_attributes.append(k)
            for i in range(len(v[1])):
                str = k+("%i" %i)
                onehot_attributes.append(str)
            
n_interval = len(interval_attributes)
n_binary   = len(binary_attributes)
n_nominal  = len(nominal_attributes)
n_onehot   = len(onehot_attributes)
print("\nFound %i Interval Attributes, " %n_interval, \
      "%i Binary," %n_binary,  \
      "and %i Nominal Attribute\n" %n_nominal)

# Put the interval data from the dataframe into a numpy array
interval_data = credit.as_matrix(columns=interval_attributes)
# Create the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Impute the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)

# Convert String Categorical Attribute to Numbers
# Create a dictionary with mapping of categories to numbers for attribute 'good_bad'
cat_map = {'good':1, 'bad':2}     
# Change the string categories of 'B' to numbers 
credit['good_bad'] = credit['good_bad'].map(cat_map)

# Put the nominal and binary data from the dataframe into a numpy array
nominal_data = credit.as_matrix(columns=nominal_attributes)
binary_data  = credit.as_matrix(columns=binary_attributes)
# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Impute the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
imputed_binary_data  = cat_imputer.fit_transform(binary_data)

# Encoding Interval Data by Scaling
scaler = preprocessing.StandardScaler() # Create an instance of StandardScaler()
scaler.fit(imputed_interval_data)
scaled_interval_data = scaler.transform(imputed_interval_data)

# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()

# Bring Interval and Categorial Data Together
# The Imputed Data
data_array= np.hstack((imputed_interval_data, imputed_binary_data, \
                       imputed_nominal_data))
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
credit_imputed = pd.DataFrame(data_array,columns=col)
print("\nImputed DataFrame:\n", credit_imputed[0:15])

# The Imputed and Encoded Data
data_array = np.hstack((scaled_interval_data, imputed_binary_data, hot_array))
#col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
credit_imputed_scaled = pd.DataFrame(data_array,columns=col)
print("\nImputed & Scaled DataFrame:\n", credit_imputed_scaled[0:15])

