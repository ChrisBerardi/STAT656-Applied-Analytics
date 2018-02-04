# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to Week 3 Assignment for Stat656 Spring 2018
Uses Python to preprocess and clean data and fit various linear models to data
"""

import pandas as pd
import numpy  as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Data/'
df = pd.read_csv(file_path+"DiamondsWMissing.csv")
n_obs      = df.shape[0]
print("\n********** Data Preprocessing ***********")
print("Data contains %i observations & %i columns.\n" %df.shape)
#df['Price'] = df['Price'].str.replace('$','')
#df['Price'] = df['Price'].astype(dtype='float')
initial_missing = df.isnull().sum()
feature_names = np.array(df.columns.values)
for feature in feature_names:
    if initial_missing[feature]>(n_obs/2):
        print(feature+":\n\t%i missing: Drop this attribute." \
                  %initial_missing[feature])

# Category Values for Nominal and Binary Attributes
#Treat obs as a interval variable to improve processing speed
n_interval = 8
n_binary   = 0
n_nominal  = 3
n_cat      = n_binary+n_nominal

#Map attributes 0: interval, 1 binary, 2 nominal
attribute_map = {
    'obs':[0,(1,53940,1),[0,0]],
    'Carat':[0,(0.2, 5.5),[0,0]],
    'cut':[2,('Fair', 'Good', 'Ideal', 'Premium', 'Very Good'),[0,0]],
    'color':[2,('D', 'E', 'F', 'G', 'H', 'I', 'J',),[0,0]],
    'clarity':[2,('IF', 'I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'),[0,0]],
    'depth':[0,(40, 80),[0,0]], 
    'table':[0,(40, 100), [0,0]],
    'x':[0,(0,11),[0,0]],
    'y':[0,(0,60),[0,0]],
    'z':[0,(0,32),[0,0]],
    'price':[0, (300, 20000), [0,0]]}

# Initialize number missing in attribute_map
for k,v in attribute_map.items():
    for feature in feature_names:
        if feature==k:
            v[2][0] = initial_missing[feature]
            break

# Scan for outliers among interval attributes
nan_map = df.isnull()

for i in range(n_obs):
    # Check for outliers in interval attributes
    for k, v in attribute_map.items():
        if nan_map.loc[i,k]==True:
            continue
        if v[0]==0: # Interval Attribute
            l_limit = v[1][0]
            u_limit = v[1][1]
            if df.loc[i,k]>u_limit or df.loc[i,k]<l_limit:
                v[2][1] += 1
                df.loc[i,k] = None
        else: # Categorical Attribute
            in_cat = False
            for cat in v[1]:
                if df.loc[i,k]==cat:
                    in_cat=True
            if in_cat==False:
                df.loc[i,k] = None
                v[2][1] += 1
    
print("\nNumber of missing values and outliers by attribute:")
feature_names = np.array(df.columns.values)
for k,v in attribute_map.items():
    print(k+":\t%i missing" %v[2][0]+ "  %i outlier(s)" %v[2][1])

#One hot encoding
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
print("\nFound %i Interval Attribute(s), " %n_interval, \
      "%i Binary," %n_binary,  \
      "and %i Nominal Attribute(s)\n" %n_nominal)

#print("Original DataFrame:\n", df[0:5])
# Put the interval data from the dataframe into a numpy array
interval_data = df.as_matrix(columns=interval_attributes)
# Create the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Impute the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)

# Convert String Categorical Attribute to Numbers

cat_map = {'Fair':1, 'Good':2, 'Ideal':3, 'Premium':4, 'Very Good':5}     
df['cut'] = df['cut'].map(cat_map)

cat_map = {'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7}     
df['color'] = df['color'].map(cat_map)

cat_map = {'I1': 1, 'IF':2, 'SI1':3, 'SI2':4, 'VS1':5, 'VS2':6, 'VVS1':7, 'VVS2':8}     
df['clarity'] = df['clarity'].map(cat_map)

# Put the nominal and binary data from the dataframe into a numpy array
nominal_data = df.as_matrix(columns=nominal_attributes)
# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Impute the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)

# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()

# Bring Interval and Categorial Data Together
# The Imputed Data
data_array= np.hstack((imputed_interval_data, imputed_nominal_data))
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
df_imputed = pd.DataFrame(data_array,columns=col)
#print("\nImputed DataFrame:\n", df_imputed[0:5])

# The Imputed and Encoded Data BUT NOT SCALED
data_array = np.hstack((imputed_interval_data, hot_array))
#col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
df_imputed_scaled = pd.DataFrame(data_array,columns=col)
df_imputed_scaled = df_imputed_scaled.drop(['cut0', 'color0', 'clarity0'], axis=1)
#print("\nImputed & Scaled DataFrame:\n", df_imputed_scaled[0:5])

#Split data into 70/30 split
price = df_imputed_scaled[['price']]
#Make list of columns to remove target
cols = df_imputed_scaled.columns.tolist()
#Remove target
cols.remove('price')
design = df_imputed_scaled[cols]
#Split the data
X_train, X_test, y_train, y_test = \
model_selection.train_test_split(design, price, test_size=0.3)

#Initialize the linear model
lr = LinearRegression()
#Fit the model with the training data
model = lr.fit(X_train, y_train)
#Make predictions
pred = lr.predict(X_test)
pred_df = pd.DataFrame(pred,columns=['Prediction'])

print('\nActual Price','\nMean',y_test.mean(),'\nMinimum', y_test.min(),\
      '\nMaximum', y_test.max())
print('\nPredicted Price','\nMean',pred.mean(),'\nMinimum', pred.min(),\
      '\nMaximum', pred.max())
print(X_test.iloc[0:15,1:8], '\n',X_test.iloc[0:15,8:16], '\n,',X_test.iloc[0:15,16:24], '\n', y_test[0:15], pred_df[0:15])
