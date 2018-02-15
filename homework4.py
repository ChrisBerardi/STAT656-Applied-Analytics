# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to Week 4 Assignment for Stat656 Spring 2018
Uses Python to preprocess and clean data and fit various logistic models to data
"""

import pandas as pd
import numpy  as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg

#Read File
file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Data/'
df = pd.read_excel(file_path+"credithistory_HW2.xlsx")
#Create a second data frame with only the attributes selected from 
# SAS Dataminer
sas_at = ['checking','history','duration','coapp','installp','savings','marital','good_bad']
sas = df.loc[:,sas_at]

# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=Binary, 2=Nominal
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
#
# Leave off the purpose attribute as it is over half missing at the combination 
# of numbers and letters does not play nice with the classes
attribute_map = {
    'age'     :[0,(1, 120),[0,0]],
    'amount'  :[0,(0, 20000),[0,0]],
    'duration':[0,(0,100),[0,0]],
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

sas_map = {
    'duration':[0,(0,100),[0,0]],
    'checking':[2,(1, 2, 3, 4),[0,0]],
    'coapp'   :[2,(1,2,3),[0,0]],
    'history' :[2,(0,1,2,3,4),[0,0]],
    'good_bad':[1,('good','bad'),[0,0]],
    'savings' :[2,(1,2,3,4,5),[0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'marital' :[2,(1,2,3,4),[0,0]]
        }
#Replace, impute, and encode using SAS encoding
rep_imp_enc = ReplaceImputeEncode(data_map=attribute_map, display=True)
encoded_df = rep_imp_enc.fit_transform(df)

# Regression requires numpy arrays containing all numeric values
y = np.asarray(encoded_df['good_bad']) 
# Drop the target, 'object'.  Axis=1 indicates the drop is for a column.
X = np.asarray(encoded_df.drop('good_bad', axis=1)) 

#Fit a logistic regression model, use k=4 fold cross validation
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
logistic = LogisticRegression()
logistic.fit(X,y)


log_tts = LogisticRegression()
log_tts.fit(X_train, y_train)
#Display Selection metrics
logreg.display_binary_split_metrics(log_tts, X_train, y_train, \
                                    X_validate, y_validate)

#Cross Validation Results
lgr_4_scores = cross_val_score(logistic, X_train, y_train, cv=4)
print("\nAccuracy Scores by Fold: ", lgr_4_scores)
print("Accuracy Mean:      %.4f" %lgr_4_scores.mean())
print("Accuracy Std. Dev.: %.4f" %lgr_4_scores.std())


#Do it with SAS
rep_imp_enc = ReplaceImputeEncode(data_map=sas_map, display=True)
encoded_sas = rep_imp_enc.fit_transform(sas)

X_sas = np.asarray(encoded_sas.drop('good_bad', axis=1)) 

X_train_sas, X_validate_sas, y_train_sas, y_validate_sas = \
            train_test_split(X_sas,y,test_size = 0.3, random_state=7)
            
logistic = LogisticRegression()
logistic.fit(X_sas,y)

log_tts = LogisticRegression()
log_tts.fit(X_train_sas, y_train_sas)

logreg.display_binary_split_metrics(log_tts, X_train_sas, y_train_sas, \
                                    X_validate_sas, y_validate_sas)

lgr_4_scores = cross_val_score(logistic, X_train_sas, y_train_sas, cv=4)
print("\nAccuracy Scores by Fold: ", lgr_4_scores)
print("Accuracy Mean:      %.4f" %lgr_4_scores.mean())
print("Accuracy Std. Dev.: %.4f" %lgr_4_scores.std())

#Print out interval means and selected nominal frequency tables
intervals = ['age','amount','duration']
nominals = ['employed','marital','savings']

print('\n Interval Attribute Min\n', pd.DataFrame.min(encoded_df.loc[:,intervals]))
print('\n Interval Attribute Max\n', pd.DataFrame.max(encoded_df.loc[:,intervals]))
print('\n Interval Attribute Means\n', pd.DataFrame.mean(encoded_df.loc[:,intervals]))

#Redo encoding with one-hot to make it easier to generate a frequency table
noms = df[nominals]
noms_map = {
    'employed':[2,(1,2,3,4,5),[0,0]], 
    'marital' :[2,(1,2,3,4),[0,0]],
    'savings' :[2,(1,2,3,4,5),[0,0]]
}
noms_rie = ReplaceImputeEncode(data_map=noms_map, nominal_encoding='one-hot')
encoded_noms = noms_rie.fit_transform(noms)
noms_num = [4,3,4]

print("\nCounts of Nominal Attributes:\n")
for i in range(10):
    print('\n',encoded_noms.iloc[:,i].value_counts())