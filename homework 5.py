# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to Week 5 Assignment for Stat656 Spring 2018
Uses Python to it decision tree models and evaluate using cross validation
"""

import pandas as pd
import numpy  as np
import graphviz as gv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn import tree
from Class_regression import logreg

file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Data/'
df = pd.read_excel(file_path+"CreditHistory_Clean.xlsx")

# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=Binary, 2=Nominal
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
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
    'telephon':[1,(1,2),[0,0]],
}

#Replace, impute, and encode using SAS encoding
rep_imp_enc = ReplaceImputeEncode(data_map=attribute_map, display=True)
encoded_df = rep_imp_enc.fit_transform(df)

# Regression requires numpy arrays containing all numeric values
y = np.asarray(encoded_df['good_bad']) 
# Drop the target, 'object'.  Axis=1 indicates the drop is for a column.
X = np.asarray(encoded_df.drop('good_bad', axis=1)) 

#Fit the tree models, with different max_depth use k=10 fold cross validation
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)

tree_5  = tree.DecisionTreeClassifier(max_depth=5)
tree_5 = tree_5.fit(X_train,y_train)

#Cross Validation Results
#Depth 5
tree_5_scores = cross_val_score(tree_5, X_train, y_train, cv=10, \
                                scoring='accuracy')
print("\nAccuracy Scores by Fold Depth 5: ", tree_5_scores)
print("Accuracy Mean:      %.4f" %tree_5_scores.mean())
print("Accuracy Std. Dev.: %.4f" %tree_5_scores.std())

tree_5_scores = cross_val_score(tree_5, X_train, y_train, cv=10, \
                                scoring='recall')
print("\nRecall Scores by Fold Depth 5: ", tree_5_scores)
print("Recall Mean:      %.4f" %tree_5_scores.mean())
print("Recall Std. Dev.: %.4f" %tree_5_scores.std())

tree_5_scores = cross_val_score(tree_5, X_train, y_train, cv=10, \
                                scoring='precision')
print("\nPrecision Scores by Fold Depth 5: ", tree_5_scores)
print("Precision Mean:      %.4f" %tree_5_scores.mean())
print("Precision Std. Dev.: %.4f" %tree_5_scores.std())

tree_5_scores = cross_val_score(tree_5, X_train, y_train, cv=10, \
                                scoring='f1')
print("\nF1 Scores by Fold Depth 5: ", tree_5_scores)
print("F1 Mean:      %.4f" %tree_5_scores.mean())
print("F1 Std. Dev.: %.4f" %tree_5_scores.std())

#Depth 6
tree_6  = tree.DecisionTreeClassifier(max_depth=6)
tree_6 = tree_6.fit(X_train,y_train)
#Cross Validation Results
tree_6_scores = cross_val_score(tree_6, X_train, y_train, cv=10)
print("\nAccuracy Scores by Fold Depth 6: ", tree_6_scores)
print("Accuracy Mean:      %.4f" %tree_6_scores.mean())
print("Accuracy Std. Dev.: %.4f" %tree_6_scores.std())

tree65_scores = cross_val_score(tree_6, X_train, y_train, cv=10, \
                                scoring='recall')
print("\nRecall Scores by Fold Depth 6: ", tree_6_scores)
print("Recall Mean:      %.4f" %tree_6_scores.mean())
print("Recall Std. Dev.: %.4f" %tree_6_scores.std())

tree_6_scores = cross_val_score(tree_6, X_train, y_train, cv=10, \
                                scoring='precision')
print("\nPrecision Scores by Fold Depth 5: ", tree_6_scores)
print("Precision Mean:      %.4f" %tree_6_scores.mean())
print("Precision Std. Dev.: %.4f" %tree_6_scores.std())

tree_6_scores = cross_val_score(tree_6, X_train, y_train, cv=10, \
                                scoring='f1')
print("\nF1 Scores by Fold Depth 5: ", tree_6_scores)
print("F1 Mean:      %.4f" %tree_6_scores.mean())
print("F1 Std. Dev.: %.4f" %tree_6_scores.std())


#Depth 8
tree_8  = tree.DecisionTreeClassifier(max_depth=8)
tree_8 = tree_8.fit(X_train,y_train)
#Cross Validation Results
tree_8_scores = cross_val_score(tree_8, X_train, y_train, cv=10)
print("\nAccuracy Scores by Fold Depth 8: ", tree_8_scores)
print("Accuracy Mean:      %.4f" %tree_8_scores.mean())
print("Accuracy Std. Dev.: %.4f" %tree_8_scores.std())

tree_8_scores = cross_val_score(tree_8, X_train, y_train, cv=10, \
                                scoring='recall')
print("\nRecall Scores by Fold Depth 5: ", tree_8_scores)
print("Recall Mean:      %.4f" %tree_8_scores.mean())
print("Recall Std. Dev.: %.4f" %tree_8_scores.std())

tree_8_scores = cross_val_score(tree_8, X_train, y_train, cv=10, \
                                scoring='precision')
print("\nPrecision Scores by Fold Depth 5: ", tree_8_scores)
print("Precision Mean:      %.4f" %tree_8_scores.mean())
print("Precision Std. Dev.: %.4f" %tree_8_scores.std())

tree_8_scores = cross_val_score(tree_8, X_train, y_train, cv=10, \
                                scoring='f1')
print("\nF1 Scores by Fold Depth 5: ", tree_8_scores)
print("F1 Mean:      %.4f" %tree_8_scores.mean())
print("F1 Std. Dev.: %.4f" %tree_8_scores.std())


#Depth 10
tree_10  = tree.DecisionTreeClassifier(max_depth=10)
tree_10 = tree_10.fit(X_train,y_train)
#Cross Validation Results
tree_10_scores = cross_val_score(tree_10, X_train, y_train, cv=10)
print("\nAccuracy Scores by Fold Depth 10: ", tree_10_scores)
print("Accuracy Mean:      %.4f" %tree_10_scores.mean())
print("Accuracy Std. Dev.: %.4f" %tree_10_scores.std())

tree_10_scores = cross_val_score(tree_10, X_train, y_train, cv=10, \
                                scoring='recall')
print("\nRecall Scores by Fold Depth 5: ", tree_10_scores)
print("Recall Mean:      %.4f" %tree_10_scores.mean())
print("Recall Std. Dev.: %.4f" %tree_10_scores.std())

tree_10_scores = cross_val_score(tree_10, X_train, y_train, cv=10, \
                                scoring='precision')
print("\nPrecision Scores by Fold Depth 5: ", tree_10_scores)
print("Precision Mean:      %.4f" %tree_10_scores.mean())
print("Precision Std. Dev.: %.4f" %tree_10_scores.std())

tree_10_scores = cross_val_score(tree_10, X_train, y_train, cv=10, \
                                scoring='f1')
print("\nF1 Scores by Fold Depth 5: ", tree_10_scores)
print("F1 Mean:      %.4f" %tree_10_scores.mean())
print("F1 Std. Dev.: %.4f" %tree_10_scores.std())


#Depth 12
tree_12  = tree.DecisionTreeClassifier(max_depth=12)
tree_12 = tree_12.fit(X_train,y_train)
#Cross Validation Results
tree_12_scores = cross_val_score(tree_12, X_train, y_train, cv=10)
print("\nAccuracy Scores by Fold Depth 12: ", tree_12_scores)
print("Accuracy Mean:      %.4f" %tree_12_scores.mean())
print("Accuracy Std. Dev.: %.4f" %tree_12_scores.std())

tree_12_scores = cross_val_score(tree_12, X_train, y_train, cv=10, \
                                scoring='recall')
print("\nRecall Scores by Fold Depth 5: ", tree_12_scores)
print("Recall Mean:      %.4f" %tree_12_scores.mean())
print("Recall Std. Dev.: %.4f" %tree_12_scores.std())

tree_12_scores = cross_val_score(tree_12, X_train, y_train, cv=10, \
                                scoring='precision')
print("\nPrecision Scores by Fold Depth 5: ", tree_12_scores)
print("Precision Mean:      %.4f" %tree_12_scores.mean())
print("Precision Std. Dev.: %.4f" %tree_12_scores.std())

tree_12_scores = cross_val_score(tree_12, X_train, y_train, cv=10, \
                                scoring='f1')
print("\nF1 Scores by Fold Depth 5: ", tree_12_scores)
print("F1 Mean:      %.4f" %tree_12_scores.mean())
print("F1 Std. Dev.: %.4f" %tree_12_scores.std())


#Tree_6 has the best results, highest accuracy and lowest variance
logreg.display_binary_split_metrics(tree_6, X_train, y_train, \
                                    X_validate, y_validate)

#Print a picture of the tree
tree = tree.export_graphviz(tree_6, out_file=None, filled=True, label='none', rotate=True) 
graph = gv.Source(tree)  
graph 