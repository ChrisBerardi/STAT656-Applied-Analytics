# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to STAT656 Midterm, Spring 2017
"""

# classes for logistic regression
from Class_regression import logreg
from sklearn.linear_model import LogisticRegression

#classes for tree model
from Class_tree import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# classes for neural network
from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPClassifier

#other needed classes
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np


file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Midterm/'
df = pd.read_excel(file_path+"CreditCard_Defaults.xlsx")

# Attribute Map:  the key is the name in the DataFrame
# The first number of 0=Interval, 1=Binary, 2=Nominal
# The 1st tuple for interval attributes is their lower and upper bounds
# The 1st tuple for categorical attributes is their allowed categories
# The 2nd tuple contains the number missing and number of outliers
attribute_map = {
    'Default'       :[1,(0,1),[0,0]], #1=Default
    'Gender'        :[1,(1,2),[0,0]], #1=Female
    'Education'     :[2,(0,1,2,3,4,5,6),[0,0]],
    'Marital_Status':[2,(0,1,2,3),[0,0]],
    'card_class'    :[2,(1,2,3),[0,0]],
    'Age'           :[0,(20,80),[0,0]],
    'Credit_Limit'  :[0,(100,80000),[0,0]],
    'Jun_Status'    :[0,(-2,8),[0,0]],
    'May_Status'    :[0,(-2,8),[0,0]],
    'Apr_Status'    :[0,(-2,8),[0,0]],
    'Mar_Status'    :[0,(-2,8),[0,0]],
    'Feb_Status'    :[0,(-2,8),[0,0]],
    'Jan_Status'    :[0,(-2,8),[0,0]],
    'Jun_Bill'      :[0,(-12000,32000),[0,0]],
    'May_Bill'      :[0,(-12000,32000),[0,0]],
    'Apr_Bill'      :[0,(-12000,32000),[0,0]],
    'Mar_Bill'      :[0,(-12000,32000),[0,0]],
    'Feb_Bill'      :[0,(-12000,32000),[0,0]],
    'Jan_Bill'      :[0,(-12000,32000),[0,0]],
    'Jun_Payment'   :[0,(0,60000),[0,0]],
    'May_Payment'   :[0,(0,60000),[0,0]],
    'Apr_Payment'   :[0,(0,60000),[0,0]],
    'Mar_Payment'   :[0,(0,60000),[0,0]],
    'Feb_Payment'   :[0,(0,60000),[0,0]],
    'Jan_Payment'   :[0,(0,60000),[0,0]],
    'Jun_PayPercent':[0,(0,1),[0,0]],
    'May_PayPercent':[0,(0,1),[0,0]],
    'Apr_PayPercent':[0,(0,1),[0,0]],
    'Mar_PayPercent':[0,(0,1),[0,0]],
    'Feb_PayPercent':[0,(0,1),[0,0]],
    'Jan_PayPercent':[0,(0,1),[0,0]]
}

#Use 3 different encodings: one for trees(no scaling), neural networks(scaling)
#, logistic(drop=True)
varlist = ['Default']

#Neural encoding
rie_n = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                          interval_scale = 'std', drop=False, display=True)
encoded_df_n = rie_n.fit_transform(df)
X_n = encoded_df_n.drop(varlist, axis=1)
y_n = encoded_df_n[varlist]
np_y_n = np.ravel(y_n) #convert dataframe column to flat array

#Logistic encoding
rie_l = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                          interval_scale = 'std', drop=True, display=True)
encoded_df_l = rie_l.fit_transform(df)
X_l = encoded_df_l.drop(varlist, axis=1)
y_l = encoded_df_l[varlist]
np_y_l = np.ravel(y_l) #convert dataframe column to flat array

#Tree encoding
rie_t = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                           drop=False,interval_scale = None, display=True)
encoded_df_t = rie_t.fit_transform(df)
X_t = encoded_df_t.drop(varlist, axis=1)
y_t = encoded_df_t[varlist]
np_y_t = np.ravel(y_t) #convert dataframe column to flat array
col = list(encoded_df_t.drop('Default', axis=1)) #Needed for variable importance list

#Tree Methods
#Start with Random Forest
# Cross-Validation
estimators_list   = [10,20,50]
max_features_list = ['auto', .6, .7, .8]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
for e in estimators_list:
    for f in max_features_list:
        print("\nNumber of Trees: ", e, " Max_features: ", f)
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                    max_depth=100, min_samples_split=2, \
                    min_samples_leaf=1, max_features=f, \
                    n_jobs=1, bootstrap=True, random_state=12345)
        rfc= rfc.fit(X_t, np_y_t)
        scores = cross_validate(rfc, X_t, np_y_t, scoring=score_list, \
                                return_train_score=False, cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_estimator    = e
            best_max_features = f

print("\nBest based on F1-Score")
print("Best Number of Estimators (trees) = ", best_estimator)
print("Best Maximum Features = ", best_max_features)

# Evaluate the random forest with the best configuration
X_train_t, X_validate_t, y_train_t, y_validate_t = \
            train_test_split(X_t, np_y_t,test_size = 0.3, random_state=12345)

rfc = RandomForestClassifier(n_estimators=best_estimator, criterion="gini", \
                    max_depth=100, min_samples_split=2, \
                    min_samples_leaf=1, max_features=best_max_features, \
                    n_jobs=1, bootstrap=True, random_state=12345)
rfc= rfc.fit(X_train_t, y_train_t)
DecisionTree.display_importance(rfc,col)
#Copy the code from Dr. Jone's class here to be able to extract from the sorted
#list of predictor importance

nx = rfc.n_features_
max_label = 6
for i in range(len(col)):
    if len(col[i]) > max_label:
        max_label = len(col[i])+4
    label_format = ("{:.<%i" %max_label)+"s}{:9.4f}"
   
    features = []
    this_col = []
    for i in range(nx):
        features.append(rfc.feature_importances_[i])
        this_col.append(col[i])
    sorted = False
    while (sorted==False):
        sorted = True
        for i in range(nx-1):
            if features[i]<features[i+1]:
                sorted=False
                x = features[i]
                c = this_col[i]
                features[i] = features[i+1]
                this_col[i] = this_col[i+1]
                features[i+1] = x
                this_col[i+1] = c

#Use the importance list to generate a list of the 20 most important
#attributes to be used for logistic reggression
model_20 = this_col[0:20]
#Decision Tree Models
# Cross Validation
depth_list = [5, 6, 7, 8, 10, 12]
max_f1 = 0
for d in depth_list:
    print("\nMaximum Tree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, \
                                 min_samples_split=5)
    dtc = dtc.fit(X_t,y_t)
    scores = cross_validate(dtc, X_t, y_t, scoring=score_list, \
                            return_train_score=False, cv=10)
    
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_depth    = d
            
print("\nBest based on F1-Score")
print("Best Depth = ", best_depth)
# Evaluate the tree with the best depth
dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5, min_samples_split=5)
dtc = dtc.fit(X_train_t,y_train_t)

#Logistic Models
max_f1 = 0
#Try logistic model with 5,10 and 20 of the most important attributes as 
#determined by the random tree as well as the full model with all predictors
predictor_list=[5,10,20,30]
for p in predictor_list:
    if p != 30:
        print("\nNumber of Predictors: ", p)
        lgr = LogisticRegression()
        lgr.fit(X_l.loc[:,model_20[0:p]], np_y_l)
        scores = cross_validate(lgr, X_l.loc[:,model_20[0:p]], np_y_l,\
                                scoring=score_list, return_train_score=False, \
                                cv=10)
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if mean > max_f1:
                max_f1 = mean
                best_predictor  = p
    else: 
        print("\nFull Model: ", p)
        lgr = LogisticRegression()
        lgr.fit(X_l, np_y_l)
        scores = cross_validate(lgr, X_l,np_y_l,\
                                scoring=score_list, return_train_score=False, \
                                cv=10)
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if mean > max_f1:
                max_f1 = mean
                best_predictor  = p
#Evaluate the best model
print("\nBest based on F1-Score")
print("Best Number of Predictors = ", best_predictor)
X_train_l, X_validate_l, y_train_l, y_validate_l = \
            train_test_split(X_l,y_l,test_size = 0.3, random_state=12345)
np_y_validate_l = np.ravel(y_validate_l)
np_y_train_l = np.ravel(y_train_l)
# Evaluate the network with the best number of predictors
lgc = LogisticRegression()
lgc = lgc.fit(X_train_l.loc[:,model_20[0:best_predictor]],np_y_train_l)

#Neural Network Models
network_list = [(3),(4),(5),(6),(7),(2,1),(2,2),(3,2),(4,3)]
max_f1 = 0
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X_n, np_y_n)
# Neural Network Cross-Validation
    mean_score = []
    std_score  = []
    for s in score_list:
        fnn_10 = cross_val_score(fnn, X_n, np_y_n, cv=10, scoring=s)
        mean_score.append(fnn_10.mean())
        std_score.append(fnn_10.std())

    print("{:.<13s}{:>6s}{:>13s}".format("\nMetric", "Mean", "Std. Dev."))
    for i in range(len(score_list)):
        score_name = score_list[i]
        mean       = mean_score[i]
        std        = std_score[i]
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(score_name, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_list = nn
        
print("\nBest based on F1-Score")
print("Best Network Configuration = ", best_list)

X_train_n, X_validate_n, y_train_n, y_validate_n = \
            train_test_split(X_n,y_n,test_size = 0.3, random_state=7)
np_y_validate_n = np.ravel(y_validate_n)
np_y_train_n = np.ravel(y_train_n)
# Evaluate the network with the best structure
nnc = MLPClassifier(hidden_layer_sizes=best_list, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
nnc = nnc.fit(X_train_n,np_y_train_n)

#Compare Best Models
print("\nTraining Data\nRandom Selection of 70% of Original Data")
print("\nRandom Forest")
print("\nEstimators (trees) = ", best_estimator)
print("\nMaximum Features = ", best_max_features)
DecisionTree.display_binary_split_metrics(rfc, X_train_t, y_train_t, \
                                              X_validate_t, y_validate_t)
print("\nDecision Tree")
print("\nDepth",best_depth)
DecisionTree.display_binary_split_metrics(dtc, X_train_t, y_train_t, \
                                     X_validate_t, y_validate_t)
print("\nNeural Network")
print("\nNetwork Configuration = ", best_list)
NeuralNetwork.display_binary_split_metrics(nnc, X_train_n, np_y_train_n, \
                                     X_validate_n, np_y_validate_n)
print("\nLogistic Regression")
print("\nNumber of Predictors = ", best_predictor)
logreg.display_binary_split_metrics(lgc, \
                                    X_train_l.loc[:,model_20[0:best_predictor]]\
                                    , np_y_train_l, \
                                     X_validate_l.loc[:,model_20[0:best_predictor]]\
                                     , np_y_validate_l)