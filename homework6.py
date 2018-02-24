# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to Week 6 Assignment for Stat656 Spring 2018
Uses Python to fit neural network models and evaluate using cross validation.
"""

# classes for neural network
from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPClassifier

#other needed classes
from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np

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
    'purpose' :[2,('0','1','2','3','4','5','6','8','9','X'),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings' :[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]],
}

rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                          interval_scale = 'std', drop=False, display=True)
encoded_df = rie.fit_transform(df)
varlist = ['good_bad']
X = encoded_df.drop(varlist, axis=1)
y = encoded_df[varlist]
np_y = np.ravel(y) #convert dataframe column to flat array


network_list = [(3), (11), (5,4), (6,5), (7,6)]
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X, np_y)
    NeuralNetwork.display_binary_metrics(fnn, X, np_y)

# Neural Network Cross-Validation
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    mean_score = []
    std_score  = []
    for s in score_list:
        fnn_10 = cross_val_score(fnn, X, np_y, cv=10, scoring=s)
        mean_score.append(fnn_10.mean())
        std_score.append(fnn_10.std())

    print("{:.<13s}{:>6s}{:>13s}".format("\nMetric", "Mean", "Std. Dev."))
    for i in range(len(score_list)):
        score_name = score_list[i]
        mean       = mean_score[i]
        std        = std_score[i]
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(score_name, mean, std))
        

X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
np_y_validate = np.ravel(y_validate)
np_y_train = np.ravel(y_train)
# Evaluate the network with the best structure
dtc = MLPClassifier(hidden_layer_sizes=(5,4), activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
dtc = dtc.fit(X_train,np_y_train)

print("\nTraining Data\nRandom Selection of 70% of Original Data")
NeuralNetwork.display_binary_split_metrics(dtc, X_train, np_y_train, \
                                     X_validate, np_y_validate)