"""
Author: Chris Berardi
Solution to STAT656 Week 8 Assigment, Spring 2017
Using random undersampling to solve a rare event problem.
"""

from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy  as np
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

# Function for calculating loss and confusion matrix
def loss_cal(y, y_predict, fp_cost, fn_cost, display=True):
    loss     = [0, 0]       #False Neg Cost, False Pos Cost
    conf_mat = [0, 0, 0, 0] #tn, fp, fn, tp
    for j in range(len(y)):
        if y[j]==0:
            if y_predict[j]==0:
                conf_mat[0] += 1 #True Negative
            else:
                conf_mat[1] += 1 #False Positive
                loss[1] += fp_cost[j]
        else:
            if y_predict[j]==1:
                conf_mat[3] += 1 #True Positive
            else:
                conf_mat[2] += 1 #False Negative
                loss[0] += fn_cost[j]
    if display:
        fn_loss = loss[0]
        fp_loss = loss[1]
        total_loss = fn_loss + fp_loss
        misc    = conf_mat[1] + conf_mat[2]
        misc    = misc/len(y)
        print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
        print("{:.<23s}{:10.0f}".format("False Negative Cost", fn_loss))
        print("{:.<23s}{:10.0f}".format("False Positive Cost", fp_loss))
        print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
    return loss, conf_mat


# Attribute Map for CreditData_RareEvent.xlsx, N=10,500
#0: Interval, 1: Binary, 2: Nominal
attribute_map = {
    'age':[0,(1, 120),[0,0]],
    'amount':[0,(0, 20000),[0,0]],
    'duration':[0,(1,100),[0,0]],
    'checking':[2,(1, 2, 3, 4),[0,0]],
    'coapp':[2,(1,2,3),[0,0]],
    'depends':[1,(1,2),[0,0]],
    'employed':[2,(1,2,3,4,5),[0,0]],
    'existcr':[2,(1,2,3,4),[0,0]],
    'foreign':[1,(1,2),[0,0]],
    'good_bad':[1,('bad', 'good'),[0,0]],
    'history':[2,(0,1,2,3,4),[0,0]],
    'housing':[2,(1, 2, 3), [0,0]],
    'installp':[2,(1,2,3,4),[0,0]],
    'job':[2,(1,2,3,4),[0,0]],
    'marital':[2,(1,2,3,4),[0,0]],
    'other':[2,(1,2,3),[0,0]],
    'property':[2,(1,2,3,4),[0,0]],
    'resident':[2,(1,2,3,4),[0,0]],
    'savings':[2,(1,2,3,4,5),[0,0]],
    'telephon':[1,(1,2),[0,0]] 
}


file_path = \
'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Week 8 Assignment/'
df = pd.read_excel(file_path+"CreditData_RareEvent.xlsx")
# Encode for Logistic Regression, drop last one-hot column
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                          interval_scale = 'std', drop=True, display=False)
encoded_df = rie.fit_transform(df)
# Create X and y, numpy arrays
# bad=0 and good=1
y = np.asarray(encoded_df['good_bad']) # The target is not scaled or imputed
X = np.asarray(encoded_df.drop('good_bad',axis=1))

# Setup false positive and false negative costs for each transaction
fp_cost = np.array(df['amount'])
fn_cost = np.array(.15*df['amount'])


#See what happens if fit is done without any RUS
tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
tree.fit(X, y)
print("\nDecision Tree Model using Entire Dataset")
col = rie.col
col.remove('good_bad')
DecisionTree.display_binary_metrics(tree, X, y)
print("\nLoss Calculations from a Decision Tree Model")
print("Model fitted to the entire dataset:")
loss, conf_mat = loss_cal(y, tree.predict(X), fp_cost, fn_cost)


# Setup random number seeds
rand_val = np.array([1, 15, 168, 1834, 12545, 54321, 5431, 431, 32, 2])
# Ratios of Majority:Minority Events
ratio = [ '50:50', '60:40', '70:30', '75:25', '80:20', '85:15', '90:10' ]
# Dictionaries contains number of minority and majority events in each ratio sample
# n_majority = ratio x n_minority
rus_ratio = ({0:500, 1:500}, {0:500, 1:750}, {0:500, 1:1167},\
             {0:500, 1:1500}, {0:500, 1:2000}, {0:500, 1:2833}, {0:500, 1:4500})


# Best model is one that minimizes the loss
min_loss   = 9e+15
best_ratio = 0
for k in range(len(rus_ratio)):
    rand_vals = (k+1)*rand_val
    print("\nDecision Tree Model using " + ratio[k] + " RUS")
    fn_loss = np.zeros(len(rand_vals))
    fp_loss = np.zeros(len(rand_vals))
    misc    = np.zeros(len(rand_vals))
    for i in range(len(rand_vals)):
        rus = RandomUnderSampler(ratio=rus_ratio[k], \
                random_state=rand_vals[i], return_indices=False, \
                replacement=False)
        X_rus, y_rus = rus.fit_sample(X, y)
        tr = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
        tr.fit(X_rus, y_rus)
        loss, conf_mat = loss_cal(y, tr.predict(X), fp_cost, fn_cost,\
                                  display=False)
        fn_loss[i] = loss[0]
        fp_loss[i] = loss[1]
        misc[i]    = conf_mat[1] + conf_mat[2]
    misc = np.sum(misc)/(10500 * len(rand_vals))
    fn_avg_loss = np.average(fn_loss)
    fp_avg_loss = np.average(fp_loss)
    total_loss  = fn_loss + fp_loss
    avg_loss    = np.average(total_loss)
    std_loss    = np.std(total_loss)
    print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
    print("{:.<23s}{:10.0f}".format("False Negative Cost", fn_avg_loss))
    print("{:.<23s}{:10.0f}".format("False Positive Cost", fp_avg_loss))
    print("{:.<23s}{:10.0f}{:5s}{:<10.2f}".format("Total Loss", avg_loss, \
                  " +/- ", std_loss))
    if avg_loss < min_loss:
        min_loss   = avg_loss
        best_ratio = k

# Ensemble Modeling - Averaging Classification Probabilities
avg_prob = np.zeros((len(y),2))
# Setup 10 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**31 - 1
rand_value = np.random.randint(1, high=max_seed, size=10)
# Model 100 random samples, each with a 70:30 ratio
for i in range(len(rand_value)):
    rus = RandomUnderSampler(ratio=rus_ratio[best_ratio], \
                    random_state=rand_value[i], return_indices=False, \
                    replacement=False)
    X_rus, y_rus = rus.fit_sample(X, y)
    ltr = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, \
                                 min_samples_split=5,criterion='gini')
    tr.fit(X_rus, y_rus)
    avg_prob += tr.predict_proba(X)
avg_prob = avg_prob/len(rand_value)
# Set y_pred equal to the predicted classification
y_pred = avg_prob[0:,0] < 0.5
y_pred.astype(np.int)
# Calculate loss from using the ensemble predictions
print("\nEnsemble Estimates based on averaging",len(rand_value), "Models")
loss, conf_mat = loss_cal(y, y_pred,fp_cost,fn_cost)
