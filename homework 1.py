# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solutions to Homework 1 for Stat656 Spring 2018
Prints Summary Statistics for the sonar data set
"""
import numpy  as np
import pandas as pd

sonar = pd.read_csv('C:/Users/Saistout/Desktop/656 Applied Analytics/Data/sonar_hw1.csv')
#Extract the sonar data without nominal data
sonar_x = sonar.iloc[:,0:60]
#Determine which indicies are nans
nan = np.isnan(sonar_x)
#Calculates the number of nans for each sonar frequency
nans = pd.DataFrame(np.sum(nan),columns=['Missing'])
#Select only the values between 0 and 1
accept = sonar_x[(sonar_x>=0) & (sonar_x<=1)]
#Median
med = pd.DataFrame(accept.median(),columns=['Median'])
#Max
maxs = pd.DataFrame(accept.max(), columns=['Max'])
#Min
mins = pd.DataFrame(accept.min(), columns=['Min'])
#Positive Outlier
hi = pd.DataFrame(np.sum(sonar_x >1), columns=['n hi'])
#Negative Outlier
low = pd.DataFrame(np.sum(sonar_x <0), columns=['n low'])
#Create desired summary table
summary = pd.concat((mins,med,maxs,nans,low,hi),axis=1)

print(summary)