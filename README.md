# Machine-learning-project-
Heart Disease Prediction Using Logistic Regression 
dataset


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn


disease_df = pd.read_csv("../input / framingham.csv")
disease_df.drop(['education'], inplace = True, axis = 1)
disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)
 
# removing NaN / NULL values
disease_df.dropna(axis = 0, inplace = True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# dataset
disease_df = pd.read_csv("../input / framingham.csv")
disease_df.drop(['education'], inplace = True, axis = 1)
disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)
 
# removing NaN / NULL values
disease_df.dropna(axis = 0, inplace = True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())
Output:
Sex_male  age  currentSmoker  ...  heartRate  glucose  TenYearCHD
0         1   39              0  ...       80.0     77.0           0
1         0   46              0  ...       95.0     76.0           0
2         1   48              1  ...       75.0     70.0           0
3         0   61              1  ...       65.0    103.0           1
4         0   46              1  ...       85.0     85.0           0

[5 rows x 15 columns] (3751, 15)
0    3179
1     572
Name: TenYearCHD, dtype: int64
