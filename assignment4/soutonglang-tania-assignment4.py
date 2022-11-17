# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:29:31 2022

@Name: soutonglang-tania-assignment4.py
@author: tsout
"""
import pandas
import math
import sys
import numpy

from sklearn import preprocessing, naive_bayes
from sklearn import metrics, neural_network
from sklearn.metrics import mean_squared_error

import itertools
import time
import matplotlib.pyplot as plt

sys.path.append('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484')
import Utility

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

## Question 1 ##
purchase_df = pandas.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment4\\Purchase_Likelihood.csv")

## 1.a-d
# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   return

subData = purchase_df[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()

catGroupSize = subData['group_size'].unique()
catHomeowner = subData['homeowner'].unique()
catMarriedCouple = subData['married_couple'].unique()
catInsurance = subData['insurance'].unique()

RowWithColumn(rowVar = subData['insurance'], columnVar = subData['group_size'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['homeowner'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['married_couple'], show = 'ROW')

subData = subData.astype('category')
xTrain = pandas.get_dummies(subData[['group_size', 'homeowner', 'married_couple']])
yTrain = numpy.where(subData['insurance'] == 0, 1, 2)

feature = ['group_size', 'homeowner', 'married_couple']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['insurance'])
yLabel = labelEnc.inverse_transform([0, 1])

uGroupSize = numpy.unique(subData['group_size'])
uHomeowner = numpy.unique(subData['homeowner'])
uMarriedCouple = numpy.unique(subData['married_couple'])

featureCategory = [uGroupSize, uHomeowner, uMarriedCouple]

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['group_size', 'homeowner', 'married_couple']])

_objNB = naive_bayes.CategoricalNB(alpha = 1e-10)
thisModel = _objNB.fit(xTrain, yTrain)

## 1.f ##
xTest = featureEnc.transform([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                              [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1],
                              [3, 0, 0], [3, 0, 1], [3, 1, 0], [3, 1, 1],
                              [4, 0, 0], [4, 0, 1], [4, 1, 0], [4, 1, 1]])

y_predProb = thisModel.predict_proba(xTest)
print('Predicted Probability: \n', y_predProb)

## Question 2 ##
importData = pandas.read_excel("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment4\\Homeowner_Claim_History.xlsx")
subData = importData[['f_primary_age_tier','f_primary_gender','f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']].dropna()

# Create the frequency column
claims = importData['num_claims']
exposure = importData['exposure']
frequency = []

for year in range(len(claims)):
    frequency.append(claims[year] / exposure[year])

subData['frequency'] = frequency

catName = ['f_primary_age_tier','f_primary_gender','f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
yName = 'frequency'

trainData = subData[catName + [yName]].dropna().reset_index(drop = True)
n_sample = trainData.shape[0]

# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category')
u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index))

# Reorder the categories of the categorical variables in ascending frequency
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

X = pandas.get_dummies(trainData[catName].astype('category'))
# X = X.join(trainData[yName])
X.insert(0, '_BIAS_', 1.0)

# Identify the aliased parameters
n_param = X.shape[1]
XtX = X.transpose().dot(X)
origDiag = numpy.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = 1.0e-7)
X_reduce = X.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_'])

y = trainData[yName].astype('category')
y_category = y.cat.categories
n_category = len(y_category)

# Grid Search for the best neural network architecture
actFunc = ['identity','tanh']
nLayer = range(1,11,1) # 1-10 layers
nHiddenNeuron = range(1,6,1) # 1-5 neurons
combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

result_list = []

## 2.a
for comb in combList:
   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   nHiddenNeuron = comb[2]

   nnObj = neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
              activation = actFunc, verbose = False, max_iter = 10000, random_state = 31010)
   thisFit = nnObj.fit(X_reduce, y.astype(int))
   n_iter = nnObj.n_iter_
   y_pred = nnObj.predict(X_reduce)

   # Calculate Root Average Squared Error
   rmse = math.sqrt(mean_squared_error(y, y_pred))
   relative_error = numpy.linalg.norm(y_pred.astype(numpy.float64) - y.astype(numpy.float64) / numpy.linalg.norm(y.astype(numpy.float64)))
   pear_corr = numpy.corrcoef(y, y_pred)[0][1]
   elapsed_time = time.time() - time_begin
   result_list.append([actFunc, nLayer, nHiddenNeuron, n_iter, nnObj.best_loss_, rmse, relative_error, pear_corr, elapsed_time])

result_df = pandas.DataFrame(result_list, columns = ['ActivationFunction', 'nLayer', 'nHiddenNeuron', 'nIteration', 'BestLoss', 'RMSE', 'RelativeError', 'PearsonCorrelation', 'Elapsed Time'])
print(result_df)

## 2.b
print(result_df.sort_values('RMSE'))

## 2.c
# Locate the optimal architecture
optima_index = result_df['RMSE'].idxmin()
optima_row = result_df.iloc[optima_index]
actFunc = optima_row['ActivationFunction']
nLayer = optima_row['nLayer']
nHiddenNeuron = optima_row['nHiddenNeuron']

nnObj = neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
           activation = actFunc, verbose = False, max_iter = 10000, random_state = 31010)
thisFit = nnObj.fit(X_reduce, y.astype(int))
n_iter = nnObj.n_iter_
y_pred = nnObj.predict(X_reduce)

## 2.d
# Review the distributions of the predicted probabilities
cmap = ['red', 'green', 'blue']
fig, axs = plt.subplots(dpi = 200, figsize = (10,4))
for i in range(n_category):
   obs_value = y_category[i]
   plotData = y_pred[y == obs_value]
   for j in range(n_category):
      pred_value = y_category[j]
      ax = axs[i,j]
      ax.hist(plotData[pred_value], bins = 10, density = True, facecolor = cmap[i], alpha = 0.75)
      ax.yaxis.grid(True, which = 'major')
      ax.xaxis.grid(True, which = 'major')
      if (i == 0):
         ax.set_title('Pred.Prob.:' + pred_value)
      if (j == 0):
         ax.set_ylabel('Obs.:' + obs_value + '%')
plt.show()