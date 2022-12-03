# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:53:16 2022

@author: tsout
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = {"X":[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
         "Y":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1]}
train = pd.DataFrame(train,columns = ["X","Y"])
train.head()

X = np.array(train["X"])
Y = np.array(train["Y"])

X = np.reshape(X,(20,1))
Y = np.reshape(Y,(20,1))

lr = LogisticRegression(tol = 10**-8)
lr.fit(X,Y)

test_A = [0,1,2,3,4]
test_B = [1,0,1,0,1]

test_A = np.reshape(test_A,(5,1))
predict = lr.predict(test_A)

MSE = np.square(np.subtract(test_B,predict)).mean()
RMSE = np.sqrt(MSE)
print("Root Mean Squared Error: ",RMSE)

# =============================================================================
# import matplotlib.pyplot as plt
# import numpy
# import pandas
# import sys
# 
# # Set some options for printing all the columns
# numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
# numpy.set_printoptions(linewidth = numpy.inf)
# 
# pandas.set_option('display.max_columns', None)
# pandas.set_option('display.expand_frame_repr', False)
# pandas.set_option('max_colwidth', None)
# 
# pandas.options.display.float_format = '{:,.10}'.format
# 
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
# from scipy.stats import chi2
# 
# sys.path.append('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484')
# import Utility
# 
# # catName = ['DriveTrain']
# intName = ["X"]
# yName = ["Y"]
# 
# train = {"X":[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
#          "Y":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1]}
# # nPredictor = len(catName) + len(intName)
# nPredictor = len(intName)
# 
# # cars = pandas.read_csv('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\week6\\cars.csv')
# 
# # trainData = cars[[yName] + catName + intName].dropna()
# trainData = train[[yName] + [intName]]
# # del cars
# 
# # n_sample = trainData.shape[0]
# 
# # Frequency of the nominal target
# # print('=== Frequency ===')
# # print(trainData[yName].value_counts())
# 
# # =============================================================================
# # # Specify the color sequence
# # cmap = ['indianred','sandybrown','royalblue']
# # 
# # # Explore categorical predictor
# # print('=== Frequency of  Categorical Predictors ===')
# # print(trainData[catName].value_counts())
# # 
# # for pred in catName:
# # 
# #     # Generate the contingency table of the categorical input feature by the target
# #     cntTable = pandas.crosstab(index = trainData[pred], columns = trainData[yName], margins = False, dropna = True)
# # 
# #     # Calculate the row percents
# #     pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')
# # 
# #     # Generate a horizontal stacked percentage bar chart
# #     barThick = 0.8
# #     yCat = cntTable.columns
# #     accPct = numpy.zeros(pctTable.shape[0])
# #     fig, ax = plt.subplots(dpi = 200)
# #     for j in range(len(yCat)):
# #         catLabel = yCat[j]
# #         plt.barh(pctTable.index, pctTable[catLabel], color = cmap[j], left = accPct, label = catLabel, height = barThick)
# #         accPct = accPct + pctTable[catLabel]
# #     ax.xaxis.set_major_locator(MultipleLocator(base = 20))
# #     ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
# #     ax.xaxis.set_minor_locator(MultipleLocator(base = 5))
# #     ax.set_xlabel('Percent')
# #     ax.set_ylabel(pred)
# #     plt.grid(axis = 'x')
# #     plt.legend(loc = 'lower center', bbox_to_anchor = (0.35, 1), ncol = 3)
# #     plt.show()
# # 
# # # Explore the continuous predictor
# # print('=== Descriptive Statistics of Continuous Predictors ===')
# # print(trainData[intName].describe())
# # 
# # for pred in intName:
# # 
# #     # Generate the contingency table of the interval input feature by the target
# #     cntTable = pandas.crosstab(index = trainData[pred], columns = trainData[yName], margins = False, dropna = True)
# # 
# #     # Calculate the row percents
# #     pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')
# #     yCat = cntTable.columns
# # 
# #     fig, ax = plt.subplots(dpi = 200)
# #     plt.stackplot(pctTable.index, numpy.transpose(pctTable), baseline = 'zero', colors = cmap, labels = yCat)
# #     ax.xaxis.set_major_locator(MultipleLocator(base = 1000))
# #     ax.xaxis.set_minor_locator(MultipleLocator(base = 200))
# #     ax.yaxis.set_major_locator(MultipleLocator(base = 20))
# #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
# #     ax.yaxis.set_minor_locator(MultipleLocator(base = 5))
# #     ax.set_xlabel(pred)
# #     ax.set_ylabel('Percent')
# #     plt.grid(axis = 'both')
# #     plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1), ncol = 3)
# #     plt.show()
# # =============================================================================
# 
# # Reorder the categories of the target variables in descending frequency
# u = trainData[yName].astype('category').copy()
# u_freq = u.value_counts(ascending = False)
# trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()
# 
# # Reorder the categories of the categorical variables in ascending frequency
# # =============================================================================
# # for pred in catName:
# #     u = trainData[pred].astype('category').copy()
# #     u_freq = u.value_counts(ascending = True)
# #     trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()
# # =============================================================================
# 
# # Generate a column of Intercept
# X0_train = trainData[[yName]].copy()
# X0_train.insert(0, 'Intercept', 1.0)
# X0_train.drop(columns = [yName], inplace = True)
# 
# y_train = trainData[yName].copy()
# 
# maxIter = 100
# tolS = 1e-8
# stepSummary = []
# 
# # Intercept only model
# resultList = Utility.binary_model_metric (X0_train, y_train, maxIter = maxIter, tolSweep = tolS)
# 
# llk0 = resultList[1]
# df0 = resultList[2]
# stepSummary.append(['Intercept', ' ', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN])
# 
# print('======= Step Detail =======')
# print('Step = ', 0)
# print('Step Statistics:')
# print(stepSummary)
# 
# # cName = catName.copy()
# iName = intName.copy()
# entryThreshold = 0.05
# 
# # The Deviance significance is the sixth element in each row of the test result
# def takeDevSig(s):
#     return s[6]
# 
# for step in range(nPredictor):
#     enterName = ''
#     stepDetail = []
# 
# # =============================================================================
# #     # Enter the next predictor
# #     for X_name in cName:
# #         X_train = pandas.get_dummies(trainData[[X_name]])
# #         X_train = X0_train.join(X_train)
# #         resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
# #         llk1 = resultList[1]
# #         df1 = resultList[2]
# #         devChiSq = 2.0 * (llk1 - llk0)
# #         devDF = df1 - df0
# #         devSig = chi2.sf(devChiSq, devDF)
# #         stepDetail.append([X_name, 'categorical', df1, llk1, devChiSq, devDF, devSig])
# # =============================================================================
# 
#     for X_name in iName:
#         X_train = trainData[[X_name]]
#         X_train = X0_train.join(X_train)
#         resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
#         llk1 = resultList[1]
#         df1 = resultList[2]
#         devChiSq = 2.0 * (llk1 - llk0)
#         devDF = df1 - df0
#         devSig = chi2.sf(devChiSq, devDF)
#         stepDetail.append([X_name, 'interval', df1, llk1, devChiSq, devDF, devSig])
# 
#     # Find a predictor to enter, if any
#     # Find a predictor to add, if any
#     stepDetail.sort(key = takeDevSig, reverse = False)
#     enterRow = stepDetail[0]
#     minPValue = takeDevSig(enterRow)
#     if (minPValue <= entryThreshold):
#         stepSummary.append(enterRow)
#         df0 = enterRow[2]
#         llk0 = enterRow[3]
# 
#         enterName = enterRow[0]
#         enterType = enterRow[1]
#         if (enterType == 'categorical'):
#             X_train = pandas.get_dummies(trainData[[enterName]].astype('category'))
#             X0_train = X0_train.join(X_train)
#             # cName.remove(enterName)
#         elif (enterType == 'interval'):
#             X_train = trainData[[enterName]]
#             X0_train = X0_train.join(X_train)
#             iName.remove(enterName)
#     else:
#         break
# 
#     # Print debugging output
#     print('======= Step Detail =======')
#     print('Step = ', step+1)
#     print('Step Statistics:')
#     print(stepDetail)
#     print('Enter predictor = ', enterName)
#     print('Minimum P-Value =', minPValue)
#     print('\n')
# 
# # End of forward selection
# print('======= Step Summary =======')
# stepSummary = pandas.DataFrame(stepSummary, columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig'])
# print(stepSummary)
# =============================================================================