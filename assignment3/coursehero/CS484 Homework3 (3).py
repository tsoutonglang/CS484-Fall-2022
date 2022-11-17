#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from scipy.stats import chi2
sys.path.append('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484')
import Utility
import statsmodels.api as smodel



# df_ch = pd.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment3\\claim_history.csv")


# =============================================================================
# df_ch
# 
# 
# 
# 
# """
# Helper Function to classify Education level
# """
# def education_classifier(df):
#     if df == "Doctors":
#         return 1
#     elif df == "Masters" :
#         return 2
#     elif df == "Bachelors" :
#         return 3
#     elif df == "High School" :
#         return 4
#     elif df == "Below High School" :
#         return 5
# 
#     
# def calculate_entropy(df):
#     #computing positive target values
#     if df['CAR_USE_Commercial'].count() != 0:
#         target_positive = df['CAR_USE_Commercial'].sum()/df['CAR_USE_Commercial'].count()
#     else:
#         target_positive = 1
#     flag = False
#     #computing negative target values
#     if target_positive != 1:
#         target_negative = 1 - target_positive
#     else:
#         target_entropy = 0
#         flag = True
#         
#     if target_positive == 0:
#         target_entropy = 0
#         flag = True
#         
#     #calculating entropy
#     if flag == False:
#         target_entropy = (-(target_positive) * math.log(target_positive, 2)) - ((target_negative) * math.log(target_negative, 2))
#     return target_entropy
# 
# def optimal_entropy(df, series):
#     entropy_list = []
#     lis = []
#     #returns dataframe combinations
#     for i in range(1,math.ceil(len(df[series].unique())/2) + 1):
#         lis += combinations(df[series].unique(),i)
#         
#     for i in lis:
#         temp = pd.DataFrame()
#         for x in i:
#             x_query = df.query("{} == '{}'".format(series, x))
#             frames = [temp, x_query]
#             temp = pd.concat(frames)
#             
#         opp_query = df.copy()
#         for x in i:
#             opp_query = opp_query.query("{} != '{}'".format(series, x))
#         total_entropy = (temp["ID"].count() / df["ID"].count()) * (calculate_entropy(temp))                               + (opp_query["ID"].count()/df["ID"].count()) * (calculate_entropy(opp_query))
#         entropy_list.append((i, total_entropy))
#     return entropy_list
# 
# def optimal_entropy_ordinal(df,series):
#     entropy_list = []
#     lis = df[series].unique()
#     #returns dataframe combinations
#         
#     for i in lis:
#         temp = pd.DataFrame()
#         x_query = df.query("{} <= {}".format(series, i))
#         frames = [temp, x_query]
#         temp = pd.concat(frames)
#             
#         opp_query = df.query("{} > {}".format(series, i))
#         total_entropy = (temp["ID"].count() / df["ID"].count()) * (calculate_entropy(temp))                               + (opp_query["ID"].count()/df["ID"].count()) * (calculate_entropy(opp_query))
#         entropy_list.append((i, total_entropy))
#     return entropy_list
# 
# def split_query(df, series, lis):
#     temp = pd.DataFrame()
#     for x in lis:
#         x_query = df.query("{} == '{}'".format(series, x))
#         frames = [temp, x_query]
#         temp = pd.concat(frames)
#             
#     opp_query = df.copy()
#     for x in lis:
#             opp_query = opp_query.query("{} != '{}'".format(series, x))
#     
#     return (temp.reset_index(), opp_query.reset_index())
# 
# def split_query_ordinal(df, series, num):
#     temp = pd.DataFrame()
#     x_query = df.query("{} <= {}".format(series, num))
#             
#     opp_query = df.query("{} > {}".format(series, num))
#     
#     return (x_query.reset_index(), opp_query.reset_index())
# 
# def min_entropy(l):
#     mini = 0
#     minv = l[0][1]
#     for x in range(len(l)):
#         if minv > l[x][1]:
#             minv = l[x][1]
#             mini = x
#     return l[mini]
# 
# df_ch["EDUCATION_RANK"] = df_ch["EDUCATION"].apply(education_classifier)
# 
# df_ch = df_ch[["ID", "CAR_USE", "CAR_TYPE", "OCCUPATION", "EDUCATION_RANK"]]
# df_ch
# 
# df_ch.columns
# 
# #Commercial Car Use dummy = 1, private = 0
# df_ch = pd.get_dummies(df_ch, columns = ["CAR_USE"]).drop(columns = "CAR_USE_Private")
# df_ch
# 
# #Problem 1a
# original_entropy = calculate_entropy(df_ch)
# original_entropy
# 
# print(df_ch.columns)
# df_ch["CAR_TYPE"].unique()
# 
# l = []
# for i in range(1,6):
#     l += combinations(df_ch['CAR_TYPE'].unique(),i)
#     
# # for i in l:
# #     print (i)
# 
# # for i in df_ch["EDUCATION_RANK"].unique():
# #     print(i)
# 
# ct = optimal_entropy(df_ch, "CAR_TYPE")
# oc = optimal_entropy(df_ch, "OCCUPATION")
# 
# ed = optimal_entropy_ordinal(df_ch, "EDUCATION_RANK")
# ed
# 
# #Optimal Splits for Car Type
# print("\ncar type", min_entropy(ct))
# #Optimal Splits for Occupation
# print("occupation", min_entropy(oc))
# #Optimal Splits for Education Rank
# print("education rank", min_entropy(ed))
# 
# sq = split_query(df_ch, "OCCUPATION", min_entropy(oc)[0])
# df_left = sq[0]
# df_right = sq[1]
# 
# df_right["OCCUPATION"].unique()
# 
# lct = optimal_entropy(df_left, "CAR_TYPE")
# loc = optimal_entropy(df_left, "OCCUPATION")
# ler = optimal_entropy_ordinal(df_left, "EDUCATION_RANK")
# ler
# 
# #Optimal Splits for Car Type
# print("\nleft car type ", min_entropy(lct))
# #Optimal Splits for Education Rank
# print("left education rank", min_entropy(ler))
# #Optimal Splits for Occupation
# print("left occupation", min_entropy(loc))
# 
# sq2 = split_query_ordinal(df_left, "EDUCATION_RANK", min_entropy(ler)[0])
# df_left2 = sq2[0]
# df_right2 = sq2[1]
# 
# df_left2["CAR_USE_Commercial"].sum()/df_left2["CAR_USE_Commercial"].count()
# 
# rct = optimal_entropy(df_right, "CAR_TYPE")
# rer = optimal_entropy_ordinal(df_right, "EDUCATION_RANK")
# roc = optimal_entropy(df_right, "OCCUPATION")
# 
# #Optimal Splits for Car Type
# print("\nright car type", min_entropy(rct))
# #Optimal Splits for Education Rank
# print("right education rank", min_entropy(rer))
# #Optimal Splits for Occupation
# print("right occupation", min_entropy(roc))
# 
# sq3 = split_query(df_right, "CAR_TYPE", min_entropy(rct)[0])
# df_left3 = sq3[0]
# df_right3 = sq3[1]
# 
# df_right3["ID"].count()
# 
# 823+3029+4594+1856
# 
# 1856/10302
# 
# =============================================================================
"""
QUESTION 2
"""
df_sv = pd.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment3\\sample_v10.csv")
df_sv

df_sv.groupby(["y"]).count()["x1"].reset_index().rename(columns = {"x1": "count"})

# =============================================================================
# # Set some options for printing all the columns
# np.set_printoptions(precision = 10, threshold = sys.maxsize)
# np.set_printoptions(linewidth = np.inf)
# 
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', None)
# pd.set_option('precision', 10)
# 
# pd.options.display.float_format = '{:,.10}'.format
# =============================================================================

intName = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6',  'x7', 'x8', 'x9', 'x10']
yName = 'y'

nPredictor = len(intName)

trainData = df_sv[[yName] + intName].dropna()

n_sample = trainData.shape[0]

# Frequency of the nominal target
print('=== Frequency of ' + yName + ' ===')
print(trainData[yName].value_counts().reset_index())


# Specify the color sequence
cmap = ['indianred','sandybrown','royalblue']

# Explore the continuous predictor
print('=== Descriptive Statistics of Continuous Predictors ===')
print(trainData[intName].describe())

for pred in intName:

    # Generate the contingency table of the interval input feature by the target
    cntTable = pd.crosstab(index = trainData[pred], columns = trainData[yName], margins = False, dropna = True)

    # Calculate the row percents
    pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')
    yCat = cntTable.columns

    fig, ax = plt.subplots(dpi = 200)
    plt.stackplot(pctTable.index, np.transpose(pctTable), baseline = 'zero', colors = cmap, labels = yCat)
    ax.xaxis.set_major_locator(MultipleLocator(base = 1000))
    ax.xaxis.set_minor_locator(MultipleLocator(base = 200))
    ax.yaxis.set_major_locator(MultipleLocator(base = 20))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_minor_locator(MultipleLocator(base = 5))
    ax.set_xlabel(pred)
    ax.set_ylabel('Percent')
    plt.grid(axis = 'both')
    plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1), ncol = 3)
    plt.show()

# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category').copy()
u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Generate a column of Intercept
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)

y_train = trainData[yName].copy()

maxIter = 20
tolS = 1e-7
stepSummary = pd.DataFrame()

# Intercept only model
print("INTERCEPT ONLY MODEL -------")
resultList = Utility.MNLogisticModel (X0_train, y_train, maxIter = maxIter, tolSweep = tolS)

modelObj = smodel.MNLogit(y_train, X0_train)
thisFit = modelObj.fit(full_output = True)
print("Model Summary///" + "\n")
print(thisFit.summary())

print("Incept only Model log-likelihood = {}".format(thisFit.llnull))

llk0 = resultList[1]
df0 = resultList[2]
stepSummary = stepSummary.append([['Intercept', ' ', df0, llk0, np.NaN, np.NaN, np.NaN]], ignore_index = True)
stepSummary.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

print('======= Step Detail =======')
print('Step = ', 0)
print('Step Statistics:')
print(stepSummary)

iName = intName.copy()
entryThreshold = 0.05

for step in range(nPredictor):
    enterName = ''
    stepDetail = pd.DataFrame()



    for X_name in iName:
        X_train = trainData[[X_name]]
        X_train = X0_train.join(X_train)
        resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[1]
        df1 = resultList[2]
        AIC = 2.0 * df1 - 2.0 * llk1
        BIC = df1 * np.log(n_sample) - 2.0 * llk1
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = stepDetail.append([[X_name, 'interval', df1, llk1, devChiSq, devDF, devSig, AIC, BIC]], ignore_index = True)

    stepDetail.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig', 'AIC', 'BIC']

    # Find a predictor to enter, if any
    stepDetail.sort_values(by = 'BIC', axis = 0, ascending = True, inplace = True)
    enterRow = stepDetail.iloc[0].copy()
    minPValue = enterRow['DevSig']
    if (minPValue <= entryThreshold):
        stepSummary = stepSummary.append([enterRow], ignore_index = True)
        df0 = enterRow['ModelDF']
        llk0 = enterRow['ModelLLK']

        enterName = enterRow['Predictor']
        enterType = enterRow['Type']
        if (enterType == 'categorical'):
            X_train = pd.get_dummies(trainData[[enterName]].astype('category'))
            X0_train = X0_train.join(X_train)
        elif (enterType == 'interval'):
            X_train = trainData[[enterName]]
            X0_train = X0_train.join(X_train)
            iName.remove(enterName)
    else:
        break
        


    # Print debugging output
    print('======= Step Detail =======')
    print('Step = ', step+1)
    print('Step Statistics:')
    print(stepDetail)
    print('Enter predictor = ', enterName)
    print('Minimum P-Value =', minPValue)
    print('\n')

# End of forward selection
print('======= Step Summary =======')
print(stepSummary)