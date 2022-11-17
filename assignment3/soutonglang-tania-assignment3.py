# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:00:19 2022

@author: tsout
"""
import pandas
import math
from itertools import combinations
import sys
sys.path.append('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484')
import Utility
import statsmodels.api as smodel
import numpy as np
from scipy.stats import chi2

## Question 1 ##
claimHistory = pandas.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment3\\claim_history.csv", delimiter=',')

## 1.a
total = claimHistory['CAR_USE']
private = claimHistory.loc[claimHistory['CAR_USE'] == 'Private']
commercial = claimHistory.loc[claimHistory['CAR_USE'] == 'Commercial']

entropyRoot = (-private.shape[0] / total.shape[0]) * math.log2(private.shape[0] / total.shape[0]) - (commercial.shape[0] / total.shape[0])*math.log2(commercial.shape[0] / total.shape[0])
print("1.a\nroot entropy: ", entropyRoot)

## 1.b
def educationLevel(education):
    if education == "Doctors":
        return 1            # highest
    elif education == "Masters" :
        return 2
    elif education == "Bachelors" :
        return 3
    elif education == "High School" :
        return 4
    elif education == "Below High School" :
        return 5            # lowest

# Organize the education levels
claimHistory["EDUCATION_LEVEL"] = claimHistory["EDUCATION"].apply(educationLevel)

# Isolate ID, CAR_USE, CAR_TYPE, OCCUPATION, and EDUCATION_LEVEL
claimHistory = claimHistory[["ID", "CAR_USE", "CAR_TYPE", "OCCUPATION", "EDUCATION_LEVEL"]]

# Create dummy column for CAR_USE (Label Field)
claimHistory = pandas.get_dummies(claimHistory, columns = ["CAR_USE"]).drop(columns = "CAR_USE_Private")

combos = []
for i in range(1,6):
    combos += combinations(claimHistory['CAR_TYPE'].unique(),i)

# To calculate the entropy after creating the dummy columns
def entropyCalculation(df):
    # Compute positives
    if df['CAR_USE_Commercial'].count() != 0:
        targetP = df['CAR_USE_Commercial'].sum()/df['CAR_USE_Commercial'].count()
    else:
        targetP = 1
    flag = False
    
    # Compute negatives
    if targetP != 1:
        targetN = 1 - targetP
    else:
        entropy = 0
        flag = True
        
    if targetP == 0:
        entropy = 0
        flag = True
        
    # Calculate entropy
    if flag == False:
        entropy = (-(targetP) * math.log(targetP, 2)) - ((targetN) * math.log(targetN, 2))
    return entropy

# Finding the optimal entropy
def entropyOptimal(df, series):
    entropy = []
    entropyList= []
    
    # Return dataframe combinations
    for i in range(1,math.ceil(len(df[series].unique())/2) + 1):
        entropyList += combinations(df[series].unique(),i)
        
    for i in entropyList:
        temp = pandas.DataFrame()
        for x in i:
            q = df.query("{} == '{}'".format(series, x))
            frames = [temp, q]
            temp = pandas.concat(frames)
            
        q = df.copy()
        for x in i:
            q = q.query("{} != '{}'".format(series, x))
        entropyTotal = (temp["ID"].count() / df["ID"].count()) * (entropyCalculation(temp)) + (q["ID"].count()/df["ID"].count()) * (entropyCalculation(q))
        entropy.append((i, entropyTotal))
    return entropy

def entropyOptimalOrdinal(df, series):
    entropy = []
    entropyList = df[series].unique()
    
    # Returns dataframe combinations
    for i in entropyList:
        temp = pandas.DataFrame()
        q = df.query("{} <= {}".format(series, i))
        frames = [temp, q]
        temp = pandas.concat(frames)
        
        q = df.query("{} > {}".format(series, i))
        entropyTotal = (temp["ID"].count() / df["ID"].count()) * (entropyCalculation(temp)) + (q["ID"].count()/df["ID"].count()) * (entropyCalculation(q))
        entropy.append((i, entropyTotal))
    return entropy

# Calculate the entropies of all the features
carType = entropyOptimal(claimHistory, "CAR_TYPE")
occupation = entropyOptimal(claimHistory, "OCCUPATION")
education = entropyOptimalOrdinal(claimHistory, "EDUCATION_LEVEL")

# Calculate the optimal splits
def splitOptimal(combos):
    minInt = 0
    minVar = combos[0][1]
    for i in range(len(combos)):
        if minVar > combos[i][1]:
            minVar = combos[i][1]
            minInt = i
    return combos[minInt]

print("\n1.b\nCar Type: ", splitOptimal(carType), "\nOccupation: ", splitOptimal(occupation), "\nEducation Level: ", splitOptimal(education))

## Question 2 ##
Sample = pandas.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment3\\sample_v10.csv", delimiter=',')

# Set some options for printing all the columns
np.set_printoptions(precision = 10, threshold = sys.maxsize)
np.set_printoptions(linewidth = np.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

# Selecting the properties
intName = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6',  'x7', 'x8', 'x9', 'x10']
yName = 'y'
nPredictor = len(intName)

# 2.a
trainData = Sample[['y'] + intName].dropna()
del(Sample)

n_sample = trainData.shape[0]

print('\n2.a\n=== Frequency of ' + yName + ' ===')
print(trainData[yName].value_counts())

# 2.b
# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category').copy()
u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Reorder the categories of the categorical variables in ascending frequency
for pred in intName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Generate a column of Intercept
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)

y_train = trainData[yName].copy()

# Train a multinominal logistic model
modelObj = smodel.MNLogit(y_train, X0_train)

print("\n2.b\nName of Target Variable:", modelObj.endog_names)
print("Name(s) of Predictors:", modelObj.exog_names)

thisFit = modelObj.fit(full_output = True)
print('\nModel Summary:\n', thisFit.summary())

print('\nIntercept Model Log-Likelihood Value = ', thisFit.llnull)
print('Current Model Log-Likelihood Value = ', thisFit.llf)

print("\nModel Parameter Estimates:\n", thisFit.params)

# 2.c / 2.d
maxIter = 20
tolS = 1e-7
stepSummary = []

# Intercept only model
resultList = Utility.MNLogisticModel (X0_train, y_train, maxIter = maxIter, tolSweep = tolS)

llk0 = resultList[1]
df0 = resultList[2]
stepSummary.append(['Intercept', ' ', df0, llk0, np.NaN, np.NaN, np.NaN])

print("\n2.c / 2.d\n======= Step Detail =======")
print('Step = ', 0)
print('Step Statistics:')
print(stepSummary)

iName = intName.copy()
entryThreshold = 0.05
# The Deviance significance is the sixth element in each row of the test result
def takeDevSig(s):
    return s[6]

for step in range(nPredictor):
    enterName = ''
    stepDetail = []

    # Enter the next predictor        
    for X_name in iName:
        X_train = trainData[[X_name]]
        X_train = X0_train.join(X_train)
        resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[1]
        df1 = resultList[2]
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        AIC = 2.0 * llk0 - 2.0 * df0
        BIC = llk0 * np.log(n_sample) - 2.0 * df0
        stepDetail.append([X_name, 'interval', df1, llk1, devChiSq, devDF, devSig, AIC, BIC])

    # Find a predictor to enter, if any
    # Find a predictor to add, if any
    stepDetail.sort(key = takeDevSig, reverse = False)
    enterRow = stepDetail[0]
    minPValue = takeDevSig(enterRow)
    if (minPValue <= entryThreshold):
        stepSummary.append(enterRow)
        df0 = enterRow[2]
        llk0 = enterRow[3]

        enterName = enterRow[0]
        enterType = enterRow[1]
        if (enterType == 'categorical'):
            X_train = pandas.get_dummies(trainData[[enterName]].astype('category'))
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
print('\n======= Step Summary =======')
stepSummary = pandas.DataFrame(stepSummary, columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig', 'AkaikeInfo', 'BayesianInfo'])
print(stepSummary)