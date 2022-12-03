# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:13:26 2022

@author: tsout
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import time
import random

import statsmodels.api as stats
from scipy.stats import norm

from sklearn.metrics import auc, roc_curve
from sklearn import metrics, ensemble, tree

from matplotlib.ticker import FormatStrFormatter
from numpy.random import default_rng

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

# import data
wineTrain = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Train.csv")
wineTest = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Test.csv")

#%% Question 1 %%#
print("--- TRAINING DATA ---")
n_Sample = wineTrain.shape[0]

yTrain_Threshold = numpy.mean(wineTrain['quality_grp'])

x_train = wineTrain[["alcohol", "citric_acid", "free_sulfur_dioxide", "residual_sugar", "sulphates"]]
y_train = wineTrain["quality_grp"]

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
treeFit = classTree.fit(x_train, y_train)
y_predProb = classTree.predict_proba(x_train)
y_predClass = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)

confusion_matrix = metrics.confusion_matrix(y_train, y_predClass)
print(confusion_matrix)

# =============================================================================
# fig, ax = plt.subplots(1, 1, figsize = (16,16), dpi = 200)
# tree.plot_tree(classTree, max_depth=5, label='all', filled=True, impurity=True, ax=ax)
# plt.show()
# =============================================================================

# Build a classification tree on the training partition
max_iteration = 50
w_train = numpy.full(n_Sample, 1.0)

ens_accuracy = numpy.zeros(max_iteration)
y_ens_predProb = numpy.zeros((n_Sample, 2))
misclass = numpy.zeros(max_iteration)

for itnum in range(max_iteration):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
    treeFit = classTree.fit(x_train, y_train, w_train)
    y_predProb = classTree.predict_proba(x_train)
    ens_accuracy[itnum] = classTree.score(x_train, y_train, w_train)
    y_ens_predProb += ens_accuracy[itnum] * y_predProb
    misclass[itnum] = 1 - ens_accuracy[itnum]

    print('\n')
    print('Iteration = ', itnum)
    print('Weighted Accuracy = ', ens_accuracy[itnum])
    print('Misclassification Rate = ', misclass[itnum], "\n")

    if (abs(1.0 - ens_accuracy[itnum]) < 0.0000001):
        break

    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - y_predProb[:,1]), (y_predProb[:,1]))
    y_predClass = numpy.where(y_predProb[:,1] >= 0.2, 1, 0)
    w_train = numpy.where(y_predClass != y_train, 2 + numpy.abs(eventError), numpy.abs(eventError))
    # w_train = numpy.abs(eventError)
    # w_train = numpy.where(y_predClass != y_train, 1.0, 0.0) + w_train

    # print('Event Error:\n', eventError)

y_ens_predProb = y_ens_predProb / numpy.sum(ens_accuracy)

# Calculate the final predicted probabilities
wineTrain['predCluster'] = numpy.where(y_ens_predProb[:,1] >= yTrain_Threshold, 1, 0)
ensembleAccuracy = numpy.mean(numpy.where(wineTrain['predCluster'] == y_train, 1, 0))

## Test data ##
print("--- TEST DATA ---")

n_Sample = wineTest.shape[0]
ytest_Threshold = numpy.mean(wineTrain['quality_grp'])

X_test = wineTest[["alcohol", "citric_acid", "free_sulfur_dioxide", "residual_sugar", "sulphates"]]
y_test = wineTest["quality_grp"]

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
treeFit = classTree.fit(X_test, y_test)
y_predProb = classTree.predict_proba(X_test)
y_predClass = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)

testconfusion_matrix = metrics.confusion_matrix(y_test, y_predClass)
print(testconfusion_matrix)

# =============================================================================
# fig, ax = plt.subplots(1, 1, figsize = (16,16), dpi = 200)
# tree.plot_tree(classTree, max_depth=5, label='all', filled=True, impurity=True, ax=ax)
# plt.show()
# =============================================================================

# Build a classification tree on the training partition
max_iteration = 18
w_test = numpy.full(n_Sample, 1.0)

ens_accuracy = numpy.zeros(max_iteration)
y_ens_predProb = numpy.zeros((n_Sample, 2))
testmisclass = numpy.zeros(max_iteration)

for itnum in range(max_iteration):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
    treeFit = classTree.fit(X_test, y_test, w_test)
    ytest_predProb = classTree.predict_proba(X_test)
    ens_accuracy[itnum] = classTree.score(X_test, y_test, w_test)
    y_ens_predProb += ens_accuracy[itnum] * ytest_predProb
    testmisclass[itnum] = 1 - ens_accuracy[itnum]

    print('\n')
    print('Iteration = ', itnum)
    print('Weighted Accuracy = ', ens_accuracy[itnum])
    print('Misclassification Rate = ', misclass[itnum], "\n")

    if (abs(1.0 - ens_accuracy[itnum]) < 0.0000001):
        break

    # Update the weights
    testeventError = numpy.where(y_test == 1, (1 - ytest_predProb[:,1]), (ytest_predProb[:,1]))
    ytest_predClass = numpy.where(ytest_predProb[:,1] >= 0.2, 1, 0)
    w_test = numpy.where(ytest_predClass != y_test, 2 + numpy.abs(testeventError), numpy.abs(testeventError))

y_ens_predProb = y_ens_predProb / numpy.sum(ens_accuracy)

# Calculate the final predicted probabilities
wineTest['predCluster'] = numpy.where(y_ens_predProb[:,1] >= ytest_Threshold, 1, 0)
testensembleAccuracy = numpy.mean(numpy.where(wineTest['predCluster'] == y_test, 1, 0))

# creating box plot
fig1, ax1 = plt.subplots()
ax1.set_title('Quality_grp Predicted Probability')
ax1.boxplot(y_predProb, labels = ['0', '1'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()

#%% QUESTION 2 %%#
x_Name = ['alcohol', 'free_sulfur_dioxide', 'sulphates', 'citric_acid', 'residual_sugar']

# Build a logistic regression
y_Train = wineTrain['quality_grp'].astype('category')
yTrain_category = y_Train.cat.categories

x_Train = wineTrain[x_Name]
x_Train = stats.add_constant(x_Train, prepend=True)

logit = stats.MNLogit(y_Train, x_Train)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(x_Train)
yTrain_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

yTrain_predictClass = yTrain_category[yTrain_predict]

yTrain_confusion = metrics.confusion_matrix(y_Train, yTrain_predictClass)
print("Confusion Matrix (Row is Data, Column is Predicted) = \n")
print(yTrain_confusion)

yTrain_accuracy = metrics.accuracy_score(y_Train, yTrain_predictClass)
print("Accuracy Score = ", yTrain_accuracy)

y_Test = wineTest['quality_grp'].astype('category')
yTest_category = y_Test.cat.categories

x_Test = wineTest[x_Name]
x_Test = stats.add_constant(x_Test, prepend=True)

##calculate the AUC metric of the Training data
y_predProb = thisFit.predict(x_Train)[1]
fpr, tpr, thresholds = roc_curve(y_Train, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('\nTrain AUC =', AUC)

##calculate the AUC metric of the Testing data
y_predProb = thisFit.predict(x_Test)[1]
fpr, tpr, thresholds = roc_curve(y_Test, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('Test AUC =', AUC)

# Set the random seed
rng = default_rng(20221225)

# Specifications
# n_sample = 100000
# normal_mu = 10
# normal_std = 2
n_sample = wineTrain.shape[0]
normal_mu = 8.0658
normal_std = 3.9287

# Generate X from a Normal with mean = 10 and sd = 7
x_sample = norm.rvs(loc = normal_mu, scale = normal_std, size = n_sample, random_state = rng)
print('Sample Median: {:.7f}'.format(numpy.median(x_sample)))

time_begin = time.time()
 
random.seed(a = 20221225)
boot_result = [numpy.nan] * 100000
 
for i_trial in range(100000):
   boot_index = [-1] * 100000
   for i in range(n_sample):
      j = int(random.random() * n_sample)
      boot_index[i] = j
   boot_sample = x_sample[boot_index]
   boot_result[i_trial] = numpy.median(boot_sample)
 
elapsed_time = time.time() - time_begin
 
print('\n')
print('Number of Trials: ', 100000)
print('Elapsed Time: {:.7f}'.format(elapsed_time))

print('Bootstrap Statistics:')
print('                 Number:', 100000)
print('Number of Failed Trials:', numpy.sum(numpy.isnan(boot_result)))
print('                   Mean: {:.7f}' .format(numpy.mean(boot_result)))
print('     Standard Deviation: {:.7f}' .format(numpy.std(boot_result, ddof = 1)))
print('             Percentile: {:.7f} {:.7f} {:.7f}'.format(numpy.percentile(boot_result, (2.5)), numpy.percentile(boot_result, (50)), numpy.percentile(boot_result, (97.5))))
 
fig, ax = plt.subplots(1,1,figsize = (8,6), dpi = 200)
ax.hist(boot_result, density = True, align = 'mid', bins = 50)
ax.set_title('Number of Bootstraps = 100,000')
ax.set_xlabel('Medians of Boostrap Samples')
ax.set_ylabel('Percent of Bootstrap Samples')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
plt.grid(axis = 'both')
plt.show()