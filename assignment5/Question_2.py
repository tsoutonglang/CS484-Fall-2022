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
from sklearn import metrics

from matplotlib.ticker import FormatStrFormatter
from numpy.random import default_rng

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

wineTrain = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Train.csv")
wineTest = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Test.csv")

## Question 2 ##
x_Name = ['alcohol', 'free_sulfur_dioxide', 'sulphates', 'citric_acid', 'residual_sugar']

x_Train = wineTrain[["alcohol", "free_sulfur_dioxide", "sulphates", "citric_acid", "residual_sugar"]]
y_Train = wineTrain["quality_grp"]

x_Test = wineTrain[["alcohol", "free_sulfur_dioxide", "sulphates", "citric_acid", "residual_sugar"]]
y_Test = wineTrain["quality_grp"]

y_Test = wineTrain['quality_grp'].astype('category')
y_category = y_Test.cat.categories
x_Train = wineTrain[x_Name]
x_Train = stats.add_constant(x_Train, prepend=True)

y_Test = wineTrain['quality_grp'].astype('category')
x_Test = wineTrain[x_Name]
x_Test = stats.add_constant(x_Test, prepend=True)

logit = stats.MNLogit(y_Train, x_Train)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(x_Train)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

##calculate the AUC metric of the Training data
y_predProb = thisFit.predict(x_Train)[1]
fpr, tpr, thresholds = roc_curve(y_Train, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('Train AUC =', AUC)

##calculate the AUC metric of the Testing data
y_predProb = thisFit.predict(x_Test)[1]
fpr, tpr, thresholds = roc_curve(y_Train, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('Test AUC =', AUC)

y_predictClass = y_category[y_predict]
y_confusion = metrics.confusion_matrix(y_Train, y_predictClass)
print("Confusion Matrix (Row is Data, Column is Predicted) = \n")
print(y_confusion)
y_accuracy = metrics.accuracy_score(y_Train, y_predictClass)
print("Accuracy Score = ", y_accuracy)

# Set the random seed
rng = default_rng(20220901)

# Specifications
n_sample = 101
normal_mu = 10
normal_std = 2

# Generate X from a Normal with mean = 10 and sd = 7
x_sample = norm.rvs(loc = normal_mu, scale = normal_std, size = n_sample, random_state = rng)
print('Sample Median: {:.7f}'.format(numpy.median(x_sample)))

for n_trial in [100, 1000, 10000, 100000]:
   time_begin = time.time()

   random.seed(a = 20221225)
   boot_result = [numpy.nan] * n_trial

   for i_trial in range(n_trial):
      boot_index = [-1] * n_sample
      for i in range(n_sample):
         j = int(random.random() * n_sample)
         boot_index[i] = j
      boot_sample = x_sample[boot_index]
      boot_result[i_trial] = numpy.median(boot_sample)

   elapsed_time = time.time() - time_begin

   print('\n')
   print('Number of Trials: ', n_trial)
   print('Elapsed Time: {:.7f}'.format(elapsed_time))
   
   print('Bootstrap Statistics:')
   print('                 Number:', n_trial)
   print('Number of Failed Trials:', numpy.sum(numpy.isnan(boot_result)))
   print('                   Mean: {:.7f}' .format(numpy.mean(boot_result)))
   print('     Standard Deviation: {:.7f}' .format(numpy.std(boot_result, ddof = 1)))
   print('95% Confidence Interval: {:.7f}, {:.7f}'
         .format(numpy.percentile(boot_result, (2.5)), numpy.percentile(boot_result, (97.5))))

   fig, ax = plt.subplots(1,1,figsize = (8,6), dpi = 200)
   ax.hist(boot_result, density = True, align = 'mid', bins = 20)
   ax.set_title('Number of Bootstraps = ' + str(n_trial))
   ax.set_xlabel('Medians of Boostrap Samples')
   ax.set_ylabel('Percent of Bootstrap Samples')
   ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
   # plt.hist(binwidth=0.01)
   plt.grid(axis = 'both')
   plt.show()
