# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:04:14 2022

@name: Soutonglang-Tania-Homework2
@author: Tania Soutonglang
"""

import pandas
from mlxtend.frequent_patterns import (apriori, association_rules)
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

## Question 5 ##
Groceries = pandas.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment2\\Groceries.csv")
ListItem = Groceries.groupby(['Customer'])['Item'].apply(list).values.tolist()

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)
nCustomer, nItem = ItemIndicator.shape

# 5.a
print(Groceries["Customer"].value_counts())
minSupport = 75/9835
maxLength = 32
freqItemsets = apriori(ItemIndicator, min_support = minSupport, max_len = maxLength)
print("5.a\n", freqItemsets)

# 5.b
assoc_rules = association_rules(freqItemsets, metric = "confidence", min_threshold = 0.01)
print("5.b\n", assoc_rules)

# 5.c
plt.figure(figsize=(10,6), dpi = 300)
plt.scatter(assoc_rules['confidence'], assoc_rules['support'],
            c = assoc_rules['lift'], s = 5**assoc_rules['lift'])
plt.grid(True, axis = 'both')
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.colorbar().set_label('lift')
plt.show()

# 5.d
assoc_rules = association_rules(freqItemsets, metric = "confidence", min_threshold = 0.6)
print(assoc_rules.columns)
print(assoc_rules['antecedents'])
print(assoc_rules['consequents'])
print(assoc_rules['support'])
print(assoc_rules['consequent support'])
print(assoc_rules['lift'])
                    
## Questions 6 ##
TwoFeatures = pandas.read_csv("C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment2\\TwoFeatures.csv")

# 6.a
plt.scatter(TwoFeatures['x1'], TwoFeatures['x2'])
plt.grid(axis = 'both')
plt.title("6.a) x1 vs x2")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 6.b
minClusters = 1
maxClusters = 8

trainData = TwoFeatures[['x1', 'x2']].dropna()
nObs = trainData.shape[0]

nClusters = np.zeros(maxClusters)
elbow = np.zeros(maxClusters)
twcss = np.zeros(maxClusters)

def KMeansCluster (trainData, nCluster, nIteration = 500, tolerance = 1e-7):

   # Initialize
   X = trainData.to_numpy()
   centroid = trainData.iloc[range(nCluster)]
   member_prev = np.zeros(trainData.shape[0])

   for iter in range(nIteration):
      distance = metrics.pairwise.manhattan_distances(X, centroid)
      member = np.argmin(distance, axis = 1)
      wc_distance = np.min(distance, axis = 1)
      
# =============================================================================
#       print('==================')
#       print('Iteration = ', iter)
#       print('Centroid: \n', centroid)
#       print('Distance: \n', distance)
#       print('Member: \n', member)
#       print('WCSS: \n', wc_distance)
# =============================================================================

      for cluster in range(nCluster):
         inCluster = (member == cluster)
         if (np.sum(inCluster) > 0):
            centroid.iloc[cluster,:] = np.mean(X[inCluster,], axis = 0)

      member_diff = np.sum(np.abs(member - member_prev))
      if (member_diff > 0):
          member_prev = member
      else:
          break

   return (member, centroid, wc_distance)
    
for c in range(maxClusters):
    KClusters = c + 1
    nClusters[c] = KClusters

    (memb, cent, wcd) = KMeansCluster(trainData, nCluster=KClusters)
    wcss = wcd * wcd
    
    WCSS = np.zeros(KClusters)
    nC = np.zeros(KClusters)
    
    twcss[c] = 0
    elbow[c] = 0
    
    for k in range(KClusters):
        nC[k] = np.sum(np.where(memb == k, 1, 0))
        WCSS[k] = np.sum(np.where(memb == k, wcss, 0.0))
        elbow[c] += WCSS[k] / nC[k]
        twcss[c] += WCSS[k]

print("Number of Clusters\t Total WCSS\t Elbow Value")
for c in range(maxClusters):
    print('{:.0f} \t {:.4f} \t {:.4f}'
          .format(nClusters[c], twcss[c], elbow[c]))

# 6.c
plt.plot(nClusters, elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.title("6.b) Elbow Values")
plt.xticks(np.arange(1, maxClusters+1, step = 1))
plt.show()

# 6.d
# x1 scaling
scale1 = TwoFeatures['x1'].max() - TwoFeatures['x1'].min()
nscale1 = (10-0)
TwoFeatures['x1_scaled'] = (((TwoFeatures['x1'] - TwoFeatures['x1'].min()) * nscale1) / scale1) + 0
print(TwoFeatures['x2'].describe())
print("--------")
print(TwoFeatures['x1_scaled'].describe())

# x2 scaling
scale2 = TwoFeatures['x2'].max() - TwoFeatures['x2'].min()
nscale2 = (10-0)
TwoFeatures['x2_scaled'] = (((TwoFeatures['x2'] - TwoFeatures['x2'].min()) * nscale2) / scale2) + 0

TwoFeaturesScaled = pandas.DataFrame(data = [TwoFeatures['x1_scaled'] , TwoFeatures['x2_scaled']])
TwoFeaturesScaled = TwoFeaturesScaled.transpose()

minClusters = 1
maxClusters = 8

trainData = TwoFeaturesScaled
nObs = trainData.shape[0]

nClusters = np.zeros(maxClusters)
elbow = np.zeros(maxClusters)
twcss = np.zeros(maxClusters)

def KMeansCluster (trainData, nCluster, nIteration = 500, tolerance = 1e-7):

   # Initialize
   X = trainData.to_numpy()
   centroid = trainData.iloc[range(nCluster)]
   member_prev = np.zeros(trainData.shape[0])

   for iter in range(nIteration):
      distance = metrics.pairwise.manhattan_distances(X, centroid)
      member = np.argmin(distance, axis = 1)
      wc_distance = np.min(distance, axis = 1)

# =============================================================================
#       print('==================')
#       print('Iteration = ', iter)
#       print('Centroid: \n', centroid)
#       print('Distance: \n', distance)
#       print('Member: \n', member)
#       print('WCSS: \n', wc_distance)
# =============================================================================

      for cluster in range(nCluster):
         inCluster = (member == cluster)
         if (np.sum(inCluster) > 0):
            centroid.iloc[cluster,:] = np.mean(X[inCluster,], axis = 0)

      member_diff = np.sum(np.abs(member - member_prev))
      if (member_diff > 0):
          member_prev = member
      else:
          break

   return (member, centroid, wc_distance)
    
for c in range(maxClusters):
    KClusters = c + 1
    nClusters[c] = KClusters

    (memb, cent, wcd) = KMeansCluster(trainData, nCluster=KClusters)
    wcss = wcd * wcd
    
    WCSS = np.zeros(KClusters)
    nC = np.zeros(KClusters)
    
    twcss[c] = 0
    elbow[c] = 0
    
    for k in range(KClusters):
        nC[k] = np.sum(np.where(memb == k, 1, 0))
        WCSS[k] = np.sum(np.where(memb == k, wcss, 0.0))
        elbow[c] += WCSS[k] / nC[k]
        twcss[c] += WCSS[k]

print("Number of Clusters\t Total WCSS\t Elbow Value")
for c in range(maxClusters):
    print('{:.0f} \t {:.4f} \t {:.4f}'
          .format(nClusters[c], twcss[c], elbow[c]))
    
# 6.e
plt.plot(nClusters, elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.title("6.e) Elbow Values")
plt.xticks(np.arange(1, maxClusters+1, step = 1))
plt.show()