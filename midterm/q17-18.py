# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:49:58 2022

@author: tsout
"""
import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import math

DistanceFromChicago = pandas.read_csv('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\midterm\\ChicagoCompletedPotHole.csv',
                      delimiter=',')

trainData = DistanceFromChicago[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']].dropna()

for i in range(len(trainData)):
    trainData['N_POTHOLES_FILLED_ON_BLOCK'][i] = math.log(trainData['N_POTHOLES_FILLED_ON_BLOCK'][i])
    
for i in range(len(trainData)):
    trainData['N_DAYS_FOR_COMPLETION'][i] = math.log(1 + trainData['N_DAYS_FOR_COMPLETION'][i])

nCity = trainData.shape[0]

# Determine the number of clusters
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
Calinski_Harabasz = numpy.zeros(maxNClusters)
Davies_Bouldin = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=2022484).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_

   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
       Calinski_Harabasz[c] = metrics.calinski_harabasz_score(trainData, kmeans.labels_)
       Davies_Bouldin[c] = metrics.davies_bouldin_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN
       Calinski_Harabasz[c] = numpy.NaN
       Davies_Bouldin[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCity):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData["N_POTHOLES_FILLED_ON_BLOCK"][i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

plt.plot(nClusters, TotalWCSS, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Total WCSS")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Calinski_Harabasz, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Davies_Bouldin, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

# Display the 4-cluster solution
kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)