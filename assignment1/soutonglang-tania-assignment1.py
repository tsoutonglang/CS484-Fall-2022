# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 01:11:04 2022

@Name: Soutonglang-Tania-Homework1
@author: Tania Soutonglang
"""

import numpy
import pandas
import math
import matplotlib.pyplot as plt
from scipy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier as kNN

# 1.a
normalSample = pandas.read_csv('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment1\\NormalSample.csv', delimiter=',')
X = normalSample["x"]
print("1.a\n", normalSample.describe())

# 1.b
izenman = round(2*(324-304)*1000**(-1/3), 1)
print("\n1.b\nBin Width = ", izenman)

# 1.c
def calcCD (X, delta):
   maxX = numpy.max(X)
   minX = numpy.min(X)
   meanX = numpy.mean(X)

   # Round the mean to integral multiples of delta
   middleX = delta * numpy.round(meanX / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = numpy.ceil((maxX - middleX) / delta)
   nBinLeft = numpy.ceil((middleX - minX) / delta)
   lowX = middleX - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   BIN_INDEX = 0;
   boundaryX = lowX
   for iBin in numpy.arange(m):
      boundaryX = boundaryX + delta
      BIN_INDEX = numpy.where(X > boundaryX, iBin+1, BIN_INDEX)

   # Count the number of observations in each bins
   uBin, binFreq = numpy.unique(BIN_INDEX, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = numpy.sum(binFreq) / m
   ssDevBinFreq = numpy.sum((binFreq - meanBinFreq)**2) / m
   CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
   return(m, middleX, lowX, CDelta)

result = []
deltaList = [1, 2, 2.5, 5, 10, 20, 25, 50]

for d in deltaList:
   nBin, middleX, lowX, CDelta = calcCD(X,d)
   highX = lowX + nBin * d
   result.append([d, CDelta, lowX, middleX, highX, nBin])
   binMid = lowX + 0.5 * d + numpy.arange(nBin) * d
   
result = pandas.DataFrame(result, columns = {0:'Delta', 1:'C(Delta)', 2:'Low X', 3:'Middle X', 4:'High X', 5:'N Bin'})
print("\n1.c\n", result)

# 1.d
plt.hist(X, bins = [(263+(x*10)) for x in range(11)], align='mid', density=True)
plt.title('1.d Density Estimator')
plt.xlim(263, 363)
plt.xticks([(263+(x*10)) for x in range(11)])
plt.ylabel('Density')
plt.grid(True)
plt.show()

# 2.a
print("\n2.a")
Group0 = normalSample[normalSample['group'] == 0]
print("Group 0:\n", Group0.describe())
q1 = 294
q3 = 306
IQR = q3 - q1
whisker1 = q3-IQR*1.5
whisker2 = q3+IQR*1.5
print("whisker 1: ", whisker1)
print("whisker 2: ", whisker2)

Group1 = normalSample[normalSample['group'] == 1]
print("\nGroup 1:\n", Group1.describe())
q1 = 314
q3 = 327
IQR = q3 - q1
whisker1 = q3-IQR*1.5
whisker2 = q3+IQR*1.5
print("whisker 1: ", whisker1)
print("whisker 2: ", whisker2)

# 2.b
fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot of x')
ax1.boxplot(X, labels = ['X'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()

# fig1, ax1 = plt.subplots()
# ax1.set_title('Box Plot of Group 0')
# ax1.boxplot(Group0, labels = ['Group 0'])
# ax1.grid(linestyle = '--', linewidth = 1)
# plt.show()

# fig1, ax1 = plt.subplots()
# ax1.set_title('Box Plot of Group 1')
# ax1.boxplot(Group0, labels = ['Group 1'])
# ax1.grid(linestyle = '--', linewidth = 1)
# plt.show()

# 3.a
print("\n3.a")
Fraud = pandas.read_csv('C:\\Users\\tsout\\OneDrive\\Desktop\\cs484\\assignment1\\Fraud.csv', delimiter=',')
FraudCol = Fraud["FRAUD"]                # isolate fraud column
fraudCount = FraudCol.value_counts()       # count the number of 0 and 1 in fraud column
fraudOcc = fraudCount[1].astype(int)          # find the number of frauds
print("number of frauds: ", fraudOcc)
fraudColCount = numpy.size(FraudCol,0)   # total number of rows in the csv file
print("number of rows: ", fraudColCount)
fraudP = round(fraudOcc/fraudColCount*100, 4)             # calculate the probability
print("probability of fraud occuring: ", fraudP,"\n")

# 3.b
print("\n3.b")
X = numpy.matrix(Fraud)
print("Input Matrix = \n", X)
Xtx = X.transpose() * X
print("\nt(x) * x = \n", Xtx)

evals, evecs = LA.eigh(Xtx)
print("\nEigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# find eigenvalues greater than one
evals_1 = evals[evals > 1.0]
evecs_1 = evecs[:,evals > 1.0]

# transformation matrix
dvals = 1.0 / numpy.sqrt(evals_1)
transf = evecs_1 * numpy.diagflat(dvals)
print("\nTransformation Matrix = \n", transf)

# transformed X
transf_x = X * transf;
transf_y = Fraud["FRAUD"]
print("\nThe Transformed x = \n", transf_x)

# columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("\nExpect an Identity Matrix = \n", xtx)

# 3.c
# kNN_score = kNN.score(Fraud, X, sample_weight=None)

# 3.d
# test = [[16300, 0, 9, 0, 80, 2, 2]] * transf

kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
# nbrs = kNNSpec.fit(transf_x, )
# neighbors = nbrs.kneighbors(test, return_distance = False)
# print("\n3.d\n", neighbors)

# 4.a
data = pandas.DataFrame([['American', 'Cathay Pacific', 'ORD', 'LAX', 'HKG', 'PVG'],
        ['American', 'Cathay Pacific',	'ORD', 'SFO', 'HKG', 'PVG'],
        ['American', 'China Southern', 'ORD', 'LAX', 'CAN', 'PVG'],
        ['American', 'Virgin Atlantic', 'ORD', 'LHR', '___', 'PVG'],
        ['British Airways', 'Virgin Atlantic', 'ORD', 'LHR', '___', 'PVG'],
        ['United' 'Virgin Atlantic', 'ORD', 'LHR', '___', 'PVG'],
        ['United', '___', 'ORD', 'DCA', 'EWR',	'PVG'],
        ['United', '___', 'ORD', 'DEN', 'LAX',	'PVG'],
        ['United', '___', 'ORD', 'EWR', '___', 'PVG'],
        ['United', '___', 'ORD', 'IAD', 'EWR', 'PVG'],
        ['United', '___', 'ORD', 'LAS', 'LAX', 'PVG'],
        ['United', '___', 'ORD', 'LAX', '___',	'PVG'],
        ['United', '___', 'ORD', 'LGA', 'EWR',	'PVG']],
        index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
        columns = ['Carrier 1', 'Carrier 2', 'Airport 1', 'Airport 2', 'Airport 3', 'Airport 4'])
x = data['Airport 2'] 
y = data['Airport 3']
plt.scatter(x, y, alpha=(0.5))
plt.title("Airport 2 vs Airport 3")
plt.xlabel('Airport 2')
plt.ylabel('Airport 3')
plt.grid(axis = 'both')
plt.show()

# 4.b
z = x.copy()
z = z.append(y)

print("4.b\nFrequency Table:\n", pandas.DataFrame(z).value_counts())

# 4.c
def CosineD (x, y):
    normX = numpy.sqrt(numpy.dot(x, x))
    normY = numpy.sqrt(numpy.dot(y, y))
    if (normX > 0.0 and normY > 0.0):
       outDistance = 1.0 - numpy.dot(x, y) / normX / normY
    else:
       outDistance = numpy.NaN
    return (outDistance)

                # LAX ___ EWR HKG LHR CAN DCA DEN IAD LAS LGA PVG SFO
x = numpy.array([['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0'],
                 ['0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1'],
                 ['1', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0']])