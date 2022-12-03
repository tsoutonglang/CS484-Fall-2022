"""
@Name: Q7.py
@Creation Date: August 22, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas

def univariate (y):

   # Initialize
   y_nvalid = 0
   y_min = None
   y_max = None
   y_mean = None

   # Loop through all the elements
   for u in y:
      if (not numpy.isnan(u)):
         y_nvalid = y_nvalid + 1

         if (y_min is not None):
            if (u < y_min):
               y_min = u
         else:
            y_min = u

         if (y_max is not None):
            if (u > y_max):
               y_max = u
         else:
            y_max = u

         if (y_mean is not None):
            y_mean = y_mean + u
         else:
            y_mean = u

   # Finalize
   if (y_nvalid > 0):
      y_mean = y_mean / y_nvalid

   return (y_nvalid, y_min, y_max, y_mean)

def shimazaki_criterion (y, d_list):

   number_bins = []
   matrix_boundary = []
   criterion = []

   y_nvalid, y_min, y_max, y_mean = univariate (y)

   if (y_nvalid <= 0):
      raise ValueError('There are no non-missing values in the data vector.')
   else:

      # Loop through the bin width candidates
      for delta in d_list:
         y_middle = delta * numpy.round(y_mean / delta)
         n_bin_left = numpy.ceil((y_middle - y_min) / delta)
         n_bin_right = numpy.ceil((y_max - y_middle) / delta)
         y_low = y_middle - n_bin_left * delta

         # Assign observations to bins starting from 0
         list_boundary = []
         n_bin = n_bin_left + n_bin_right
         bin_index = 0
         bin_boundary = y_low
         for i in numpy.arange(n_bin):
            bin_boundary = bin_boundary + delta
            bin_index = numpy.where(y > bin_boundary, i+1, bin_index)
            list_boundary.append(bin_boundary)

         # Count the number of observations in each bins
         uvalue, ucount = numpy.unique(bin_index, return_counts = True)

         # Calculate the average frequency
         mean_ucount = numpy.mean(ucount)
         ssd_ucount = numpy.mean(numpy.power((ucount - mean_ucount), 2))
         value = (2.0 * mean_ucount - ssd_ucount) / delta / delta

         number_bins.append(n_bin)
         matrix_boundary.append(list_boundary)
         criterion.append(value)
        
   return(number_bins, matrix_boundary, criterion)

input_data = pandas.read_csv('C:\\IIT\\Machine Learning\\Test\\2022S2\\1000Values.csv')
x = input_data['Value']

x.describe()

d_list = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50]
number_bins, matrix_boundary, criterion = shimazaki_criterion (x, d_list)

result = pandas.DataFrame([d_list, number_bins, criterion]).transpose()
result.columns = ['Bin Width', 'Number of Bins', 'Criterion']

for delta, bin_boundary in zip(d_list, matrix_boundary):
   plt.figure(figsize = (10,6), dpi = 200)
   plt.hist(x, bins = bin_boundary, align = 'mid')
   plt.title('Delta = ' + str(delta))
   plt.ylabel('Number of Observations')
   plt.grid(axis = 'y')
   plt.show()

