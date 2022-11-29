# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:13:26 2022

@author: tsout
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

wineTrain = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Train.csv")
wineTest = pandas.read_csv(r"C:\Users\tsout\OneDrive\Desktop\cs484\cs484-labs\assignment5\WineQuality_Test.csv")

