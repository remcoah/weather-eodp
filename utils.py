asdadawewqeqeqesdsdadsdds
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path
import sys
import math
from sklearn import feature_selection as feat_select
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
def str_to_date(string):    
    strarr = string.split()
    if len(strarr) <= 1:
        return datetime.strptime(string, '%Y-%m-%d')
    DT = datetime.strptime(string, '%Y/%m/%d %H:%M:%S')
    return DT

adelaide_column_titles = ['Date', 'Total Demand', 'Price Surge','Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)', 'Rainfall (mm)',  \
                        'Direction of Maximum Wind Gust', 'Speed of Maximum Wind Gust (km/h)', 'Time of Maximum Wind Gust', '9am Temperature (Â°C)', \
                        '9am Relative Humidity (%)', '9am Wind Direction', '9am Wind Speed (km/h)', '9am MSL Pressure (hPa)', '3pm Temperature (Â°C)', \
                        '3pm Relative Humidity (%)', '3pm Wind Direction', '3pm Wind Speed (km/h)', '3pm MSL Pressure (hPa)']

brisbane_column_titles = ['Date', 'Total Demand', 'Price Surge','Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)', 'Rainfall (mm)', 'Sunshine (hours)',  \
                        'Direction of Maximum Wind Gust', 'Speed of Maximum Wind Gust (km/h)', 'Time of Maximum Wind Gust', '9am Temperature (Â°C)', \
                        '9am Relative Humidity (%)', '9am Cloud Amount (oktas)', '9am Wind Direction', '9am Wind Speed (km/h)', '9am MSL Pressure (hPa)', '3pm Temperature (Â°C)', \
                        '3pm Relative Humidity (%)', '3pm Cloud Amount (oktas)', '3pm Wind Direction', '3pm Wind Speed (km/h)', '3pm MSL Pressure (hPa)']

melb_sydney_column_titles = ['Date', 'Total Demand', 'Price Surge','Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)', 'Rainfall (mm)', 'Evaporation (mm)', \
                        'Sunshine (hours)', 'Direction of Maximum Wind Gust', 'Speed of Maximum Wind Gust (km/h)', 'Time of Maximum Wind Gust', '9am Temperature (Â°C)', \
                        '9am Relative Humidity (%)', '9am Cloud Amount (oktas)', '9am Wind Direction', '9am Wind Speed (km/h)', '9am MSL Pressure (hPa)', '3pm Temperature (Â°C)', \
                        '3pm Relative Humidity (%)', '3pm Cloud Amount (oktas)', '3pm Wind Direction', '3pm Wind Speed (km/h)', '3pm MSL Pressure (hPa)']
col_titles = {'SA1':adelaide_column_titles,'QLD1':brisbane_column_titles,'VIC1':melb_sydney_column_titles,'NSW1':melb_sydney_column_titles}
def mean_imputation_column_titles(state):
     return np.intersect1d(col_titles[state],['Total Demand', 'Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Sunshine (hours)', '9am Temperature (Â°C)', '9am Relative Humidity (%)', '9am Cloud Amount (oktas)', '3pm Temperature (Â°C)', '3pm Relative Humidity (%)','3pm Cloud Amount (oktas)'])
def dist_graph_columns(state):
    return np.intersect1d(['Total Demand', 'Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)', 'Rainfall (mm)', 'Evaporation (mm)', 'Sunshine (hours)','9am Temperature (Â°C)', '9am Relative Humidity (%)', '9am Cloud Amount (oktas)', '3pm Temperature (Â°C)', '3pm Relative Humidity (%)','3pm Cloud Amount (oktas)'],col_titles[state])
def box_plot_columns(state):
    return np.intersect1d(col_titles[state],['Total Demand', 'Minimum Temperature (Â°C)', 'Maximum Temperature (Â°C)','9am Temperature (Â°C)', '9am Relative Humidity (%)', '3pm Temperature (Â°C)', '3pm Relative Humidity (%)'])
graph_path = 'Graphs/'
merge_path = 'merged/'
processed_path = 'preprocessed/'
display_names = {'Total Demand':'Total Demand', 'Minimum Temperature (Â°C)':'Min Temperature', 'Maximum Temperature (Â°C)':'Max Temperature',\
                        'Rainfall (mm)':'Rainfall', 'Evaporation (mm)':'Evaporation', 'Sunshine (hours)':'Sunshine', \
                        '9am Temperature (Â°C)':'9am Temp', '9am Relative Humidity (%)':'9am Humidity', '9am Cloud Amount (oktas)':'9am Cloud Count', \
                        '3pm Temperature (Â°C)':'3pm Temp', '3pm Relative Humidity (%)':'3pm Humidity', \
                        '3pm Cloud Amount (oktas)':'3pm Cloud Count'}
windspeed_column_titles = ['Speed of Maximum Wind Gust (km/h)', '9am Wind Speed (km/h)', '3pm Wind Speed (km/h)']

