'''
How to create a target diagram with overlaid markers

A sixth example of how to create a target diagram given one set of
reference observations and multiple model predictions for the quantity.

This example is a variation on the third example (target3) where now a
fourth data point having a negative bias is overlaid on an existing
target diagram that already has 3 data points with positive bias. It is
chosen to have the first 3 data points appear in red while the fourth
data point is displayed in blue.

All functions in the Skill Metrics library are designed to only work with
one-dimensional arrays, e.g. time series of observations at a selected
location. The one-dimensional data are read in as dictionaries via a 
pickle file: ref['data'], pred1['data'], pred2['data'], 
and pred3['data']. The plot is written to a file in Portable Network 
Graphics (PNG) format.

The reference data used in this example are cell concentrations of a
phytoplankton collected from cruise surveys at selected locations and 
time. The model predictions are from three different simulations that
have been space-time interpolated to the location and time of the sample
collection. Details on the contents of the dictionary (once loaded) can 
be obtained by simply executing the following two statements

>> key_to_value_lengths = {k:len(v) for k, v in ref.items()}
>> print key_to_value_lengths
{'units': 6, 'longitude': 57, 'jday': 57, 'date': 57, 'depth': 57, 
'station': 57, 'time': 57, 'latitude': 57, 'data': 57}

Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com

Created on Dec 3, 2016

@author: prochford@thesymplectic.com
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import skill_metrics as sm
from sys import version_info

def load_obj(name):
    # Load object from file in pickle format
    if version_info[0] == 2:
        suffix = 'pkl'
    else:
        suffix = 'pkl3'

    with open(name + '.' + suffix, 'rb') as f:
        return pickle.load(f) # Python2 succeeds

class Container(object): 
    
    def __init__(self, pred1, pred2, pred3, ref):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.ref = ref
        
if __name__ == '__main__':
    
    # Close any previously open graphics windows
    # ToDo: fails to work within Eclipse
    plt.close('all')
        
    # Read data from pickle file
    data = load_obj('target_data')

    # Calculate statistics for target diagram
    target_stats1 = sm.target_statistics(data.pred1,data.ref,'data')
    target_stats2 = sm.target_statistics(data.pred2,data.ref,'data')
    target_stats3 = sm.target_statistics(data.pred3,data.ref,'data')
     
    # Store statistics in arrays
    bias = np.array([target_stats1['bias'], target_stats2['bias'], 
                     target_stats3['bias']])
    crmsd = np.array([target_stats1['crmsd'], target_stats2['crmsd'], 
                      target_stats3['crmsd']])
    rmsd = np.array([target_stats1['rmsd'], target_stats2['rmsd'], 
                     target_stats3['rmsd']])

    # Specify labels for points in a list (M1 for model prediction # 1, 
    # etc.).
    label = ['M1', 'M2', 'M3']

    '''
    Produce the target diagram

    Label the points and change the axis options. Increase the upper 
    limit for the axes, change color and line style of circles. 
    Increase the line width of circles.

    For an exhaustive list of options to customize your diagram, 
    please call the function at a Python command line:
    >> target_diagram
    '''
    sm.target_diagram(bias,crmsd,rmsd, \
                      markerLabel = label, markerLabelColor = 'r', \
                      ticks = np.arange(-60,70,10), axismax = 50, \
                      circleLineSpec = '-.b', circleLineWidth = 1.5)

    # Calculate a negative uRMSD for one of the data values.
    data.pred3['data'] = -data.pred3['data']
    target_stats3 = sm.target_statistics(data.pred3,data.ref,'data')
    bias = target_stats3['bias']
    crmsd = target_stats3['crmsd']
    rmsd = target_stats3['rmsd']

    # Overlay new data point (blue) on existing diagram
    label = ['M4']
    sm.target_diagram(bias,crmsd,rmsd, \
                      overlay = 'on', \
                      markerLabel = label, markerLabelColor = 'b', \
                      markerColor = 'b')

    # Write plot to file
    plt.savefig('target6.png')

    # Show plot
    plt.show()
