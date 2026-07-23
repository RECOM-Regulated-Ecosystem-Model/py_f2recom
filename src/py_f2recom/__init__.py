"""
py_f2recom - Python tools for FESOM2-REcoM2 model analysis
"""

__version__ = "0.1.0"
__author__ = """RECOM team"""
__email__ = "laurent.oziel@awi.de"

from .maps import *
from .timeseries import *
from .bars import *
from .profiles import *
from .loading import *
from .core import *
from .cli import *
from .datasets import *


##############################################
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=2)
plt.rc('axes', grid=False)
plt.rc('axes', edgecolor='black')

plt.rc('ytick.major', size = 2)
plt.rc('ytick.major', width = 2)
plt.rc('xtick.minor', visible = True)
plt.rc('xtick.major', size = 2)
plt.rc('xtick.minor', size = 1)
plt.rc('xtick.major', width = 2)
plt.rc('xtick.minor', width = 1)

plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('pdf', fonttype = 42)
#################################################