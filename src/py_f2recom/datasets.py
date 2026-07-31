"""
Defining parameters and datasets for FESOM2-REcoM2 analysis.
To be adapted to your needs
"""

import socket
import sys
import os
import numpy as np
import time
from datetime import date
home = os.path.expanduser("~")


# defines paths ----------------------------------------------------------------------------------------- 

simu_name = 'A' # only usefull if you want to save figures

if socket.gethostname()[:5] == 'blogi':
    meshpath = '/scratch/usr/hbkoziel/mesh/farc'
    resultpath = '/scratch/projects/hbk00083/model_outputs/fesom2.1_recom'+simu_name+'/'
    savepath = home+'/pyfesom2/codes/py_f2recom_develop/outputs/'+simu_name+'/'
    evalpath      = '/scratch/usr/hbkoziel/evaluation/'
    evalpath2      = '/scratch/usr/hbkoziel/corrected_input/'
elif socket.gethostname()[:5] in ['albed','prod-','fat-0']:
    #resultpath = '/albedo/work/projects/p_bio/model_output/A_riv'
    #resultpath = '/albedo/work/projects/p_bio/model_output/RivPI-ClimANT_1850_1957'
    #resultpath = '/albedo/work/projects/p_oceanpeak/Simone/FESOM2.7/1800_sp/'
    #resultpath ='/albedo/work/user/silech001/'
    resultpath = '/albedo/scratch/user/simuel001/phd/output_JRA/a_varClim_1958_2024'
    #resultpath = '/albedo/work/projects/MarESys/GCB2023/NEW1/A/'
    savepath = home+'/test_py_f2recom_submodules/py_f2recom/outputs/'+simu_name+'/'
    evalpath = '/albedo/work/projects/p_pool_recom/eval/'
    meshpath = '/albedo/work/projects/p_bio/mesh/core2/'
    #meshpath = '/albedo/work/user/yye/fesom2/meshes/core2_albedo/'
else:
    print('sorry, machine unknown, please customize your paths yourself')


# period of analysis ------------------------------------------------------------------------------------
first_year_maps = 2020
first_year = 2000
last_year  = 2023
    
years = np.arange(first_year,last_year+1,1)
years_last10 = np.arange(first_year_maps,last_year+1,1)

# specification of analysis ------------------------------------------------------------------------------------
layerwise = False
depths = (0,50,200,1000,2000) # If layerwise is True, you can define depths here, by defaut: (0,50,200,1000,2000,4000)
uplow = [0, 100]
mapproj = 'rob'

# export of analysis ------------------------------------------------------------------------------------
# Be aware that exporting figures may alter (crop) the display but the printed figures are okay
# This is because of bugs in the 'constrained_layout' matplotlib experimental function 
# that may be fixed in the future matplotlib version but out of our control
# If you prefer having a nice HTML, savefig must be turned off
#--------------------------------------------------------------------------------------------------------
today = date.today().strftime("_%Y_%m_%d")
savefig = True 
htmlname     =  simu_name+'_'+ today +'_ocean_ice.html'
htmlpath = savepath
verbose = True

if not os.path.exists(htmlpath): # create folders if do not exist
    os.makedirs(htmlpath)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
# initialization file specifications -----------------------------------------------------------
ncfileTemp               = evalpath+'data/woa18_decav_t00_01_fesom2.nc'
ncfileSal                = evalpath+'data/woa18_decav_s00_01_fesom2.nc'
ncfilePHC3               = evalpath+'phc3.0_annual.nc'      
matfileMLD               = evalpath+'GlobalML_Climato_1970_2018.mat'
ncfilesic                = evalpath+'NSIDC/*.csv'
ncfileDSi                = evalpath+'data/woa13_all_i00_01_fesom2.nc'
ncfileDIN                = evalpath+'data/woa13_all_n00_01_fesom2.nc'
ncfileDO2                = evalpath+'data/woa18_all_o00_01_mmol_fesom2.nc'
ncfileDFe_pisces         = evalpath+'fe_pisces_opa_eq_init_3D.nc' 
ncfileDFe_Huangetal      = evalpath+'Monthly_dFe_Huang2021.nc' 
ncfileAlk                = evalpath+'data/GLODAPv2.2016b.TAlk_mmol.nc'
ncfileDIC                = evalpath+'data/GLODAPv2.2016b.TCO2_mmol.nc'
ncfilepCO2               = evalpath+'SOCATv2020_tracks_gridded_monthly.nc'
ncfileCO2f               = evalpath+'dataset_CO2_Chauetal2020.nc'
txtfileCO2flux           = evalpath+'CO2_flux2015_Takahashietal2009_original.txt'
matfileChlGloOCCCI       = evalpath+'climatology_annual_chl_1deg_OCCCI_2012_2015.mat'
ncfileChlSouthernJohnson = evalpath+'Johnson2013_MEAN_1x1_Chl_mg_m3.npy'
matfileNPPvgpm           = evalpath+'VGPM_CLIM.mat'
matfileNPPcpbm           = evalpath+'CBPM_CLIM.mat'
ncfileMaredatDia         = evalpath+'MarEDat20120716Diatoms.nc'
ncfileMaredatCocco       = evalpath+'MarEDat20130523Coccolithophores.nc'
ncfileMaredatPhaeo       = evalpath+'MarEDat20120424Phaeocystis_filtered.nc'
ncfileMaredatMicro       = evalpath+'MarEDat20120424Microzooplankton.nc'
ncfileMaredatMeso        = evalpath+'MarEDat20120524Mesozooplankton.nc'
ncfileMaredatMacro       = evalpath+'MarEDat20120216Macrozooplankton.nc'
ncfileChlArcLewis        = evalpath+'AO_SAT/LEWIS_CLIMATOLOGY_2003_2021_CHL.nc'
ncfileChlArcOCCCI        = evalpath+'AO_SAT/OCCCI_CLIMATOLOGY_2000_2019_May_Sept_CHL.nc'
npfileNPPArcLewis        = evalpath+'NPP_ARRIGO_2003_2018_reg.npy'
npfileNPParcCMEMS        = evalpath+'NPP_CMEMS_2003_2018_reg.npy'


# visual check
if(verbose):
    print('\n==> Processing years from {4} to {5}\n\nReading out of {0}\nStoring graphs to {1}\nStoring html to {2} as {3}'.format(
        resultpath, savepath, htmlpath, htmlname,years[0],years[-1]))
    #print('\nLast ten years are \n{0}'.format(years_last10))

print('\n==> You can configure the post-processing parameters & paths in : py_f2recom/src/datasets.py\n')