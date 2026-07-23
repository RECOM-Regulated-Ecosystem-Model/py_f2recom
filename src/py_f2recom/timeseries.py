"""
Visualization tools for timeseries in REcoM model output.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyfesom2 as pf
import numpy as np
import matplotlib.cm as cm
import math 
import skill_metrics as sm
from scipy.interpolate import griddata
from pathlib import Path
from glob import glob as gg
import pandas as pd
from netCDF4 import Dataset
from .loading import *
from .datasets import *
from .core import *

class plot_timeseries_seaice:   
    '''
    class SEAICE_timeseries(runname,resultpath,savepath,mesh,first_year,last_year,savefig=False,regional='N')
    
    Output: class Summer Sea-Ice Extent [million km2]
    Calculated and returned within required region:
    (N = "Arctic", S = "Southern")
    type can be area or extent.
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilesic,
                 first_year=first_year,
                 last_year=last_year,
                 savefig=False,
                 region='N',
                 type = 'area',
                 plotting=True,
                 output=False):

        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.region = region
        self.plotting = plotting
        self.ncpath = ncpath
        self.type = type

        print('## Observations only for the 1979-2019 period ##')
        years = np.arange(self.fyear, self.lyear+1,1)
        files = gg(self.ncpath, recursive = True)
        files = np.sort(files)
        ice = pf.get_data(resultpath,'a_ice', years, mesh, how=None, compute=False, silent=True)
 
        if region == 'N':
            df = pd.read_csv(files[8], sep='\s*,\s*', index_col = 'year', parse_dates=True) #
            df = df.loc[((df.index.year>=self.fyear) & (df.index.year<=self.lyear))]
        elif region == 'S':
            df = pd.read_csv(files[14], sep='\s*,\s*', index_col = 'year', parse_dates=True) #
            df = df.loc[((df.index.year>=self.fyear) & (df.index.year<=self.lyear))]
        
        if type == 'area':
            ice = pf.ice_area(ice, mesh, hemisphere=self.region)
        elif type == 'extent':
            ice = pf.ice_ext(ice, mesh, hemisphere=self.region)

        if region == 'N':
            ice= ice.loc[ice['time'].dt.month==9]
        elif region == 'S':
            ice= ice.loc[ice['time'].dt.month==3]
        
        fig = plt.figure(figsize=(8,5), constrained_layout=True)
        plt.plot(years, ice/1e12, label = 'model', lw =3)
        plt.plot(df.index.year, df[self.type], label = 'satellite', lw =3)
        if region == 'N':
            plt.title('September Arctic Ocean sea-ice')
        elif region == 'S':
            plt.title('March Southern Ocean sea-ice')
        plt.legend()
        plt.ylabel('sea-ice '+self.type+' [million km$^{-2}$]')
        plt.xlabel('time [year]')

        if self.savefig:
            plt.savefig('SEAICE.png', dpi = 300, bbox_inches='tight')
            plt.savefig('SEAICE.pdf', bbox_inches='tight') 

class plot_seasonalcycle_bio: 
    '''
    class SeasonalCycle

    Derive and plot the seasonal distribution of Chl-a and NPP of the four PFTs

    '''

    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 type='Chl',
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='rob',
                 cmap = 'viridis',
                 savefig=False,
                 verbose=True,
                 plotting=True,
                 output=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.type = type
        self.cmap = cmap
        self.savefig = savefig
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.years = np.arange(self.fyear, self.lyear+1,1)
        self.months = np.arange(0,12)
        
        # set indexes to selected regions: 
        # Arctic:
        Arc1 = np.squeeze(np.where((mesh.y2>=60) & (mesh.y2<70)))
        Arc2 = np.squeeze(np.where((mesh.y2>=70) & (mesh.y2<80)))
        Arc3 = np.squeeze(np.where((mesh.y2>=80) & (mesh.y2<90)))
        # SO:
        SO1 = np.squeeze(np.where((mesh.y2<=-40) & (mesh.y2>-50)))
        SO2 = np.squeeze(np.where((mesh.y2<=-50) & (mesh.y2>-60)))
        SO3 = np.squeeze(np.where((mesh.y2<=-60) & (mesh.y2>-70)))

        regions = [("Arc1", Arc1), ("Arc2", Arc2), ("Arc3", Arc3), ("SO1", SO1), ("SO2", SO2), ("SO3", SO3)]

        # load FESOM data ---------------------------------------------------------------------------------------
        if type == 'NPP':
            NPPd = pf.get_data(resultpath, "NPPd", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            NPPn = pf.get_data(resultpath, "NPPn", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            NPPn = NPPn.groupby('time.month').mean().compute()
            NPPd = NPPd.groupby('time.month').mean().compute()

            cocco_path = Path(self.resultpath + '/CoccoChl.fesom.'+str(first_year)+'.nc')
            phaeo_path = Path(self.resultpath + '/PhaeoChl.fesom.'+str(first_year)+'.nc')
            if phaeo_path.is_file():
                NPPp = pf.get_data(resultpath, "NPPp", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
                NPPp = NPPp.groupby('time.month').mean().compute()
            if cocco_path.is_file():
                NPPc = pf.get_data(resultpath, "NPPc", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
                NPPc = NPPc.groupby('time.month').mean().compute()
            
            if 'month' in NPPd.dims:
                datad = np.zeros((len(NPPd.month),len(regions)))*np.nan
                datan = np.zeros((len(NPPn.month),len(regions)))*np.nan
                if phaeo_path.is_file():
                     datap = np.zeros((len(NPPp.month),len(regions)))*np.nan
                if cocco_path.is_file():
                    datac = np.zeros((len(NPPc.month),len(regions)))*np.nan
                    
                i = 0
                for region,mask in regions: 
                    datad[:,i] = pf.areamean_data(NPPd, mesh, mask=mask)
                    datan[:,i] = pf.areamean_data(NPPn, mesh, mask=mask)
                    if phaeo_path.is_file():
                        datap[:,i] = pf.areamean_data(NPPp, mesh, mask=mask)
                    if cocco_path.is_file():
                        datac[:,i] = pf.areamean_data(NPPc, mesh, mask=mask)
                    i = i+1
            else:
                print('data is yearly, seasonal cycle calculation not possible')
                
        elif type == 'Chl':
            NPPn = pf.get_data(resultpath, "PhyChl", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            NPPd = pf.get_data(resultpath, "DiaChl", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            NPPn = NPPn.groupby('time.month').mean().compute()
            NPPd = NPPd.groupby('time.month').mean().compute()

            cocco_path = Path(self.resultpath + '/CoccoChl.fesom.'+str(first_year)+'.nc')
            phaeo_path = Path(self.resultpath + '/PhaeoChl.fesom.'+str(first_year)+'.nc')
            if phaeo_path.is_file():
                NPPp = pf.get_data(resultpath, "PhaeoChl", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
                NPPp = NPPp.groupby('time.month').mean().compute()
            if cocco_path.is_file():
                NPPc = pf.get_data(resultpath, "CoccoChl", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
                NPPc = NPPc.groupby('time.month').mean().compute()

            if 'month' in NPPd.dims:
                datad = np.zeros((len(NPPd.month),len(regions)))*np.nan
                datan = np.zeros((len(NPPn.month),len(regions)))*np.nan
                if phaeo_path.is_file():
                     datap = np.zeros((len(NPPp.month),len(regions)))*np.nan
                if cocco_path.is_file():
                    datac = np.zeros((len(NPPc.month),len(regions)))*np.nan
                    
                i = 0
                for region,mask in regions:
                    datad[:,i] = pf.areamean_data(NPPd[:,:,0], mesh, mask=mask)
                    datan[:,i] = pf.areamean_data(NPPn[:,:,0], mesh, mask=mask)
                    if phaeo_path.is_file():
                        datap[:,i] = pf.areamean_data(NPPp[:,:,0], mesh, mask=mask)
                    if cocco_path.is_file():
                        datac[:,i] = pf.areamean_data(NPPc[:,:,0], mesh, mask=mask)
                    i = i+1
            else:
                print('data is yearly, seasonal cycle calculation not possible')

        #================================
        # Plotting: 

        months_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_name_SO = np.concatenate((months_name[6:], months_name[:6]))

        fig, axes = plt.subplots(2, 3, figsize=(12, 9),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.1}, 
                             facecolor='w', edgecolor='k', tight_layout = True)
        axes = axes.flatten()

        # ARCTIC -----------------------------------------------
        
        axes[0].plot(months_name, datan[:,0], label='SmallPhy',lw=3)
        axes[0].plot(months_name, datad[:,0], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[0].plot(months_name, datac[:,0], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[0].plot(months_name, datap[:,0], label='Phaeo',lw=3)

        #axes[0].set_ylabel(ylabel)
        

        if self.type == 'Chl':
            axes[0].set_ylim(0,1.)
            axes[0].set_ylabel('Chl [mg Chl m$^{-3}$]')
        elif self.type == 'NPP':
            axes[0].set_ylim(0,35)
            axes[0].set_ylabel('NPP [mg C m$^{-2}$ d$^{-1}$]')
        axes[0].set_title('60-70 $^\circ$N')

        axes[1].plot(months_name, datan[:,1], label='SmallPhy',lw=3)
        axes[1].plot(months_name, datad[:,1], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[1].plot(months_name, datac[:,1], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[1].plot(months_name, datap[:,1], label='Phaeo',lw=3)

        
        if self.type == 'Chl':
            axes[1].set_ylim(0,1.)
        elif self.type == 'NPP':
            axes[1].set_ylim(0,35)
        axes[1].set_title('70-80 $^\circ$N')

        axes[2].plot(months_name, datan[:,2], label='SmallPhy',lw=3)
        axes[2].plot(months_name, datad[:,2], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[2].plot(months_name, datac[:,2], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[2].plot(months_name, datap[:,2], label='Phaeo',lw=3)

        if self.type == 'Chl':
            axes[2].set_ylim(0,1.)
        elif self.type == 'NPP':
            axes[2].set_ylim(0,35)
        axes[2].set_title('80-90 $^\circ$N')

        # SO --------------------------------------------------
        
        axes[3].plot(months_name_SO, datan[:,3], label='SmallPhy',lw=3)
        axes[3].plot(months_name_SO, datad[:,3], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[3].plot(months_name_SO, datac[:,3], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[3].plot(months_name_SO, datap[:,3], label='Phaeo',lw=3)

        if self.type == 'Chl':
            axes[3].set_ylim(0,1.)
            axes[3].set_ylabel('Chl [mg Chl m$^{-3}$]')
        elif self.type == 'NPP':
            axes[3].set_ylim(0,35)
            axes[3].set_ylabel('NPP [mg C m$^{-2}$ d$^{-1}$]')
        axes[3].set_title('40-50 $^\circ$S')

        axes[4].plot(months_name_SO, datan[:,4], label='SmallPhy',lw=3)
        axes[4].plot(months_name_SO, datad[:,4], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[4].plot(months_name_SO, datac[:,4], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[4].plot(months_name_SO, datap[:,4], label='Phaeo',lw=3)

        if self.type == 'Chl':
            axes[4].set_ylim(0,1.)
        elif self.type == 'NPP':
            axes[4].set_ylim(0,35)
        axes[4].set_title('50-60 $^\circ$S')

        axes[5].plot(months_name_SO, datan[:,5], label='SmallPhy',lw=3)
        axes[5].plot(months_name_SO, datad[:,5], label='Diatoms',lw=3)
        if cocco_path.is_file():
            axes[5].plot(months_name_SO, datac[:,5], label='Cocco',lw=3)
        if phaeo_path.is_file():
            axes[5].plot(months_name_SO, datap[:,5], label='Phaeo',lw=3)

        if self.type == 'Chl':
            axes[5].set_ylim(0,1.)
        elif self.type == 'NPP':
            axes[5].set_ylim(0,35)
        axes[5].set_title('60-70 $^\circ$S')

        axes[0].set_xticklabels(months_name, rotation=90, ha='right')
        axes[1].set_xticklabels(months_name, rotation=90, ha='right')
        axes[2].set_xticklabels(months_name, rotation=90, ha='right')
        axes[3].set_xticklabels(months_name, rotation=90, ha='right')
        axes[4].set_xticklabels(months_name, rotation=90, ha='right')
        axes[5].set_xticklabels(months_name, rotation=90, ha='right')

        labels=('SmallPhy','Diatoms','Coccos','Phaeo')

        fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6)


class plot_timeseries_npp:
    '''
    class NPP_TotalGlobal(resultpath,savepath,meshpath,first_year,last_year,
                 mapproj='pc',savefig=False,mask="Global Ocean")
                 
    Output:
    self.NPPtotal [Pg C/year]
    self.PhyTotal [Pg C/year]
    self.DiaTotal [Pg C/year]
    self.EPtotal [Pg C/yr]
    self.SiEtotal [Tmol Si/yr]
    
    '''
    def __init__(self,resultpath,savepath,mesh,first_year,last_year,
                 mapproj='pc',
                 plotting = True,
                 savefig=False,
                 mask = "Global Ocean",
                 output = False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.plotting = plotting
        self.savefig = savefig
        self.mask = mask

        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # load nodal area -------------------------------------------------------------------------------------
        meshdiag = pf.get_meshdiag(mesh)
        nod_area_surface = np.ma.masked_equal(meshdiag.nod_area.values, 0)
        
        # get depth of export production
        i_ep_depth = pf.ind_for_depth(100,mesh)
        nod_depth = meshdiag.zbar_n_bottom
        ep_depth = nod_depth[i_ep_depth]
        #print('EP for selected depth = {0} m'.format(ep_depth))
        #print('shape nod_area: {0}\nshape nod_area_surface: {1}'.format(np.shape(nod_area),np.shape(nod_area_surface)))
        
        # check if coccos, second det, and variable sinking velocity are used or not -------------------------------------------------------------------------------
        cocco_path  = Path(self.resultpath + '/NPPc.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path  = Path(self.resultpath + '/NPPp.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        
        secdet_path = Path(self.resultpath + '/idetz2c.fesom.'+str(years[0])+'.nc')
        cram_path   = Path(self.resultpath + '/wsink_det1.fesom.'+str(years[0])+'.nc')
        
        # calculating total NPP per year -------------------------------------------------------------------------------
            
        NPPn = pf.get_data(resultpath, "NPPn", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
        NPPd = pf.get_data(resultpath, "NPPd", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
        if cocco_path.is_file():
            NPPc = pf.get_data(resultpath, "NPPc", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            print('Coccolithophores are used')
        if phaeo_path.is_file():
            NPPp = pf.get_data(resultpath, "NPPp", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            print('Phaeocystis is used')
        DetC1 = pf.get_data(resultpath, "DetC", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
        SiE1 = pf.get_data(resultpath, "DetSi", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
        DetCalc1 = pf.get_data(resultpath, "DetCalc", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
        if secdet_path.is_file():
            DetC2 = pf.get_data(resultpath, "idetz2c", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            SiE2 = pf.get_data(resultpath, "idetz2si", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            DetCalc2 = pf.get_data(resultpath, "idetz2calc", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            print('Second detritus group is used')
        
        ## NPPn:units = "mmolC/(m2*d)"
        
        NPPn = NPPn.resample(time='YS').mean(dim='time').compute()
        NPPd = NPPd.resample(time='YS').mean(dim='time').compute()
        if cocco_path.is_file():
            NPPc = NPPc.resample(time='YS').mean(dim='time').compute()
        if phaeo_path.is_file():
            NPPp = NPPp.resample(time='YS').mean(dim='time').compute()
        SiE1 = SiE1.resample(time='YS').mean(dim='time').compute() 
        DetCalc1 = DetCalc1.resample(time='YS').mean(dim='time').compute()
        DetC1 = DetC1.resample(time='YS').mean(dim='time').compute()
        if secdet_path.is_file():
            DetC2 = DetC2.resample(time='YS').mean(dim='time').compute()
            SiE2 = SiE2.resample(time='YS').mean(dim='time').compute()
            DetCalc2 = DetCalc2.resample(time='YS').mean(dim='time').compute()
        
        # can provide mask or mask name
        if isinstance(mask, str):
            maskstr = mask
            mask = pf.get_mask(mesh, mask)
        else:
            maskstr = mask
            mask = mask
        
        ## Primary Production = "[Pg C/year]"  
        if cocco_path.is_file() & phaeo_path.is_file():
            NPP    = 365* (NPPd+NPPn+NPPc+NPPp)*12.01 /1e18 # Conversion from [mg/m2/day]   => [mg/m2/yr] => [Pg C/year]
            NPPc   = 365* (NPPc)*12.01 /1e18
            NPPp   = 365* (NPPp)*12.01 /1e18
        
        elif cocco_path.is_file() and not phaeo_path.is_file():
            NPP    = 365* (NPPd+NPPn+NPPc)*12.01 /1e18 # Conversion from [mg/m2/day]   => [mg/m2/yr] => [Pg C/year]
            NPPc   = 365* (NPPc)*12.01 /1e18
        
        
        else:
            NPP    = 365* (NPPd+NPPn)*12.01 /1e18 # Conversion from [mg/m2/day]   => [mg/m2/yr] => [Pg C/year]
        NPPd       = 365* (NPPd)*12.01 /1e18 
        NPPn       = 365* (NPPn)*12.01 /1e18 
        
        NPP_timeseries = pf.areasum_data(NPP,mesh,mask)
        NPPd_timeseries = pf.areasum_data(NPPd,mesh,mask)
        NPPn_timeseries = pf.areasum_data(NPPn,mesh,mask)
        if cocco_path.is_file():
            NPPc_timeseries = pf.areasum_data(NPPc,mesh,mask)

        if phaeo_path.is_file():
            NPPp_timeseries = pf.areasum_data(NPPp,mesh,mask)
        
        del NPP, NPPd, NPPn
        if cocco_path.is_file():
            del NPPc
        if phaeo_path.is_file():
            del NPPp
            
        secperyear = 31536000 # = 86400 sec/day * 365 days; necessary to compute yearly fluxes with sinking velocity in m/s
        
        ## Carbon export, DetC:units = "[mmol/m3]"
        if cram_path.is_file(): 
            Vdet1 = pf.get_data(resultpath, "wsink_det1", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
            Vdet1 = Vdet1.resample(time='YS').mean(dim='time').compute()
            print('Cram parameterization with variable sinking velocities is used')
            if secdet_path.is_file():
                Vdet2 = pf.get_data(resultpath, "wsink_det2", years, mesh, how=None, compute=False, runid=self.runname, silent=True)
                Vdet2 = Vdet2.resample(time='YS').mean(dim='time').compute()
        else:
            Vdet1 = 0.0288 * 100. + 20. ## sinking velocity
            if secdet_path.is_file():
                Vdet2 = 200. ## sinking velocity
                        
        if cram_path.is_file():
            print('shape DetC1: ',np.shape(DetC1))
            print('shape Vdet1: ',np.shape(Vdet1))
            detc1 = secperyear  * np.squeeze(DetC1[:,:,i_ep_depth]) * 12.01 * np.squeeze(-Vdet1[:,:,i_ep_depth]) /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]; wsink_det1 is a negative number
            if secdet_path.is_file():
                detc2 = secperyear  * np.squeeze(DetC2[:,:,i_ep_depth]) * 12.01 * np.squeeze(-Vdet2[:,:,i_ep_depth]) /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]; wsink_det2 is a negative number
        else:
            detc1 = 365 * DetC1[:,:,i_ep_depth] * 12.01 * Vdet1 /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]
            if secdet_path.is_file():
                detc2 = 365 *DetC2[:,:,i_ep_depth] * 12.01 * Vdet2 /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]
        
        detct = detc1
        if secdet_path.is_file():
            print('shape detct: ',np.shape(detct))
            print('shape detct2: ',np.shape(detct))
            detct = detct + detc2
        
        EP_timeseries = pf.areasum_data(detct,mesh,mask)
        
        del detc1, detct
        if secdet_path.is_file():
            del detc2
        
        ## Si export, DetSi:units = "[mmol/m3]"
        #SiE = 365. * SiE *28.085 * Vdet /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]
        if cram_path.is_file():
            Sie1 =  secperyear * np.squeeze(SiE1[:,:,i_ep_depth]) * np.squeeze(-Vdet1[:,:,i_ep_depth]) /1e15 # [mmol/m3] => [mmol/m2/yr]  =>  [Tmol Si/yr]; wsink_det1 is a negative number
            if secdet_path.is_file():
                Sie2 = secperyear * np.squeeze(SiE2[:,:,i_ep_depth]) * np.squeeze(-Vdet2[:,:,i_ep_depth]) /1e15 # [mmol/m3] => [mmol/m2/yr]  =>  [Tmol Si/yr]; wsink_det2 is a negative number
        else:
            Sie1 = 365 * SiE1[:,:,i_ep_depth] * Vdet1 /1e15 # [mmol/m3] => [mmol/m2/yr]  =>  [Tmol Si/yr]      
            if secdet_path.is_file():
                Sie2 = 365 * SiE2[:,:,i_ep_depth] * Vdet2 /1e15 # [mmol/m3] => [mmol/m2/yr]  =>  [Tmol Si/yr]   
        
        siect = Sie1
        if secdet_path.is_file():
            siect = siect + Sie2
        
        SiE_timeseries = pf.areasum_data(siect,mesh,mask)   
        
        del Sie1, siect
        if secdet_path.is_file():
            del Sie2
        
        if cram_path.is_file():
            Detcalc1 = secperyear * np.squeeze(DetCalc1[:,:,i_ep_depth]) * 12.01 * np.squeeze(-Vdet1[:,:,i_ep_depth]) /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]; wsink_det1 is a negative number
            if secdet_path.is_file():
                Detcalc2 = secperyear * np.squeeze(DetCalc2[:,:,i_ep_depth]) * 12.01 * np.squeeze(-Vdet2[:,:,i_ep_depth]) /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]; wsink_det2 is a negative number
        else:
            Detcalc1 = 365 * DetCalc1[:,:,i_ep_depth] * 12.01 * Vdet1 /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]
            if secdet_path.is_file():
                Detcalc2 = 365 * DetCalc2[:,:,i_ep_depth] * 12.01 * Vdet2 /1e18 # [mmol/m3] => [mg/m2/yr] => [Pg C/yr]
        
        Detcalcct = Detcalc1
        if secdet_path.is_file():
            Detcalcct = Detcalcct + Detcalc2
    
        DetCalc_timeseries = pf.areasum_data(Detcalcct,mesh,mask)
        
        del Detcalc1, Detcalcct
        if secdet_path.is_file():
            del Detcalc2
        
        
        if output:
            self.NPPtotal = NPP_timeseries
            self.PhyTotal = NPPn_timeseries
            self.DiaTotal = NPPd_timeseries
            if cocco_path.is_file():
                self.CoccoTotal = NPPc_timeseries
            if phaeo_path.is_file():
                self.PhaeoTotal = NPPp_timeseries
            self.EPtotal = EP_timeseries
            self.SiEtotal = SiE_timeseries
            self.Calctotal = DetCalc_timeseries
        
        print('TIME-SERIES AVERAGES:')
        print('NPP mean = ',np.nanmean(NPP_timeseries))
        print('NPPd mean = ',np.nanmean(NPPd_timeseries))
        print('NPPn mean = ',np.nanmean(NPPn_timeseries))
        if cocco_path.is_file():
            print('NPPc mean = ',np.nanmean(NPPc_timeseries))
        if phaeo_path.is_file():
            print('NPPp mean = ',np.nanmean(NPPp_timeseries))
        print('EP mean = ',np.nanmean(EP_timeseries))
        print('SiE mean = ',np.nanmean(SiE_timeseries))
        print('CalcE mean = ',np.nanmean(DetCalc_timeseries))
        
        if plotting:
            # plotting total NPP -------------------------------------------------------------------------------        
            fig = plt.figure(figsize=(12,9), facecolor='w', edgecolor='k', tight_layout = True)
            
            plt.subplot(3, 3, 1)
            plt.plot(years,NPP_timeseries,'.-',color='C0',label='Total')
            plt.title('Total NPP')
            plt.ylabel(r'[Pg C yr$^{-1}$]')

            plt.subplot(3, 3, 2)
            plt.plot(years,NPPn_timeseries,'.-g',label='Sphy',color='C1')
            plt.title('small phytoplankton NPP')

            plt.subplot(3, 3, 3)
            plt.plot(years,NPPd_timeseries,'.-r',label='Dia',color='C2')
            plt.title('diatom NPP')
            
            if cocco_path.is_file():
                plt.subplot(3, 3, 4)
                plt.plot(years,NPPc_timeseries,'.-r',label='Cocco',color='C3')
                plt.title('coccolithophore NPP')
            if phaeo_path.is_file():
                plt.subplot(3, 3, 5)
                plt.plot(years,NPPp_timeseries,'.-r',label='Phaeo',color='C4')
                plt.title('phaeocystis NPP')

            # all NPP together
            plt.subplot(3, 3, 6)
            plt.plot(years,NPP_timeseries,'.-',label='Total',color='C0')
            plt.plot(years,NPPn_timeseries,'.-',label='Sphy',color='C1')
            plt.plot(years,NPPd_timeseries,'.-',label='Dia',color='C2')
            if cocco_path.is_file():
                plt.plot(years,NPPc_timeseries,'.-',label='Cocco',color='C3')
            if phaeo_path.is_file():
                plt.plot(years,NPPp_timeseries,'.-',label='Phaeo',color='C4')
            plt.title('NPP')
            plt.ylabel(r'[Pg C yr$^{-1}$]')
            plt.legend(loc='best')

            # Export production
            plt.subplot(3, 3, 7)
            plt.plot(years,EP_timeseries,'.-',color='C3')
            plt.title('EP at 100 m')
            plt.ylabel(r'[Pg C yr$^{-1}$]')

            # Si export
            plt.subplot(3, 3, 8)
            plt.plot(years,SiE_timeseries,'.-',color='C4')
            plt.title('Si export at 100 m')
            plt.ylabel(r'[Tmol Si yr$^{-1}$]')
            
            # CaCO3 export
            plt.subplot(3, 3, 9)
            plt.plot(years,DetCalc_timeseries,'.-',color='C5')
            plt.title('CaCO3 export at 100 m')
            plt.ylabel(r'[Pg C yr$^{-1}$]')
            


            if(savefig):
                plt.savefig(self.savepath+self.runname+'_'+'NPP_timeseries_'+maskstr+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'NPP_timeseries_'+maskstr+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', bbox_inches='tight')
            plt.show(block=False)

class plot_timeseries_dsi:   
    '''
    class Nutrienflux(resultpath,savepath,mesh,first_year,last_year,savefig=False,regional='Global',runname)
    
    Output: class din, dsi, and dfe with self.din, self.dsi and self.dfe containing nutrients concentrations.
    Nutrient concentrations are averaged within 3 water layers (0-50m, 50-300m, 300-100m). 
    This configuration can be adapted if needed.
    The nutrient concentrations are calculated and returned within required region:
    ("Arctic", "Southern", "Pacific", "Atlantic", "Indian", "Global")
    '''
    def __init__(self,resultpath,savepath,mesh,first_year,last_year,
                 savefig=False,regional='Global',plotting=True, output=False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.plotting = plotting

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # loading data and converting units
        DSi = pf.get_data(resultpath, "DSi", years, mesh, how=None, compute=False, 
                               runid=self.runname, silent=True)
        DSi = DSi.resample(time='YS').mean(dim='time').compute()
        
        if regional == 'Global':
            mask = pf.get_mask(mesh, "Global Ocean")
            
            dsi_0_50 = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300 = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000 = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            # plotting CO2 flux -------------------------------------------------------------------------------        
        
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,dsi_0_50,'.-',color='k')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300,'.-',color='k')
                axs[2].plot(years,dsi_300_1000,'.-',color='k')


                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                if output:
                    self.dsi_0_50 = dsi_0_50
                    self.dsi_50_300 = dsi_50_300
                    self.dsi_300_1000 = dsi_300_1000
                
        if regional == 'all':
            print('calculating time series...')
            mask = pf.get_mask(mesh, "Global Ocean")
            dsi_0_50_global = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_global = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_global = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Atlantic_Basin")
            dsi_0_50_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            dsi_0_50_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Arctic_Basin")           
            dsi_0_50_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            dsi_0_50_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Indian_Basin")
            dsi_0_50_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            print('calculation done.')
            
            del DSi
            
            if output:
                self.dsi_0_50_global = dsi_0_50_global
                self.dsi_50_300_global = dsi_50_300_global
                self.dsi_300_1000_global = dsi_300_1000_global

                self.dsi_0_50_atlantic = dsi_0_50_atlantic
                self.dsi_50_300_atlantic = dsi_50_300_atlantic
                self.dsi_300_1000_atlantic = dsi_300_1000_atlantic

                self.dsi_0_50_pacific = dsi_0_50_pacific
                self.dsi_50_300_pacific = dsi_50_300_pacific
                self.dsi_300_1000_pacific = dsi_300_1000_pacific

                self.dsi_0_50_arctic = dsi_0_50_arctic
                self.dsi_50_300_arctic = dsi_50_300_arctic
                self.dsi_300_1000_arctic = dsi_300_1000_arctic

                self.dsi_0_50_southern = dsi_0_50_southern
                self.dsi_50_300_southern = dsi_50_300_southern
                self.dsi_300_1000_southern = dsi_300_1000_southern
                
                self.dsi_0_50_indian = dsi_0_50_indian
                self.dsi_50_300_indian = dsi_50_300_indian
                self.dsi_300_1000_indian = dsi_300_1000_indian
            
            if plotting:
                print('plotting...')
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,dsi_0_50_global,'.-',color='k')
                axs[0].plot(years,dsi_0_50_pacific,'.-',color='C0')
                axs[0].plot(years,dsi_0_50_atlantic,'.-',color='C1')
                axs[0].plot(years,dsi_0_50_arctic,'.-',color='C2')
                axs[0].plot(years,dsi_0_50_southern,'.-',color='C3')
                axs[0].plot(years,dsi_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                
                axs[1].plot(years,dsi_50_300_global,'.-',color='k')
                axs[1].plot(years,dsi_50_300_pacific,'.-',color='C0')
                axs[1].plot(years,dsi_50_300_atlantic,'.-',color='C1')
                axs[1].plot(years,dsi_50_300_arctic,'.-',color='C2')
                axs[1].plot(years,dsi_50_300_southern,'.-',color='C3')
                axs[1].plot(years,dsi_50_300_indian,'.-',color='C4')
                axs[1].set_title('50-300m ')
                
                axs[2].plot(years,dsi_300_1000_global,'.-',color='k', label = 'Global')
                axs[2].plot(years,dsi_300_1000_pacific,'.-',color='C0', label = 'Pacific')
                axs[2].plot(years,dsi_300_1000_atlantic,'.-',color='C1', label = 'Atlantic')
                axs[2].plot(years,dsi_300_1000_arctic,'.-',color='C2', label = 'Arctic')
                axs[2].plot(years,dsi_300_1000_southern,'.-',color='C3', label = 'Southern')
                axs[2].plot(years,dsi_300_1000_indian,'.-',color='C4', label = 'Indian')
                axs[2].set_title('300-1000m ')
                
                axs[2].legend(bbox_to_anchor=(1.1, 0.5), loc='lower left', borderaxespad=0., fontsize=14)

                print('plotting done.')
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Arctic':
            mask = pf.get_mask(mesh, "Arctic_Basin")
            
            dsi_0_50_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_arctic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.dsi_0_50_arctic = dsi_0_50_arctic
                self.dsi_50_300_arctic = dsi_50_300_arctic
                self.dsi_300_1000_arctic = dsi_300_1000_arctic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dsi_0_50_arctic,'.-',color='C2')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300_arctic,'.-',color='C2')
                axs[2].plot(years,dsi_300_1000_arctic,'.-',color='C2')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            
        if regional == 'Southern':
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            
            dsi_0_50_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_southern = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.din_0_50_southern = din_0_50_southern
                self.dsi_0_50_southern = dsi_0_50_southern
                self.dfe_0_50_southern = dfe_0_50_southern
                self.din_50_300_southern = din_50_300_southern
                self.dsi_50_300_southern = dsi_50_300_southern
                self.dfe_50_300_southern = dfe_50_300_southern
                self.din_300_1000_southern = din_300_1000_southern
                self.dsi_300_1000_southern = dsi_300_1000_southern
                self.dfe_300_1000_southern = dfe_300_1000_southern
            
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dsi_0_50_southern,'.-',color='C3')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300_southern,'.-',color='C3')
                axs[2].plot(years,dsi_300_1000_southern,'.-',color='C3')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

        if regional == 'Pacific':
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            
            dsi_0_50_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_pacific = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.dsi_0_50_pacific = dsi_0_50_pacific
                self.dsi_50_300_pacific = dsi_50_300_pacific
                self.dsi_300_1000_pacific = dsi_300_1000_pacific
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dsi_0_50_pacific,'.-',color='C0')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300_pacific,'.-',color='C0')
                axs[2].plot(years,dsi_300_1000_pacific,'.-',color='C0')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Atlantic':
            mask = pf.get_mask(mesh, "Atlantic_Basin")            
            dsi_0_50_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_atlantic = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.dsi_0_50_atlantic = dsi_0_50_atlantic
                self.dsi_50_300_atlantic = dsi_50_300_atlantic
                self.dsi_300_1000_atlantic = dsi_300_1000_atlantic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dsi_0_50_atlantic,'.-',color='C1')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300_atlantic,'.-',color='C1')
                axs[2].plot(years,dsi_300_1000_atlantic,'.-',color='C1')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Indian':
            mask = pf.get_mask(mesh, "Indian_Basin")
            din_0_50_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            dsi_0_50_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [0, 50])
            dsi_50_300_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [50, 300])
            dsi_300_1000_indian = pf.volmean_data(DSi, mesh, mask = mask, uplow = [300, 1000])
            
            dfe_0_50_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.din_0_50_indian = din_0_50_indian
                self.dsi_0_50_indian = dsi_0_50_indian
                self.dfe_0_50_indian = dfe_0_50_indian
                self.din_50_300_indian = din_50_300_indian
                self.dsi_50_300_indian = dsi_50_300_indian
                self.dfe_50_300_indian = dfe_50_300_indian
                self.din_300_1000_indian = din_300_1000_indian
                self.dsi_300_1000_indian = dsi_300_1000_indian
                self.dfe_300_1000_indian = dfe_300_1000_indian
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dsi_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DSi [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dsi_50_300_indian,'.-',color='C4')
                axs[2].plot(years,dsi_300_1000_indian,'.-',color='C4')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DSi_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

class plot_timeseries_din:   
    '''
    class Nutrienflux(resultpath,savepath,meshpath,first_year,last_year,savefig=False,regional='Global',runname = 'fesom')
    
    Output: class din, din, and dfe with self.din, self.din and self.dfe containing nutrients concentrations.
    Nutrient concentrations are averaged within 3 water layers (0-50m, 50-300m, 300-100m). 
    This configuration can be adapted if needed.
    The nutrient concentrations are calculated and returned within required region:
    ("Arctic", "Southern", "Pacific", "Atlantic", "Indian", "Global")
    '''
    def __init__(self,resultpath,savepath,mesh,first_year,last_year,
                 savefig=False,regional='Global',plotting=True, output=False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.plotting = plotting

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # loading data and converting units
        DIN = pf.get_data(resultpath, "DIN", years, mesh, how=None, compute=False, 
                               runid=self.runname, silent=True)
        DIN = DIN.resample(time='YS').mean(dim='time').compute()
        
        if regional == 'Global':
            mask = pf.get_mask(mesh, "Global Ocean")
            
            din_0_50 = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300 = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000 = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            # plotting CO2 flux -------------------------------------------------------------------------------        
        
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,din_0_50,'.-',color='k')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300,'.-',color='k')
                axs[2].plot(years,din_300_1000,'.-',color='k')


                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                if output:
                    self.din_0_50 = din_0_50
                    self.din_50_300 = din_50_300
                    self.din_300_1000 = din_300_1000
                
        if regional == 'all':
            print('calculating time series...')
            mask = pf.get_mask(mesh, "Global Ocean")
            din_0_50_global = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_global = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_global = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Atlantic_Basin")
            din_0_50_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            din_0_50_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Arctic_Basin")           
            din_0_50_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            din_0_50_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Indian_Basin")
            din_0_50_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            print('calculation done.')
            
            del DIN
            
            if output:
                self.din_0_50_global = din_0_50_global
                self.din_50_300_global = din_50_300_global
                self.din_300_1000_global = din_300_1000_global

                self.din_0_50_atlantic = din_0_50_atlantic
                self.din_50_300_atlantic = din_50_300_atlantic
                self.din_300_1000_atlantic = din_300_1000_atlantic

                self.din_0_50_pacific = din_0_50_pacific
                self.din_50_300_pacific = din_50_300_pacific
                self.din_300_1000_pacific = din_300_1000_pacific

                self.din_0_50_arctic = din_0_50_arctic
                self.din_50_300_arctic = din_50_300_arctic
                self.din_300_1000_arctic = din_300_1000_arctic

                self.din_0_50_southern = din_0_50_southern
                self.din_50_300_southern = din_50_300_southern
                self.din_300_1000_southern = din_300_1000_southern
                
                self.din_0_50_indian = din_0_50_indian
                self.din_50_300_indian = din_50_300_indian
                self.din_300_1000_indian = din_300_1000_indian
            
            if plotting:
                print('plotting...')
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,din_0_50_global,'.-',color='k')
                axs[0].plot(years,din_0_50_pacific,'.-',color='C0')
                axs[0].plot(years,din_0_50_atlantic,'.-',color='C1')
                axs[0].plot(years,din_0_50_arctic,'.-',color='C2')
                axs[0].plot(years,din_0_50_southern,'.-',color='C3')
                axs[0].plot(years,din_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                
                axs[1].plot(years,din_50_300_global,'.-',color='k')
                axs[1].plot(years,din_50_300_pacific,'.-',color='C0')
                axs[1].plot(years,din_50_300_atlantic,'.-',color='C1')
                axs[1].plot(years,din_50_300_arctic,'.-',color='C2')
                axs[1].plot(years,din_50_300_southern,'.-',color='C3')
                axs[1].plot(years,din_50_300_indian,'.-',color='C4')
                axs[1].set_title('50-300m ')
                
                axs[2].plot(years,din_300_1000_global,'.-',color='k', label = 'Global')
                axs[2].plot(years,din_300_1000_pacific,'.-',color='C0', label = 'Pacific')
                axs[2].plot(years,din_300_1000_atlantic,'.-',color='C1', label = 'Atlantic')
                axs[2].plot(years,din_300_1000_arctic,'.-',color='C2', label = 'Arctic')
                axs[2].plot(years,din_300_1000_southern,'.-',color='C3', label = 'Southern')
                axs[2].plot(years,din_300_1000_indian,'.-',color='C4', label = 'Indian')
                axs[2].set_title('300-1000m ')
                
                axs[2].legend(bbox_to_anchor=(1.1, 0.5), loc='lower left', borderaxespad=0., fontsize=14)
                
                print('plotting done.')
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Arctic':
            mask = pf.get_mask(mesh, "Arctic_Basin")
            
            din_0_50_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_arctic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.din_0_50_arctic = din_0_50_arctic
                self.din_50_300_arctic = din_50_300_arctic
                self.din_300_1000_arctic = din_300_1000_arctic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,din_0_50_arctic,'.-',color='C2')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300_arctic,'.-',color='C2')
                axs[2].plot(years,din_300_1000_arctic,'.-',color='C2')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
        if regional == 'Southern':
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            
            din_0_50_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_southern = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.din_0_50_southern = din_0_50_southern
                self.din_0_50_southern = din_0_50_southern
                self.dfe_0_50_southern = dfe_0_50_southern
                self.din_50_300_southern = din_50_300_southern
                self.din_50_300_southern = din_50_300_southern
                self.dfe_50_300_southern = dfe_50_300_southern
                self.din_300_1000_southern = din_300_1000_southern
                self.din_300_1000_southern = din_300_1000_southern
                self.dfe_300_1000_southern = dfe_300_1000_southern
            
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,din_0_50_southern,'.-',color='C3')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300_southern,'.-',color='C3')
                axs[2].plot(years,din_300_1000_southern,'.-',color='c3')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                
        if regional == 'Pacific':
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            
            din_0_50_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_pacific = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.din_0_50_pacific = din_0_50_pacific
                self.din_50_300_pacific = din_50_300_pacific
                self.din_300_1000_pacific = din_300_1000_pacific
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,din_0_50_pacific,'.-',color='C0')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300_pacific,'.-',color='C0')
                axs[2].plot(years,din_300_1000_pacific,'.-',color='C0')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                
        if regional == 'Atlantic':
            mask = pf.get_mask(mesh, "Atlantic_Basin")            
            din_0_50_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_atlantic = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.din_0_50_atlantic = din_0_50_atlantic
                self.din_50_300_atlantic = din_50_300_atlantic
                self.din_300_1000_atlantic = din_300_1000_atlantic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,din_0_50_atlantic,'.-',color='C1')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300_atlantic,'.-',color='C1')
                axs[2].plot(years,din_300_1000_atlantic,'.-',color='C1')
            
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                
        if regional == 'Indian':
            mask = pf.get_mask(mesh, "Indian_Basin")
            din_0_50_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            din_0_50_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            dfe_0_50_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.din_0_50_indian = din_0_50_indian
                self.din_0_50_indian = din_0_50_indian
                self.dfe_0_50_indian = dfe_0_50_indian
                self.din_50_300_indian = din_50_300_indian
                self.din_50_300_indian = din_50_300_indian
                self.dfe_50_300_indian = dfe_50_300_indian
                self.din_300_1000_indian = din_300_1000_indian
                self.din_300_1000_indian = din_300_1000_indian
                self.dfe_300_1000_indian = dfe_300_1000_indian
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,din_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DIN [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,din_50_300_indian,'.-',color='C4')
                axs[2].plot(years,din_300_1000_indian,'.-',color='C4')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_DIN_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

class plot_timeseries_dfe:   
    '''
    class Nutrienflux(resultpath,savepath,meshpath,first_year,last_year,savefig=False,regional='Global',runname='fesom')
    
    Output: class din, dfe, and dfe with self.din, self.dfe and self.dfe containing nutrients concentrations.
    Nutrient concentrations are averaged within 3 water layers (0-50m, 50-300m, 300-100m). 
    This configuration can be adapted if needed.
    The nutrient concentrations are calculated and returned within required region:
    ("Arctic", "Southern", "Pacific", "Atlantic", "Indian", "Global")
    '''
    def __init__(self,resultpath,savepath,mesh,first_year,last_year,
                 savefig=False,regional='Global',plotting=True, output=False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.plotting = plotting

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # loading data and converting units
        DFe = pf.get_data(resultpath, "DFe", years, mesh, how=None, compute=False, 
                               runid=self.runname, silent=True)
        DFe = DFe.resample(time='YS').mean(dim='time').compute()
        
        if regional == 'Global':
            mask = pf.get_mask(mesh, "Global Ocean")
            
            dfe_0_50 = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300 = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000 = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            # plotting CO2 flux -------------------------------------------------------------------------------        
        
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,dfe_0_50,'.-',color='k')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300,'.-',color='k')
                axs[2].plot(years,dfe_300_1000,'.-',color='k')


                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')

                plt.show(block=False)
                
                if output:
                    self.dfe_0_50 = dfe_0_50
                    self.dfe_50_300 = dfe_50_300
                    self.dfe_300_1000 = dfe_300_1000
                
        if regional == 'all':
            print('calculating time series...')
            mask = pf.get_mask(mesh, "Global Ocean")
            dfe_0_50_global = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_global = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_global = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Atlantic_Basin")
            dfe_0_50_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            dfe_0_50_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Arctic_Basin")           
            dfe_0_50_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            dfe_0_50_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            mask = pf.get_mask(mesh, "Indian_Basin")
            dfe_0_50_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            print('calculation done.')
            
            del DFe
            
            if output:
                self.dfe_0_50_global = dfe_0_50_global
                self.dfe_50_300_global = dfe_50_300_global
                self.dfe_300_1000_global = dfe_300_1000_global

                self.dfe_0_50_atlantic = dfe_0_50_atlantic
                self.dfe_50_300_atlantic = dfe_50_300_atlantic
                self.dfe_300_1000_atlantic = dfe_300_1000_atlantic

                self.dfe_0_50_pacific = dfe_0_50_pacific
                self.dfe_50_300_pacific = dfe_50_300_pacific
                self.dfe_300_1000_pacific = dfe_300_1000_pacific

                self.dfe_0_50_arctic = dfe_0_50_arctic
                self.dfe_50_300_arctic = dfe_50_300_arctic
                self.dfe_300_1000_arctic = dfe_300_1000_arctic

                self.dfe_0_50_southern = dfe_0_50_southern
                self.dfe_50_300_southern = dfe_50_300_southern
                self.dfe_300_1000_southern = dfe_300_1000_southern
                
                self.dfe_0_50_indian = dfe_0_50_indian
                self.dfe_50_300_indian = dfe_50_300_indian
                self.dfe_300_1000_indian = dfe_300_1000_indian
            
            if plotting:
                print('plotting...')
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                
                axs[0].plot(years,dfe_0_50_global,'.-',color='k')
                axs[0].plot(years,dfe_0_50_pacific,'.-',color='C0')
                axs[0].plot(years,dfe_0_50_atlantic,'.-',color='C1')
                axs[0].plot(years,dfe_0_50_arctic,'.-',color='C2')
                axs[0].plot(years,dfe_0_50_southern,'.-',color='C3')
                axs[0].plot(years,dfe_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                
                axs[1].plot(years,dfe_50_300_global,'.-',color='k')
                axs[1].plot(years,dfe_50_300_pacific,'.-',color='C0')
                axs[1].plot(years,dfe_50_300_atlantic,'.-',color='C1')
                axs[1].plot(years,dfe_50_300_arctic,'.-',color='C2')
                axs[1].plot(years,dfe_50_300_southern,'.-',color='C3')
                axs[1].plot(years,dfe_50_300_indian,'.-',color='C4')
                axs[1].set_title('50-300m ')
                
                axs[2].plot(years,dfe_300_1000_global,'.-',color='k', label = 'Global')
                axs[2].plot(years,dfe_300_1000_pacific,'.-',color='C0', label = 'Pacific')
                axs[2].plot(years,dfe_300_1000_atlantic,'.-',color='C1', label = 'Atlantic')
                axs[2].plot(years,dfe_300_1000_arctic,'.-',color='C2', label = 'Arctic')
                axs[2].plot(years,dfe_300_1000_southern,'.-',color='C3', label = 'Southern')
                axs[2].plot(years,dfe_300_1000_indian,'.-',color='C4', label = 'Indian')
                axs[2].set_title('300-1000m ')
                
                axs[2].legend(bbox_to_anchor=(1.1, 0.5), loc='lower left', borderaxespad=0., fontsize=14)
                
                print('plotting done.')
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')

                plt.show(block=False)
            
        if regional == 'Arctic':
            mask = pf.get_mask(mesh, "Arctic_Basin")
            
            dfe_0_50_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_arctic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.dfe_0_50_arctic = dfe_0_50_arctic
                self.dfe_50_300_arctic = dfe_50_300_arctic
                self.dfe_300_1000_arctic = dfe_300_1000_arctic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dfe_0_50_arctic,'.-',color='C2')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300_arctic,'.-',color='C2')
                axs[2].plot(years,dfe_300_1000_arctic,'.-',color='C2')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Southern':
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            
            dfe_0_50_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_southern = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])

            
            if output:
                self.din_0_50_southern = din_0_50_southern
                self.dfe_0_50_southern = dfe_0_50_southern
                self.dfe_0_50_southern = dfe_0_50_southern
                self.din_50_300_southern = din_50_300_southern
                self.dfe_50_300_southern = dfe_50_300_southern
                self.dfe_50_300_southern = dfe_50_300_southern
                self.din_300_1000_southern = din_300_1000_southern
                self.dfe_300_1000_southern = dfe_300_1000_southern
                self.dfe_300_1000_southern = dfe_300_1000_southern
            
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dfe_0_50_southern,'.-',color='C3')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300_southern,'.-',color='C3')
                axs[2].plot(years,dfe_300_1000_southern,'.-',color='C3')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

        if regional == 'Pacific':
            
            mask = pf.get_mask(mesh, "Pacific_Basin")
            
            dfe_0_50_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_pacific = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.dfe_0_50_pacific = dfe_0_50_pacific
                self.dfe_50_300_pacific = dfe_50_300_pacific
                self.dfe_300_1000_pacific = dfe_300_1000_pacific
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dfe_0_50_pacific,'.-',color='C0')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300_pacific,'.-',color='C0')
                axs[2].plot(years,dfe_300_1000_pacific,'.-',color='C0')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Atlantic':
            mask = pf.get_mask(mesh, "Atlantic_Basin")            
            dfe_0_50_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_atlantic = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.dfe_0_50_atlantic = dfe_0_50_atlantic
                self.dfe_50_300_atlantic = dfe_50_300_atlantic
                self.dfe_300_1000_atlantic = dfe_300_1000_atlantic
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dfe_0_50_atlantic,'.-',color='C1')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300_atlantic,'.-',color='C1')
                axs[2].plot(years,dfe_300_1000_atlantic,'.-',color='C1')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Indian':
            mask = pf.get_mask(mesh, "Indian_Basin")
            din_0_50_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [0, 50])
            din_50_300_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [50, 300])
            din_300_1000_indian = pf.volmean_data(DIN, mesh, mask = mask, uplow = [300, 1000])
            
            dfe_0_50_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            dfe_0_50_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [0, 50])
            dfe_50_300_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [50, 300])
            dfe_300_1000_indian = pf.volmean_data(DFe, mesh, mask = mask, uplow = [300, 1000])
            
            if output:
                self.din_0_50_indian = din_0_50_indian
                self.dfe_0_50_indian = dfe_0_50_indian
                self.dfe_0_50_indian = dfe_0_50_indian
                self.din_50_300_indian = din_50_300_indian
                self.dfe_50_300_indian = dfe_50_300_indian
                self.dfe_50_300_indian = dfe_50_300_indian
                self.din_300_1000_indian = din_300_1000_indian
                self.dfe_300_1000_indian = dfe_300_1000_indian
                self.dfe_300_1000_indian = dfe_300_1000_indian
                
            if plotting:
                fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True)
                axs[0].plot(years,dfe_0_50_indian,'.-',color='C4')
                axs[0].set_ylabel('DFe [mmol m$^{-3}$]')
                axs[0].set_title('0-50m ')
                axs[1].set_title('50-300m ')
                axs[2].set_title('300-1000m ')
                axs[1].plot(years,dfe_50_300_indian,'.-',color='C4')
                axs[2].plot(years,dfe_300_1000_indian,'.-',color='C4')
                
                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

class plot_timeseries_co2f:   
    '''
    class CO2flux(resultpath,savepath,meshpath,first_year,last_year,savefig=False,regional='Global',runname)
    
    Out: class fco2 with self.fco2 containing CO2 flux into surface waters
    The nutrient concentrations are calculated and returned within required region.
    '''
    def __init__(self,resultpath,savepath,mesh,first_year,last_year,
                 savefig=False,regional='Global',plotting=True,output=False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.plotting = plotting
        self.output = output

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # loading data and converting units
        data = pf.get_data(resultpath, "CO2f", years, mesh, how=None, compute=False, 
                               runid=self.runname, silent=True)
        
        data2 = data.resample(time='YS').mean(dim='time')
        data2        = 12.01 * data2 * 365 /1e18
        # CO2f:description = "CO2-flux into the surface water" 
        # CO2f:units = [mmolC/m2/d] converting to [Pg C/year]
        
        if regional == 'Global':
            mask = pf.get_mask(mesh, "Global Ocean")
            flux = pf.areasum_data(data2, mesh, mask)
            
            if output:
                self.fco2_global = flux
            
            # plotting CO2 flux -------------------------------------------------------------------------------        
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux,'.-')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')
                plt.title('Global')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_TotalGlobal'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')

                plt.show(block=False)
                
                
                
        if regional == 'all':
            mask_global = pf.get_mask(mesh, "Global Ocean")
            flux_global = pf.areasum_data(data2, mesh, mask_global)
            
            mask_atlantic = pf.get_mask(mesh, "Atlantic_Basin")
            flux_atlantic = pf.areasum_data(data2, mesh, mask_atlantic)
            
            mask_pacific = pf.get_mask(mesh, "Pacific_Basin")
            flux_pacific = pf.areasum_data(data2, mesh, mask_pacific)
            
            mask_arctic = pf.get_mask(mesh, "Arctic_Basin")
            flux_arctic = pf.areasum_data(data2, mesh, mask_arctic)
            
            mask_southern = pf.get_mask(mesh, "Southern_Ocean_Basin")
            flux_southern = pf.areasum_data(data2, mesh, mask_southern)
            
            mask_indian = pf.get_mask(mesh, "Indian_Basin")
            flux_indian = pf.areasum_data(data2, mesh, mask_indian)
            
            if output:
                self.fco2_global = flux_global
                self.fco2_arctic = flux_arctic
                self.fco2_southern = flux_southern
                self.fco2_indian = flux_indian
                self.fco2_atlantic = flux_atlantic
                self.fco2_pacific = flux_pacific
            
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_global,'.-', color = 'k', label='Global')
                plt.plot(years,flux_pacific,'.-', color = 'C0', label='Pacific')
                plt.plot(years,flux_atlantic,'.-', color = 'C1', label='Atlantic')
                plt.plot(years,flux_arctic,'.-', color = 'C2', label='Arctic')
                plt.plot(years,flux_southern,'.-', color = 'C3', label='Southern')
                plt.plot(years,flux_indian,'.-', color = 'C4', label='Indian')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Arctic':
            mask = pf.get_mask(mesh, "Arctic_Basin")
            flux = pf.areasum_data(data2, mesh, mask)
            if output:
                self.fco2_arctic = flux
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_arctic,'.-', color = 'C2', label='Arctic')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            dpi = 300, bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Southern':
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")
            flux = pf.areasum_data(data2, mesh, mask)
            if output:
                self.fco2_southern = flux
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_southern,'.-', color = 'C3', label='Southern')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

        if regional == 'Pacific':
            mask = pf.get_mask(mesh, "Pacific_Basin")
            flux = pf.areasum_data(data2, mesh, mask)
            if output:
                self.fco2_pacific = flux
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_pacific,'.-', color = 'C0', label='Pacific')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
        if regional == 'Atlantic':
            mask = pf.get_mask(mesh, "Atlantic_Basin")
            flux = pf.areasum_data(data2, mesh, mask)
            if output:
                self.fco2_atlantic = flux
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_atlantic,'.-', color = 'C1', label='Atlantic')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')

                plt.show(block=False)
            
        if regional == 'Indian':
            mask = pf.get_mask(mesh, "Indian_Basin")
            flux = pf.areasum_data(data2, mesh, mask)
            if output:
                self.fco2_indian = flux
            if plotting:
                fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
                plt.plot(years,flux_indian,'.-', color = 'C4', label='Indian')
                plt.legend(loc='best')
                plt.title(self.runname+': CO$_2$ flux into surface water')
                plt.ylabel(r'[Pg C yr$^{-1}$]')

                if(self.savefig == True):
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_CO2flux_timeseries_'+regional+'_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')

                plt.show(block=False)

