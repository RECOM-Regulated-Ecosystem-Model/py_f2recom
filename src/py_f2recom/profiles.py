"""
Visualization tools for vertical profiles in REcoM model output.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyfesom2 as pf
import numpy as np
import math 
import skill_metrics as sm
from scipy.interpolate import griddata
from pathlib import Path
from .loading import *
from .datasets import *
from .core import *

class plot_profiles_ts:   
    '''
    class Nut_depth(runname,resultpath,savepath,meshpath,ncfileWOA,first_year,last_year,
                 savefig=False,regional=True)
                 
    if regional = True, profiles will plotted for each main basins + Global Ocean. 
    Otherwise, just the Global Ocean.
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncfileTemp=ncfileTemp,
                 ncfileSal=ncfileSal,
                 first_year=first_year,
                 last_year=last_year,
                 savefig=False, 
                 regional=True, 
                 maxdepth = 5,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncfileTemp = ncfileTemp
        self.ncfileSal = ncfileSal
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.maxdepth = maxdepth 
        
        years = np.arange(first_year, last_year+1,1)
        meshdiag = pf.get_meshdiag(mesh)
        runid      =  self.runname
        
        labelfesom = 'FESOM'
        labelwoa = 'WOA'
        unittemp = 'T [$^{\circ}$C]'
        unitsal = 'Salinity'
        
        # load data -------------------------------------------------------------------------------------
        TEMPfesom = pf.get_data(resultpath, "temp", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        SALfesom = pf.get_data(resultpath, "salt", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)
        
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncfileSal,'s_an', get_overview=False)
        sal_int = WOA_input.woa_int  
        
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncfileTemp,'t_an', get_overview=False)
        temp_int = WOA_input.woa_int  

        SALwoa = np.copy(sal_int)
        SALwoa[SALfesom == 0] = 0
        
        TEMPwoa = np.copy(temp_int)
        TEMPwoa[TEMPfesom == 0] = 0
        
        # Load and derive profiles

        nod_area = np.ma.masked_equal(meshdiag.nod_area.values, 0)
        mask = pf.get_mask(mesh, "Global Ocean")

        TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        TEMPfesom_weighted_Global = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        TEMPwoa_weighted_Global = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

        SALfesom_by_area = ((np.ma.masked_equal(SALfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        SALwoa_by_area = ((np.ma.masked_equal(SALwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        SALfesom_weighted_Global = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        SALwoa_weighted_Global = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        
        if regional:
            mask = pf.get_mask(mesh, "Atlantic_Basin")

            TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            TEMPfesom_weighted_Atlantic = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            TEMPwoa_weighted_Atlantic = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            SALfesom_by_area = ((np.ma.masked_equal(SALfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            SALwoa_by_area = ((np.ma.masked_equal(SALwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            SALfesom_weighted_Atlantic = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            SALwoa_weighted_Atlantic = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Pacific_Basin")

            TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            TEMPfesom_weighted_Pacific = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            TEMPwoa_weighted_Pacific = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            SALfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            SALwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            SALfesom_weighted_Pacific = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            SALwoa_weighted_Pacific = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Indian_Basin")

            TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            TEMPfesom_weighted_Indian = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            TEMPwoa_weighted_Indian = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            SALfesom_by_area = ((np.ma.masked_equal(SALfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            SALwoa_by_area = ((np.ma.masked_equal(SALwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            SALfesom_weighted_Indian = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            SALwoa_weighted_Indian = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Arctic_Basin")

            TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            TEMPfesom_weighted_Arctic = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            TEMPwoa_weighted_Arctic = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            SALfesom_by_area = ((np.ma.masked_equal(SALfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            SALwoa_by_area = ((np.ma.masked_equal(SALwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            SALfesom_weighted_Arctic = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            SALwoa_weighted_Arctic = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")

            TEMPfesom_by_area = ((np.ma.masked_equal(TEMPfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            TEMPwoa_by_area = ((np.ma.masked_equal(TEMPwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            TEMPfesom_weighted_Southern = TEMPfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            TEMPwoa_weighted_Southern = TEMPwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            SALfesom_by_area = ((np.ma.masked_equal(SALfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            SALwoa_by_area = ((np.ma.masked_equal(SALwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            SALfesom_weighted_Southern = SALfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            SALwoa_weighted_Southern = SALwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            # plotting
            
            fig, axs = plt.subplots(2,6, figsize=(14, 7), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0,0].plot(TEMPfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0,0].plot(TEMPwoa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0,0].set_ylabel('Depth [km]',fontsize=14)
            axs[0,0].set_xlabel(unittemp,fontsize=14)
            axs[0,0].set_title('Global Ocean',size=16, weight='bold')
            axs[0,0].tick_params(labelsize=14)
            axs[0,0].grid()
            axs[0,0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1,0].plot(SALfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[1,0].plot(SALwoa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[1,0].set_ylabel('Depth [km]',fontsize=14)
            axs[1,0].set_xlabel(unitsal,fontsize=14)
            axs[1,0].tick_params(labelsize=14)
            axs[1,0].grid()
            
            axs[0,1].plot(TEMPfesom_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C0', lw=3)
            axs[0,1].plot(TEMPwoa_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[0,1].set_xlabel(unittemp,fontsize=14)
            axs[0,1].set_title('Pacific Ocean',size=16, weight='bold')
            axs[0,1].tick_params(labelsize=14)
            axs[0,1].grid()
            
            axs[1,1].plot(SALfesom_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C0', lw=3)
            axs[1,1].plot(SALwoa_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[1,1].set_xlabel(unitsal,fontsize=14)   
            axs[1,1].tick_params(labelsize=14)
            axs[1,1].grid()
            
            axs[0,2].plot(TEMPfesom_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C1', lw=3)
            axs[0,2].plot(TEMPwoa_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[0,2].set_xlabel(unittemp,fontsize=14)
            axs[0,2].set_title('Atlantic Ocean',size=16, weight='bold')
            axs[0,2].tick_params(labelsize=14)
            axs[0,2].grid()
            
            axs[1,2].plot(SALfesom_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C1', lw=3)
            axs[1,2].plot(SALwoa_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[1,2].set_xlabel(unitsal,fontsize=14)
            axs[1,2].tick_params(labelsize=14)
            axs[1,2].grid()
            
            axs[0,3].plot(TEMPfesom_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C2', lw=3)
            axs[0,3].plot(TEMPwoa_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[0,3].set_xlabel(unittemp,fontsize=14)
            axs[0,3].set_title('Arctic Ocean',size=16, weight='bold')
            axs[0,3].tick_params(labelsize=14)
            axs[0,3].grid()
            
            axs[1,3].plot(SALfesom_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C2', lw=3)
            axs[1,3].plot(SALwoa_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[1,3].set_xlabel(unitsal,fontsize=14)
            axs[1,3].tick_params(labelsize=14)
            axs[1,3].grid()
            
            axs[0,4].plot(TEMPfesom_weighted_Southern, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C3', lw=3)
            axs[0,4].plot(TEMPwoa_weighted_Southern, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[0,4].set_xlabel(unittemp,fontsize=14)
            axs[0,4].set_title('Southern Ocean',size=16, weight='bold')
            axs[0,4].tick_params(labelsize=14)
            axs[0,4].grid()
            
            axs[1,4].plot(SALfesom_weighted_Southern, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C3', lw=3)
            axs[1,4].plot(SALwoa_weighted_Southern, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[1,4].set_xlabel(unitsal,fontsize=14)
            axs[1,4].tick_params(labelsize=14)
            axs[1,4].grid()
            
            axs[0,5].plot(TEMPfesom_weighted_Indian, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C4', lw=3)
            axs[0,5].plot(TEMPwoa_weighted_Indian, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[0,5].set_xlabel(unittemp,fontsize=14)
            axs[0,5].set_title('Indian Ocean',size=16, weight='bold')
            axs[0,5].tick_params(labelsize=14)
            axs[0,5].grid()
            
            axs[1,5].plot(SALfesom_weighted_Indian, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C4', lw=3)
            axs[1,5].plot(SALwoa_weighted_Indian, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[1,5].set_xlabel(unitsal,fontsize=14)
            axs[1,5].tick_params(labelsize=14)
            axs[1,5].grid()
            
            axs[0,0].set_ylim(-self.maxdepth,0)
            axs[1,0].set_ylim(-self.maxdepth,0)
            
        else:
            
            fig, axs = plt.subplots(1,2, figsize=(10, 5), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0].plot(TEMPfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0].plot(TEMPwoa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0].set_ylabel('Depth [km]',fontsize=14)
            axs[0].set_xlabel(unittemp,fontsize=14)
            axs[0].set_title('Global Ocean',size=16, weight='bold')
            axs[0].tick_params(labelsize=14)
            axs[0].grid()
            
            axs[1].plot(SALfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[1].plot(SALwoa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[1].set_ylabel('Depth [km]',fontsize=14)
            axs[1].set_xlabel(unitsal,fontsize=14)
            axs[1].tick_params(labelsize=14)
            axs[1].set_title('Global Ocean',size=16, weight='bold')
            axs[1].grid()
            axs[1].legend(bbox_to_anchor=(1.1, 0.5), loc='lower left', borderaxespad=0., fontsize=14)
            
            axs[0].set_ylim(-self.maxdepth,0)
            axs[1].set_ylim(-self.maxdepth,0)
        
        if(savefig):
                plt.savefig(self.savepath+self.runname+'_TS_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.png', dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_TS_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', bbox_inches='tight')
        plt.show(block=False)



class plot_profiles_nut:   
    '''
    class Nut_depth(resultpath,savepath,meshpath,ncfileDSi,ncfileDIN,ncfileDFe,first_year,last_year,
                 savefig=False,regional=True,runname)
                 
    c.f. Danilov et al. (2017):
    "in the vertical direction, the horizontal velocities and scalars are located at mid-levels" 
                 
    if regional = True, profiles will plotted for each main basins + Global Ocean. 
    Otherwise, just the Global Ocean.
    '''
    def __init__(self,resultpath,savepath,mesh,ncfileDSi,ncfileDIN,ncfileDFe,
                 first_year,last_year,savefig=False, regional=True, maxdepth = 6000, runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncfileDSi = ncfileDSi
        self.ncfileDIN = ncfileDIN
        self.ncfileDFe = ncfileDFe
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        self.maxdepth = maxdepth
        
        
        years = np.arange(first_year, last_year+1,1)
        meshdiag = pf.get_meshdiag(mesh)
        runid      =  self.runname
        unitsDIN = 'DIN [mmol m$^{-3}$]'
        unitsDSi = 'DSi [mmol m$^{-3}$]'
        unitsDFe = 'DFe [mmol m$^{-3}$]'
        
        
        # load data -------------------------------------------------------------------------------------
        DINfesom = pf.get_data(resultpath, "DIN", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        DSifesom = pf.get_data(resultpath, "DSi", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)
        
        DFefesom = pf.get_data(resultpath, "DFe", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        DINwoa_input = load_woa_data(runid,resultpath,mesh,ncfileDIN,'n_an', get_overview=False)
        DSiwoa_input = load_woa_datadata(runid,resultpath,mesh,ncfileDSi,'i_an', get_overview=False)
        DFepisces_input = load_pisces_data(runid,resultpath,mesh,ncfileDFe,'Fe', get_overview=False)
        
        DFepisces = DFepisces_input.pisces_int
        DFepisces[DFepisces>1000]= 0
        DFepisces[DFefesom == 0] = 0
        
        DINwoa = DINwoa_input.woa_int
        DINwoa[DINfesom == 0] = 0

        DSiwoa = DSiwoa_input.woa_int
        DSiwoa[DSifesom == 0] = 0
        
        # Load and derive profiles

        nod_area = np.ma.masked_equal(meshdiag.nod_area.values, 0)
        mask = pf.get_mask(mesh, "Global Ocean")

        DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        DINfesom_weighted_Global = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        DINwoa_weighted_Global = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

        DSifesom_by_area = ((np.ma.masked_equal(DSifesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        DSiwoa_by_area = ((np.ma.masked_equal(DSiwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        DSifesom_weighted_Global = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        DSiwoa_weighted_Global = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        
        DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        DFefesom_weighted_Global = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        DFepisces_weighted_Global = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        
        if regional:
            mask = pf.get_mask(mesh, "Atlantic_Basin")

            DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DINfesom_weighted_Atlantic = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DINwoa_weighted_Atlantic = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            DSifesom_by_area = ((np.ma.masked_equal(DSifesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DSiwoa_by_area = ((np.ma.masked_equal(DSiwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DSifesom_weighted_Atlantic = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DSiwoa_weighted_Atlantic = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DFefesom_weighted_Atlantic = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DFepisces_weighted_Atlantic = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Pacific_Basin")

            DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DINfesom_weighted_Pacific = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DINwoa_weighted_Pacific = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            DSifesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DSiwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DSifesom_weighted_Pacific = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DSiwoa_weighted_Pacific = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DFefesom_weighted_Pacific = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DFepisces_weighted_Pacific = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Indian_Basin")

            DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DINfesom_weighted_Indian = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DINwoa_weighted_Indian = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            DSifesom_by_area = ((np.ma.masked_equal(DSifesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DSiwoa_by_area = ((np.ma.masked_equal(DSiwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DSifesom_weighted_Indian = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DSiwoa_weighted_Indian = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DFefesom_weighted_Indian = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DFepisces_weighted_Indian = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Arctic_Basin")

            DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DINfesom_weighted_Arctic = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DINwoa_weighted_Arctic = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            DSifesom_by_area = ((np.ma.masked_equal(DSifesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DSiwoa_by_area = ((np.ma.masked_equal(DSiwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DSifesom_weighted_Arctic = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DSiwoa_weighted_Arctic = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DFefesom_weighted_Arctic = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DFepisces_weighted_Arctic = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")

            DINfesom_by_area = ((np.ma.masked_equal(DINfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DINwoa_by_area = ((np.ma.masked_equal(DINwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DINfesom_weighted_Southern = DINfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DINwoa_weighted_Southern = DINwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            DSifesom_by_area = ((np.ma.masked_equal(DSifesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DSiwoa_by_area = ((np.ma.masked_equal(DSiwoa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DSifesom_weighted_Southern = DSifesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DSiwoa_weighted_Southern = DSiwoa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            DFefesom_by_area = ((np.ma.masked_equal(DFefesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DFepisces_by_area = ((np.ma.masked_equal(DFepisces[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DFefesom_weighted_Southern = DFefesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DFepisces_weighted_Southern = DFepisces_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            # plotting
            
            fig, axs = plt.subplots(3,6, figsize=(14, 11), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0,0].plot(DINfesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[0,0].plot(DINwoa_weighted_Global, mesh.zlev[:-1],label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0,0].set_ylabel('Depth [km]',fontsize=14)
            axs[0,0].set_xlabel(unitsDIN,fontsize=14)
            axs[0,0].set_title('Global Ocean',size=16, weight='bold')
            axs[0,0].tick_params(labelsize=14)
            axs[0,0].grid()
            axs[0,0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1,0].plot(DSifesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[1,0].plot(DSiwoa_weighted_Global, mesh.zlev[:-1],label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[1,0].set_ylabel('Depth [km]',fontsize=14)
            axs[1,0].set_xlabel(unitsDSi,fontsize=14)
            axs[1,0].tick_params(labelsize=14)
            axs[1,0].grid()
            
            axs[2,0].plot(DFefesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[2,0].plot(DFepisces_weighted_Global, mesh.zlev[:-1],label = 'PISCES', color = 'k', lw=3, linestyle = '--')
            axs[2,0].set_ylabel('Depth [km]',fontsize=14)
            axs[2,0].set_xlabel(unitsDFe,fontsize=14)
            axs[2,0].tick_params(labelsize=14)
            axs[2,0].grid()
            axs[2,0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[0,1].plot(DINfesom_weighted_Pacific, mesh.zlev[:-1],label = 'FESOM', color = 'C0', lw=3)
            axs[0,1].plot(DINwoa_weighted_Pacific, mesh.zlev[:-1],label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[0,1].set_xlabel(unitsDIN,fontsize=14)
            axs[0,1].set_title('Pacific Ocean',size=16, weight='bold')
            axs[0,1].tick_params(labelsize=14)
            axs[0,1].grid()
            
            axs[1,1].plot(DSifesom_weighted_Pacific, mesh.zlev[:-1],label = 'FESOM', color = 'C0', lw=3)
            axs[1,1].plot(DSiwoa_weighted_Pacific, mesh.zlev[:-1],label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[1,1].set_xlabel(unitsDSi,fontsize=14)   
            axs[1,1].tick_params(labelsize=14)
            axs[1,1].grid()
            
            axs[2,1].plot(DFefesom_weighted_Pacific, mesh.zlev[:-1],label = 'FESOM', color = 'C0', lw=3)
            axs[2,1].plot(DFepisces_weighted_Pacific, mesh.zlev[:-1],label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[2,1].set_xlabel(unitsDFe,fontsize=14)   
            axs[2,1].tick_params(labelsize=14)
            axs[2,1].grid()
            
            axs[0,2].plot(DINfesom_weighted_Atlantic, mesh.zlev[:-1],label = 'FESOM', color = 'C1', lw=3)
            axs[0,2].plot(DINwoa_weighted_Atlantic, mesh.zlev[:-1],label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[0,2].set_xlabel(unitsDIN,fontsize=14)
            axs[0,2].set_title('Atlantic Ocean',size=16, weight='bold')
            axs[0,2].tick_params(labelsize=14)
            axs[0,2].grid()
            
            axs[1,2].plot(DSifesom_weighted_Atlantic, mesh.zlev[:-1],label = 'FESOM', color = 'C1', lw=3)
            axs[1,2].plot(DSiwoa_weighted_Atlantic, mesh.zlev[:-1],label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[1,2].set_xlabel(unitsDSi,fontsize=14)
            axs[1,2].tick_params(labelsize=14)
            axs[1,2].grid()
            
            axs[2,2].plot(DFefesom_weighted_Atlantic, mesh.zlev[:-1],label = 'FESOM', color = 'C1', lw=3)
            axs[2,2].plot(DFepisces_weighted_Atlantic, mesh.zlev[:-1],label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[2,2].set_xlabel(unitsDFe,fontsize=14)
            axs[2,2].tick_params(labelsize=14)
            axs[2,2].grid()
            
            axs[0,3].plot(DINfesom_weighted_Arctic, mesh.zlev[:-1],label = 'FESOM', color = 'C2', lw=3)
            axs[0,3].plot(DINwoa_weighted_Arctic, mesh.zlev[:-1],label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[0,3].set_xlabel(unitsDIN,fontsize=14)
            axs[0,3].set_title('Arctic Ocean',size=16, weight='bold')
            axs[0,3].tick_params(labelsize=14)
            axs[0,3].grid()
            
            axs[1,3].plot(DSifesom_weighted_Arctic, mesh.zlev[:-1],label = 'FESOM', color = 'C2', lw=3)
            axs[1,3].plot(DSiwoa_weighted_Arctic, mesh.zlev[:-1],label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[1,3].set_xlabel(unitsDSi,fontsize=14)
            axs[1,3].tick_params(labelsize=14)
            axs[1,3].grid()
            
            axs[2,3].plot(DFefesom_weighted_Arctic, mesh.zlev[:-1],label = 'FESOM', color = 'C2', lw=3)
            axs[2,3].plot(DFepisces_weighted_Arctic, mesh.zlev[:-1],label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[2,3].set_xlabel(unitsDFe,fontsize=14)
            axs[2,3].tick_params(labelsize=14)
            axs[2,3].grid()
            
            axs[0,4].plot(DINfesom_weighted_Southern, mesh.zlev[:-1],label = 'FESOM', color = 'C3', lw=3)
            axs[0,4].plot(DINwoa_weighted_Southern, mesh.zlev[:-1],label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[0,4].set_xlabel(unitsDIN,fontsize=14)
            axs[0,4].set_title('Southern Ocean',size=16, weight='bold')
            axs[0,4].tick_params(labelsize=14)
            axs[0,4].grid()
            
            axs[1,4].plot(DSifesom_weighted_Southern, mesh.zlev[:-1],label = 'FESOM', color = 'C3', lw=3)
            axs[1,4].plot(DSiwoa_weighted_Southern, mesh.zlev[:-1],label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[1,4].set_xlabel(unitsDSi,fontsize=14)
            axs[1,4].tick_params(labelsize=14)
            axs[1,4].grid()
            
            axs[2,4].plot(DFefesom_weighted_Southern, mesh.zlev[:-1],label = 'FESOM', color = 'C3', lw=3)
            axs[2,4].plot(DFepisces_weighted_Southern, mesh.zlev[:-1],label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[2,4].set_xlabel(unitsDFe,fontsize=14)
            axs[2,4].tick_params(labelsize=14)
            axs[2,4].grid()
            
            axs[0,5].plot(DINfesom_weighted_Indian, mesh.zlev[:-1],label = 'FESOM', color = 'C4', lw=3)
            axs[0,5].plot(DINwoa_weighted_Indian, mesh.zlev[:-1],label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[0,5].set_xlabel(unitsDIN,fontsize=14)
            axs[0,5].set_title('Indian Ocean',size=16, weight='bold')
            axs[0,5].tick_params(labelsize=14)
            axs[0,5].grid()
            
            axs[1,5].plot(DSifesom_weighted_Indian, mesh.zlev[:-1],label = 'FESOM', color = 'C4', lw=3)
            axs[1,5].plot(DSiwoa_weighted_Indian, mesh.zlev[:-1],label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[1,5].set_xlabel(unitsDSi,fontsize=14)
            axs[1,5].tick_params(labelsize=14)
            axs[1,5].grid()
            
            
            axs[2,5].plot(DFefesom_weighted_Indian, mesh.zlev[:-1],label = 'FESOM', color = 'C4', lw=3)
            axs[2,5].plot(DFepisces_weighted_Indian, mesh.zlev[:-1],label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[2,5].set_xlabel(unitsDFe,fontsize=14)
            axs[2,5].tick_params(labelsize=14)
            axs[2,5].grid()
            
            axs[0,0].set_ylim(-self.maxdepth,0)
            axs[1,0].set_ylim(-self.maxdepth,0)
            axs[2,0].set_ylim(-self.maxdepth,0)
            
        else:
            
            fig, axs = plt.subplots(1,3, figsize=(10, 5), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0].plot(DINfesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[0].plot(DINwoa_weighted_Global, mesh.zlev[:-1],label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0].set_ylabel('Depth [km]',fontsize=14)
            axs[0].set_xlabel(unitsDIN,fontsize=14)
            axs[1].set_title('Global Ocean',size=16, weight='bold')
            axs[0].tick_params(labelsize=14)
            axs[0].grid()
            axs[0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1].plot(DSifesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[1].plot(DSiwoa_weighted_Global, mesh.zlev[:-1],label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[1].set_ylabel('Depth [km]',fontsize=14)
            axs[1].set_xlabel(unitsDSi,fontsize=14)
            axs[1].tick_params(labelsize=14)
            axs[1].grid()
            
            axs[2].plot(DFefesom_weighted_Global, mesh.zlev[:-1],label = 'FESOM', color = 'k', lw=3)
            axs[2].plot(DFepisces_weighted_Global, mesh.zlev[:-1],label = 'PISCES', color = 'k', lw=3, linestyle = '--')
            axs[2].set_ylabel('Depth [km]',fontsize=14)
            axs[2].set_xlabel(unitsDFe,fontsize=14)
            axs[2].tick_params(labelsize=14)
            axs[2].grid()
            axs[2].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[0].set_ylim(-self.maxdepth,0)
            axs[1].set_ylim(-self.maxdepth,0)
            axs[2].set_ylim(-self.maxdepth,0)
            
        if(savefig):
                plt.savefig(self.savepath+self.runname+'_'+'Nutrient_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.png', dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'Nutrient_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', bbox_inches='tight')
        plt.show(block=False)


class plot_profiles_do2:   
    '''
    class Nut_depth(resultpath,savepath,meshpath,ncfileDSi,ncfileDO2,ncfileDFe,first_year,last_year,
                 savefig=False,regional=True,runname)
                 
    c.f. Danilov et al. (2017):
    "in the vertical direction, the horizontal velocities and scalars are located at mid-levels" 
                 
    if regional = True, profiles will plotted for each main basins + Global Ocean. 
    Otherwise, just the Global Ocean.
    '''
    def __init__(self,resultpath,savepath,mesh,ncfileDO2,
                 first_year,last_year,savefig=False, regional=True,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncfileDO2 = ncfileDO2
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        
        years = np.arange(first_year, last_year+1,1)
        meshdiag = pf.get_meshdiag(mesh)
        runid      =  self.runname
        
        unitsDO2 = 'DO2 [mmol m$^{-3}$]'
        
        
        # load data -------------------------------------------------------------------------------------
        DO2fesom = pf.get_data(resultpath, "O2", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        DO2woa_input = load_woa_data(runid,resultpath,mesh,ncfileDO2,'oxygen_mmol', get_overview=False)

        DO2woa = DO2woa_input.woa_int
        DO2woa[DO2fesom == 0] = 0
        
        # Load and derive profiles

        nod_area = np.ma.masked_equal(meshdiag.nod_area.values, 0)
        mask = pf.get_mask(mesh, "Global Ocean")

        DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        DO2fesom_weighted_Global = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        DO2woa_weighted_Global = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        
        if regional:
            mask = pf.get_mask(mesh, "Atlantic_Basin")

            DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DO2fesom_weighted_Atlantic = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DO2woa_weighted_Atlantic = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Pacific_Basin")

            DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DO2fesom_weighted_Pacific = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DO2woa_weighted_Pacific = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Indian_Basin")

            DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DO2fesom_weighted_Indian = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DO2woa_weighted_Indian = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Arctic_Basin")

            DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DO2fesom_weighted_Arctic = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DO2woa_weighted_Arctic = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")

            DO2fesom_by_area = ((np.ma.masked_equal(DO2fesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DO2woa_by_area = ((np.ma.masked_equal(DO2woa[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DO2fesom_weighted_Southern = DO2fesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DO2woa_weighted_Southern = DO2woa_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            # plotting
            
            fig, axs = plt.subplots(1,6, figsize=(14, 4), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0].plot(DO2fesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0].plot(DO2woa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0].set_ylabel('Depth [km]',fontsize=14)
            axs[0].set_xlabel(unitsDO2,fontsize=14)
            axs[0].set_title('Global Ocean',size=16, weight='bold')
            axs[0].tick_params(labelsize=14)
            axs[0].grid()
            axs[0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1].plot(DO2fesom_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C0', lw=3)
            axs[1].plot(DO2woa_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C0', lw=3, linestyle = '--')
            axs[1].set_xlabel(unitsDO2,fontsize=14)
            axs[1].set_title('Pacific Ocean',size=16, weight='bold')
            axs[1].tick_params(labelsize=14)
            axs[1].grid()
            
            axs[2].plot(DO2fesom_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C1', lw=3)
            axs[2].plot(DO2woa_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C1', lw=3, linestyle = '--')
            axs[2].set_xlabel(unitsDO2,fontsize=14)
            axs[2].set_title('Atlantic Ocean',size=16, weight='bold')
            axs[2].tick_params(labelsize=14)
            axs[2].grid()
            
            axs[3].plot(DO2fesom_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C2', lw=3)
            axs[3].plot(DO2woa_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C2', lw=3, linestyle = '--')
            axs[3].set_xlabel(unitsDO2,fontsize=14)
            axs[3].set_title('Arctic Ocean',size=16, weight='bold')
            axs[3].tick_params(labelsize=14)
            axs[3].grid()
            
            axs[4].plot(DO2fesom_weighted_Southern, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C3', lw=3)
            axs[4].plot(DO2woa_weighted_Southern, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C3', lw=3, linestyle = '--')
            axs[4].set_xlabel(unitsDO2,fontsize=14)
            axs[4].set_title('Southern Ocean',size=16, weight='bold')
            axs[4].tick_params(labelsize=14)
            axs[4].grid()
            
            axs[5].plot(DO2fesom_weighted_Indian, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C4', lw=3)
            axs[5].plot(DO2woa_weighted_Indian, mesh.zlev[:-1]/1000,label = 'WOA', color = 'C4', lw=3, linestyle = '--')
            axs[5].set_xlabel(unitsDO2,fontsize=14)
            axs[5].set_title('Indian Ocean',size=16, weight='bold')
            axs[5].tick_params(labelsize=14)
            axs[5].grid()

        else:
            
            fig, axs = plt.subplots(1,1, figsize=(10, 5), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0].plot(DO2fesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0].plot(DO2woa_weighted_Global, mesh.zlev[:-1]/1000,label = 'WOA', color = 'k', lw=3, linestyle = '--')
            axs[0].set_ylabel('Depth [km]',fontsize=14)
            axs[0].set_xlabel(unitsDO2,fontsize=14)
            axs[1].set_title('Global Ocean',size=16, weight='bold')
            axs[0].tick_params(labelsize=14)
            axs[0].grid()
            axs[0].legend(loc='best', borderaxespad=0., fontsize=14)
            
        if(self.savefig == True):
                plt.savefig(self.savepath+self.runname+'_DO2_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                            dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_DO2_profiles_'+str(self.fyear)+'to'+str(self.lyear)+'.pdf', 
                            bbox_inches='tight')
        plt.show(block=False)


class plot_profiles_carbs:   
    '''
    class Nut_depth(resultpath,savepath,meshpath,ncfileAlk,ncfileDIC,ncfileDFe,first_year,last_year,
                 savefig=False,regional=True,runname='fesom')
                 
    if regional = True, profiles will plotted for each main basins + Global Ocean. 
    Otherwise, just the Global Ocean.
    '''
    def __init__(self,resultpath,savepath,mesh,ncfileAlk,ncfileDIC,
                 first_year,last_year,savefig=False, regional=True,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncfileAlk = ncfileAlk
        self.ncfileDIC = ncfileDIC
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.regional = regional
        
        years = np.arange(first_year, last_year+1,1)

        meshdiag = pf.get_meshdiag(mesh)
        runid      =  self.runname
        
        unitsDIC = 'DIC [mmol m$^{-3}$]'
        unitsAlk = 'Alk [mmol m$^{-3}$]'
        
        
        # load data -------------------------------------------------------------------------------------
        DICfesom = pf.get_data(resultpath, "DIC", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        Alkfesom = pf.get_data(resultpath, "Alk", years, mesh,
                               how="mean", compute=True, runid=runid, silent=True)

        DICglodap_input = load_glodap_data(runid,resultpath,mesh,ncfileDIC,'TCO2_mmol', get_overview=False)
        Alkglodap_input = load_glodap_data(runid,resultpath,mesh,ncfileAlk,'TAlk_mmol', get_overview=False)
        
        DICglodap = DICglodap_input.glodap_int
        DICglodap[DICfesom == 0] = 0

        Alkglodap = Alkglodap_input.glodap_int
        Alkglodap[Alkfesom == 0] = 0
        
        # Load and derive profiles

        nod_area = np.ma.masked_equal(meshdiag.nod_area.values, 0)
        mask = pf.get_mask(mesh, "Global Ocean")

        DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        DICfesom_weighted_Global = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        DICglodap_weighted_Global = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

        Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
        Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

        Alkfesom_weighted_Global = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        Alkglodap_weighted_Global = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
        
        if regional:
            mask = pf.get_mask(mesh, "Atlantic_Basin")

            DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DICfesom_weighted_Atlantic = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DICglodap_weighted_Atlantic = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            Alkfesom_weighted_Atlantic = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            Alkglodap_weighted_Atlantic = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Pacific_Basin")

            DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DICfesom_weighted_Pacific = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DICglodap_weighted_Pacific = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            Alkfesom_weighted_Pacific = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            Alkglodap_weighted_Pacific = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            mask = pf.get_mask(mesh, "Indian_Basin")

            DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DICfesom_weighted_Indian = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DICglodap_weighted_Indian = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            Alkfesom_weighted_Indian = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            Alkglodap_weighted_Indian = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Arctic_Basin")

            DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DICfesom_weighted_Arctic = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DICglodap_weighted_Arctic = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            Alkfesom_weighted_Arctic = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            Alkglodap_weighted_Arctic = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            mask = pf.get_mask(mesh, "Southern_Ocean_Basin")

            DICfesom_by_area = ((np.ma.masked_equal(DICfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            DICglodap_by_area = ((np.ma.masked_equal(DICglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            DICfesom_weighted_Southern = DICfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            DICglodap_weighted_Southern = DICglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)

            Alkfesom_by_area = ((np.ma.masked_equal(Alkfesom[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))
            Alkglodap_by_area = ((np.ma.masked_equal(Alkglodap[mask,:],0) * nod_area[:-1,:].T[mask]).mean(axis=0))

            Alkfesom_weighted_Southern = Alkfesom_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            Alkglodap_weighted_Southern = Alkglodap_by_area/nod_area[:-1,:].T[mask].mean(axis=0)
            
            # plotting
            
            fig, axs = plt.subplots(2,6, figsize=(14, 8), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0,0].plot(DICfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0,0].plot(DICglodap_weighted_Global, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'k', lw=3, linestyle = '--')
            axs[0,0].set_ylabel('Depth [km]',fontsize=14)
            axs[0,0].set_xlabel(unitsDIC,fontsize=14)
            axs[0,0].set_title('Global Ocean',size=16, weight='bold')
            axs[0,0].tick_params(labelsize=14)
            axs[0,0].grid()
            axs[0,0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1,0].plot(Alkfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[1,0].plot(Alkglodap_weighted_Global, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'k', lw=3, linestyle = '--')
            axs[1,0].set_ylabel('Depth [km]',fontsize=14)
            axs[1,0].set_xlabel(unitsAlk,fontsize=14)
            axs[1,0].tick_params(labelsize=14)
            axs[1,0].grid()
            
            
            axs[0,1].plot(DICfesom_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C0', lw=3)
            axs[0,1].plot(DICglodap_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C0', lw=3, linestyle = '--')
            axs[0,1].set_xlabel(unitsDIC,fontsize=14)
            axs[0,1].set_title('Pacific Ocean',size=16, weight='bold')
            axs[0,1].tick_params(labelsize=14)
            axs[0,1].grid()
            
            axs[1,1].plot(Alkfesom_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C0', lw=3)
            axs[1,1].plot(Alkglodap_weighted_Pacific, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C0', lw=3, linestyle = '--')
            axs[1,1].set_xlabel(unitsAlk,fontsize=14)   
            axs[1,1].tick_params(labelsize=14)
            axs[1,1].grid()
            
            axs[0,2].plot(DICfesom_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C1', lw=3)
            axs[0,2].plot(DICglodap_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C1', lw=3, linestyle = '--')
            axs[0,2].set_xlabel(unitsDIC,fontsize=14)
            axs[0,2].set_title('Atlantic Ocean',size=16, weight='bold')
            axs[0,2].tick_params(labelsize=14)
            axs[0,2].grid()
            
            axs[1,2].plot(Alkfesom_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C1', lw=3)
            axs[1,2].plot(Alkglodap_weighted_Atlantic, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C1', lw=3, linestyle = '--')
            axs[1,2].set_xlabel(unitsAlk,fontsize=14)
            axs[1,2].tick_params(labelsize=14)
            axs[1,2].grid()
            
            axs[0,3].plot(DICfesom_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C2', lw=3)
            axs[0,3].plot(DICglodap_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C2', lw=3, linestyle = '--')
            axs[0,3].set_xlabel(unitsDIC,fontsize=14)
            axs[0,3].set_title('Arctic Ocean',size=16, weight='bold')
            axs[0,3].tick_params(labelsize=14)
            axs[0,3].grid()
            
            axs[1,3].plot(Alkfesom_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C2', lw=3)
            axs[1,3].plot(Alkglodap_weighted_Arctic, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C2', lw=3, linestyle = '--')
            axs[1,3].set_xlabel(unitsAlk,fontsize=14)
            axs[1,3].tick_params(labelsize=14)
            axs[1,3].grid()
            
            axs[0,4].plot(DICfesom_weighted_Southern, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C3', lw=3)
            axs[0,4].plot(DICglodap_weighted_Southern, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C3', lw=3, linestyle = '--')
            axs[0,4].set_xlabel(unitsDIC,fontsize=14)
            axs[0,4].set_title('Southern Ocean',size=16, weight='bold')
            axs[0,4].tick_params(labelsize=14)
            axs[0,4].grid()
            
            axs[1,4].plot(Alkfesom_weighted_Southern, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C3', lw=3)
            axs[1,4].plot(Alkglodap_weighted_Southern, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C3', lw=3, linestyle = '--')
            axs[1,4].set_xlabel(unitsAlk,fontsize=14)
            axs[1,4].tick_params(labelsize=14)
            axs[1,4].grid()
            
            axs[0,5].plot(DICfesom_weighted_Indian, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C4', lw=3)
            axs[0,5].plot(DICglodap_weighted_Indian, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C4', lw=3, linestyle = '--')
            axs[0,5].set_xlabel(unitsDIC,fontsize=14)
            axs[0,5].set_title('Indian Ocean',size=16, weight='bold')
            axs[0,5].tick_params(labelsize=14)
            axs[0,5].grid()
            
            axs[1,5].plot(Alkfesom_weighted_Indian, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'C4', lw=3)
            axs[1,5].plot(Alkglodap_weighted_Indian, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'C4', lw=3, linestyle = '--')
            axs[1,5].set_xlabel(unitsAlk,fontsize=14)
            axs[1,5].tick_params(labelsize=14)
            axs[1,5].grid()

        else:
            
            fig, axs = plt.subplots(1,2, figsize=(7, 5), facecolor='w', edgecolor='k', constrained_layout=True, sharey=True)

            axs[0].plot(DICfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[0].plot(DICglodap_weighted_Global, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'k', lw=3, linestyle = '--')
            axs[0].set_ylabel('Depth [km]',fontsize=14)
            axs[0].set_xlabel(unitsDIC,fontsize=14)
            axs[0].set_title('Global Ocean',size=16, weight='bold')
            axs[0].tick_params(labelsize=14)
            axs[0].grid()
            axs[0].legend(loc='best', borderaxespad=0., fontsize=14)
            
            axs[1].plot(Alkfesom_weighted_Global, mesh.zlev[:-1]/1000,label = 'FESOM', color = 'k', lw=3)
            axs[1].plot(Alkglodap_weighted_Global, mesh.zlev[:-1]/1000,label = 'GLODAP', color = 'k', lw=3, linestyle = '--')
            axs[1].set_ylabel('Depth [km]',fontsize=14)
            axs[1].set_xlabel(unitsAlk,fontsize=14)
            axs[1].set_title('Global Ocean',size=16, weight='bold')
            axs[1].tick_params(labelsize=14)
            axs[1].grid()
            
        # fig export  -------------------------------------------------------------------------------------
        if savefig:                
            plt.savefig(savepath+runid+'_'+'Carbs_profiles'+'_'+str(first_year)+'to'+str(last_year)+'.png', 
                        dpi = 300, bbox_inches='tight')
            plt.savefig(savepath+runid+'_'+'Carbs_profiles'+'_'+str(first_year)+'to'+str(last_year)+'.pdf', 
                        bbox_inches='tight')