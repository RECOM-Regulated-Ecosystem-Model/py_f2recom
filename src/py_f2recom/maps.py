"""
Visualization tools for maps in REcoM model output.
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
import scipy.io as spio
from pathlib import Path
import xarray as xr
from .loading import *
from .datasets import *
from .core import *

class plot_maps_woa_sal:
    '''
    class SALcomp
             
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    -Projection 'rob' (Robinson) results in strange values, please use only 'pc' Plate Carree!
    -Use of 'use_temp_mask' loads additional FESOM 'temp' out put to use as a mask for the 'salt' output.
    '''
    def __init__(self,
                 mesh,
                 resultpath = resultpath,
                 savepath = savepath,
                 ncpath = ncfileSal,
                 first_year = first_year,
                 last_year = last_year,
                 WOAvar='s_an',
                 mapproj = 'rob',
                 savefig=False,
                 cmap = 'viridis',
                 get_overview = False,
                 use_temp_mask = False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='both',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.WOAvar = WOAvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.get_overview = get_overview
        self.use_temp_mask = use_temp_mask
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.Taylor = Taylor

        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]

        self.mapproj = pf.get_proj(self.mapproj)  
        years = np.arange(first_year,last_year+1,1)

        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'Salinity' 

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "salt", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=self.verbose)
            
        # load WOA data  -------------------------------------------------------------------------------------
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.WOAvar, get_overview=self.get_overview)
        woa_int = WOA_input.woa_int    
        
        labelwoa = 'WOA'
        unitwoa = 'Salinity' 

        # apply sea mask to WOA as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        woa_int_ma = np.copy(woa_int)
        woa_int_ma[fesom == 0] = 0

        # plot WOA and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for d in depth_array:
                if d < np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(d,np.max(PHC.layer_depths)))
    
                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))

                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(23,37,.5)
                    f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box)

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box)

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-5,5.25,0.25)
                    f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                titles='FESOM - WOA' + ' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box)

                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                        size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'E', transform=m3.transAxes,
                        size=30, weight='bold')
                    
                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_'+str(plot_depth)+'m.png', 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_'+str(plot_depth)+'m.pdf', 
                                bbox_inches='tight')
                    plt.show(block=False)
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(sal_int_ma)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(SALfesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')       

                    title = 'Normalized Taylor Diagram for Salinity'
                    plot_taylor_comp(sal_int_ma,SALfesom, mask=True, title=title, depth_array=depth_array, mesh=mesh,verbose = self.verbose)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):                
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.png', dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.pdf', bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(23,37,.5)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box)
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,cmap_extension=self.cmap_extension,
                            titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box)
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-5,5.25,0.25)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            titles='FESOM - WOA' + ' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box)
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.pdf'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                plt.show(block=False)
            
            if Taylor:    
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(woa_int_ma_mean)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for Salinity (mean over depth, max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(woa_int_ma_mean,fesom_mean,mask=True,title=title,verbose = self.verbose)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', bbox_inches='tight')
                plt.show(block=False)
                
            if output:
                self.sal_fesom = fesom_mean
                self.sal_phc   = woa_int_ma_mean


class plot_maps_woa_temp:
    '''
    class TEMPcomp
    
    Compare temperature data of FESOM-REcoM to PHC3 Atlas.
    
    -Specify depth_limit to define upper/lower boundaries of the layer.
    -Use layerwise = True to compare a set of depth layers.
    '''
    def __init__(self,
                 mesh,
                 resultpath = resultpath,
                 savepath = savepath,
                 ncpath = ncfileTemp,
                 first_year = first_year,
                 last_year = last_year,
                 WOAvar='t_an',
                 mapproj='rob',
                 cmap = 'inferno',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 get_overview = False,
                 runname='fesom'):
        
        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.WOAvar = WOAvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.get_overview = get_overview
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
        
        self.mapproj = pf.get_proj(self.mapproj)         
        years = np.arange(first_year,last_year+1,1)


        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'T [$^{\circ}$C]'       

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "temp", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=self.verbose)

        # load WOA data  -------------------------------------------------------------------------------------
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.WOAvar, get_overview=self.get_overview)
        woa_int = WOA_input.woa_int    
        
        labelwoa = 'WOA'
        unitwoa = 'T [$^{\circ}$C]' 

        # apply sea mask to WOA as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        woa_int_ma = np.copy(woa_int)
        woa_int_ma[fesom == 0] = 0

        # plot WOA and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))

                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(-2,25,1)
                    f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                        levels = levels,
                        units=unitwoa, 
                        mapproj=self.mapproj, # robinson projection takes more time!
                        cmap = self.cmap,
                        cmap_extension=self.cmap_extension,
                        titles=labelfesom+'\n ({0}-{1} m m)'.format(uplow[0],uplow[1]),
                        box=box,
                        )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-5,5.25,0.25)
                    f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - WOA'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )
                    
                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                        size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'E', transform=m3.transAxes,
                        size=30, weight='bold')
                    
                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)

                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.pdf'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                    
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(woa_int_ma)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(fesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Normalized Taylor Diagram for Temperature'
                    plot_taylor_comp(temp_int_ma,TEMPfesom,mask=True,title=title, depth_array=depth_array, mesh=mesh,verbose = self.verbose)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.pdf'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))
            
        
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(-2,25,1)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-5,5.25,0.25)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - WOA'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                            bbox_inches='tight')
                plt.show(block=False)
            
            if self.Taylor: 
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(woa_int_ma_mean)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for T \n(mean over depth,  max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(woa_int_ma_mean,fesom_mean,mask=True,title=title, verbose = self.verbose)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'TEMP_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
        
            if output:
                    self.temp_fesom = fesom_mean
                    self.temp_phc   = woa_int_ma_mean

class plot_maps_phc_sal:
    '''
    class SALcomp
             
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    -Projection 'rob' (Robinson) results in strange values, please use only 'pc' Plate Carree!
    -Use of 'use_temp_mask' loads additional FESOM 'temp' out put to use as a mask for the 'salt' output.
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilePHC3,
                 first_year=first_year,
                 last_year=last_year,
                 PHCvar='salt',
                 mapproj = 'rob',
                 savefig=False,
                 cmap = 'viridis',
                 get_overview = False,
                 use_temp_mask = False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='both',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.PHCvar = PHCvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.get_overview = get_overview
        self.use_temp_mask = use_temp_mask
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.Taylor = Taylor

        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)   
        years = np.arange(first_year,last_year+1,1)

        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'Salinity' 

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "salt", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=self.verbose)

        if(use_temp_mask == True):
            if(self.verbose):
                print('use_temp_mask = True\nLoad temperature data for masking...')
            TEMPfesom = pf.get_data(resultpath, "temp", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=self.verbose)
            fesom[TEMPfesom == 0] = 0 # apply mask

            
        # load PHC data  -------------------------------------------------------------------------------------
        PHC_input = load_phc_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.PHCvar, get_overview=self.get_overview)
        sal_int = PHC_input.phc_int    
        
        labelphc = 'PHC'
        unitphc = 'Salinity' 

        # apply sea mask to PHC as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        sal_int_ma = np.copy(sal_int)
        sal_int_ma[fesom == 0] = 0

        # plot PHC and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(PHC_input.layer_depths)))

                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(sal_int_ma, mesh, uplow = uplow)

                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(sal_int_ma_mean),np.nanstd(sal_int_ma_mean),np.nanmin(sal_int_ma_mean),np.nanmax(sal_int_ma_mean)))


                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(23,37,.5)
                    f1 = pf.subplot(mesh, fig, m1, [fesom[:,i]],
                                levels = levels,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,cmap_extension=self.cmap_extension,
                                titles=labelfesom +'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [sal_int_ma[:,i]], 
                                levels = levels,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,cmap_extension=self.cmap_extension,
                                titles=labelphc +'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-5,5.25,0.25)
                    f3 = pf.subplot(mesh, fig, m3, [fesom[:,i]-sal_int_ma[:,i]], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                titles='FESOM - PHC\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    fig.subplots_adjust(bottom=0.16)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    plt.show(block=False)
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    
                          

            if Taylor:
                # statistics  -------------------------------------------------------------------------------------
                # preparation of datasets
                if np.isnan(np.min(sal_int_ma)): print('WARNING: The interpolated PHC field contains NaNs at depth')
                if np.isnan(np.min(fesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')       

                title = 'Normalized Taylor Diagram for Salinity'
                plot_taylor_comp(sal_int_ma,SALfesom, mask=True, title=title, depth_array=depth_array, mesh=mesh,verbose = True)
                
                plt.show(block=False)
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1])),
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1])),
                
                
            if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(PHC_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            sal_int_ma_mean = pf.layermean_data(sal_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPHC mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(sal_int_ma_mean),np.nanstd(sal_int_ma_mean),np.nanmin(sal_int_ma_mean),np.nanmax(sal_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                            """
                            AB
                            CC
                            """,
                            gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                            subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(23,37,.5)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,cmap_extension=self.cmap_extension,
                            titles=labelfesom +'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [sal_int_ma_mean], 
                            levels = levels,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,cmap_extension=self.cmap_extension,
                            titles=labelphc +'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-5,5.25,0.25)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-sal_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            titles='FESOM - PHC'+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                                dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                                bbox_inches='tight')
                plt.show(block=False)
                     
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(sal_int_ma_mean)): print('WARNING: The interpolated PHC field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for Salinity \n(mean over depth, max = {0}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(sal_int_ma_mean,fesom_mean,mask=True,title=title,verbose = True)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Sal_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', bbox_inches='tight')
                plt.show(block=False)
                
        if output:
            self.sal_fesom = fesom_mean
            self.sal_phc   = sal_int_ma_mean

class plot_maps_phc_temp:
    '''
    class TEMPcomp
    
    Compare temperature data of FESOM-REcoM to PHC.
    
    -Specify depth_limit to use only upper layers with this depth maximum.
    -Use layerwise = True to compare a set of depth.
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilePHC3,
                 first_year=first_year,
                 last_year=last_year,
                 PHCvar='temp',
                 mapproj='rob',
                 cmap = 'inferno',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=True,
                 get_overview = False,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):
        

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.PHCvar = PHCvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.get_overview = get_overview
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
        
        self.mapproj = pf.get_proj(self.mapproj)     
        years = np.arange(first_year,last_year+1,1)

        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'T [$^{\circ}$C]'       

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "temp", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=self.verbose)

        # load PHC data  -------------------------------------------------------------------------------------
        PHC_input = load_phc_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.PHCvar, get_overview=self.get_overview)
        temp_int = PHC_input.phc_int    
        
        labelphc = 'PHC'
        unitphc = 'T [$^{\circ}$C]' 

        # apply sea mask to PHC as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        temp_int_ma = np.copy(temp_int)
        temp_int_ma[fesom == 0] = 0

        # plot PHC and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(PHC_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(PHC_input.layer_depths)))

                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                temp_int_ma_mean = pf.layermean_data(temp_int_ma, mesh, uplow = uplow)

                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPHC mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(temp_int_ma_mean),np.nanstd(temp_int_ma_mean),np.nanmin(temp_int_ma_mean),np.nanmax(temp_int_ma_mean)))

                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(-2,25,1)
                    f1 = pf.subplot(mesh, fig, m1, [fesom[:,i]],
                                levels = levels,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m m)'.format(uplow[0],uplow[1]),
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [temp_int_ma[:,i]], 
                                levels = levels,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelphc+'\n ({0}-{1} m m)'.format(uplow[0],uplow[1]),
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-5,5.25,0.25)
                    f3 = pf.subplot(mesh, fig, m3, [fesom[:,i]-temp_int_ma[:,i]], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitphc, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - PHC'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                               )

                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False) 
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------
                # preparation of datasets
                if np.isnan(np.min(temp_int_ma)): print('WARNING: The interpolated PHC field contains NaNs at depth')
                if np.isnan(np.min(fesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Normalized Taylor Diagram for Temperature'
                plt_taylor_comp(temp_int_ma,fesom,mask=True,title=title, depth_array=depth_array, mesh=mesh,verbose = True)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):          
                    plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC_Taylor'+'_'+str(years[0])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC_Taylor'+'_'+str(years[0])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                plt.show(block=False)  
                
            if output:
                    print('Only return non-layerwise output')
                
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(PHC_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            temp_int_ma_mean = pf.layermean_data(temp_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(temp_int_ma_mean),np.nanstd(temp_int_ma_mean),np.nanmin(temp_int_ma_mean),np.nanmax(temp_int_ma_mean)))
            
        
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                            """
                            AB
                            CC
                            """,
                            gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                            subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(-2,25,1)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [temp_int_ma_mean], 
                            levels = levels,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelphc+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-5,5.25,0.25)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-temp_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitphc, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - PHC'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                           )
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),
                            bbox_inches='tight')
                plt.show(block=False) 
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------            
                    # preparation of datasets
                    if np.isnan(np.min(temp_int_ma_mean)): print('WARNING: The interpolated PHC field contains NaNs at depth')
                    if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Taylor Diagram for T \n(mean over depth,  max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                    plot_taylor_norm(temp_int_ma_mean,fesom_mean,mask=True,title=title, verbose = True)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                                bbox_inches='tight')
                    plt.show(block=False)
                    
            if output:
                self.temp_fesom = fesom_mean
                self.temp_phc   = temp_int_ma_mean
        
class plot_maps_phc_temp_regulargrid:
    '''
    class PHC3tempcomp
    
    Compare PHC Temperature and Salinity to FESOM
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 meshpath=meshpath,
                 ncfile=ncfilePHC3,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='pc',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,depth_array=[],
                 depth_limit=100,
                 cmap_extension='max',
                 verbose=False,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.meshpath = meshpath
        self.ncfile = ncfile
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.depth_limit = depth_limit
        self.verbose = verbose
        self.cmap_extension = cmap_extension

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)

        # check variables
        #NCfesom = self.resultpath + '/DFe.'+self.runname+'.'+str(self.fyear)+'.nc'
        #!ncdump -h $NCfesom

        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        unitfesom = 'T [$^{\circ}$C]'
        
        labelphc3 = 'PHC3'
        unitphc3 = 'T [$^{\circ}$C]'

        # load FESOM data -------------------------------------------------------------------------------------
        Tempfesom = pf.get_data(resultpath, "temp", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.meshpath+'/'+self.runname+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runname)             
        w = pf.climatology(ncfile)
        
        box=[-180, 180, -89, 90]
        left, right, down, up = box
            
        levels = np.arange(-2,25,1)
        levels_diff = np.arange(-5,5.25,.25)

        # plot PHC3 and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)

            for d in depth_array:
                if d < np.nanmin(w.z):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(d,np.max(PHC.layer_depths)))
    
                # get mesh index closest to desired depth
                ii = pf.ind_for_depth(d,mesh)
                iz, xx, yy, zz_temp = pf.fesom2clim(Tempfesom[:,ii], d, mesh, w, verbose=False)
                
                tfesom = zz_temp
                tphc3 = np.squeeze(w.T[iz,:,:]) 
            
                fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(15,15),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))

             
                axes[0].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im0 = axes[0].contourf(xx, yy, tphc3, levels = levels,cmap=cmap, extend=cmap_extension, zlev=0,
                    transform=ccrs.PlateCarree());
                axes[0].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes[0].set_title(labelphc3, size = 20)

                axes[1].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im1 = axes[1].contourf(xx, yy, tfesom, levels = levels,cmap=cmap, extend=cmap_extension, zlev=0,
                    transform=ccrs.PlateCarree());
                axes[1].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes[1].set_title(labelfesom, size = 20)

                cb = fig.colorbar(
                                im1, orientation="horizontal", ax=axes.ravel().tolist(), pad=0.01, shrink=0.9
                            )
                cb.ax.tick_params(labelsize=15)
                cb.set_label(unitfesom, size=20)


                fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True,figsize=(15,10),
                                         subplot_kw=dict(projection=ccrs.PlateCarree()))
                axes.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im = axes.contourf(xx, yy, tphc3 - tfesom, levels = levels_diff,cmap='RdBu_r', extend='both', zlev=0,
                    transform=ccrs.PlateCarree());
                axes.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes.set_title('PHC3 - FESOM', size = 20)

                cb = fig.colorbar(
                                im, orientation="horizontal", ax=axes, pad=0.01, shrink=0.9
                            )
                cb.ax.tick_params(labelsize=15)
                cb.set_label(unitfesom, size=20)
       
            if(self.savefig==True): print('\n***\n***Too many figures to export...\n***')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            ii = pf.ind_for_depth(depth_limit,mesh)
            iz, xx, yy, zz_temp = pf.fesom2clim(np.nanmean(Tempfesom[:,:ii],axis=1), depth_limit, mesh, w, verbose=False)
            
            # mean over depth                       
            tfesom = zz_temp
            tphc3 = np.mean(w.T[:iz,:,:], axis = 0) 
            
            fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(15,15),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))

#            fig = plt.figure(constrained_layout=True,figsize=(15,15))
#             axd = fig.subplot_mosaic(
#                                 """
#                                 AB
#                                 CC
#                                 CC
#                                 """
#                             ,subplot_kw=dict(projection=ccrs.PlateCarree()))
             
            axes[0].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im0 = axes[0].contourf(xx, yy, tphc3, levels = levels,cmap=cmap, extend=cmap_extension, zlev=0,
                transform=ccrs.PlateCarree());
            axes[0].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes[0].set_title(labelphc3, size = 20)
            
            axes[1].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im1 = axes[1].contourf(xx, yy, tfesom, levels = levels,cmap=cmap, extend=cmap_extension, zlev=0,
                transform=ccrs.PlateCarree());
            axes[1].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes[1].set_title(labelfesom, size = 20)
        
            cb = fig.colorbar(
                            im1, orientation="horizontal", ax=axes.ravel().tolist(), pad=0.01, shrink=0.9
                        )
            cb.ax.tick_params(labelsize=15)
            cb.set_label(unitfesom, size=20)
            
            
            fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True,figsize=(15,10),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))
            axes.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im = axes.contourf(xx, yy, tphc3 - tfesom, levels = levels_diff,cmap='RdBu_r', extend='both', zlev=0,
                transform=ccrs.PlateCarree());
            axes.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes.set_title('PHC3 - FESOM', size = 20)
            
            cb = fig.colorbar(
                            im, orientation="horizontal", ax=axes, pad=0.01, shrink=0.9
                        )
            cb.ax.tick_params(labelsize=15)
            cb.set_label(unitfesom, size=20)
            
            plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC3'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight') 


class plot_maps_phc_sal_regulargrid:
    '''
    class PHC3tempcomp
    
    Compare PHC Temperature and Salinity to FESOM
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncfile=ncfilePHC3,
                 first_year=first_year,
                 last_year=last_year,
                 meshpath=meshpath,
                 mapproj='pc',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,depth_array=[],
                 depth_limit=100,
                 verbose=False,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.meshpath = meshpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.depth_limit = depth_limit
        self.verbose = verbose

        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)

        # check variables
        #NCfesom = self.resultpath + '/DFe.'+self.runname+'.'+str(self.fyear)+'.nc'
        #!ncdump -h $NCfesom

        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        unitfesom = ''
        
        labelphc3 = 'PHC3'
        unitphc3 = ''

        # load FESOM data -------------------------------------------------------------------------------------
        Tempfesom = pf.get_data(resultpath, "salt", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.meshpath+'/'+self.runname+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runname)             
        w = pf.climatology(ncfile)
        
        levels = np.arange(23,37,.5)
        levels_diff = np.arange(-3,3.2,0.2)
        
        box=[-180, 180, -89, 90]
        left, right, down, up = box
        
        # plot PHC3 and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)

            for d in depth_array:
                if d < np.min(w.z):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(d,np.max(PHC.layer_depths)))
    
                # get mesh index closest to desired depth
                ii = pf.ind_for_depth(d,mesh)
                iz, xx, yy, zz_temp = pf.fesom2clim(Tempfesom[:,ii], d, mesh, w, verbose=False)
                
                tfesom = zz_temp
                tphc3 = np.squeeze(w.S[iz,:,:]) 
            
                fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(15,15),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))

             
                axes[0].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im0 = axes[0].contourf(xx, yy, tphc3, levels = levels,cmap=cmap, extend='both', zlev=0,
                    transform=ccrs.PlateCarree());
                axes[0].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes[0].set_title(labelphc3, size = 20)

                axes[1].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im1 = axes[1].contourf(xx, yy, tfesom, levels = levels,cmap=cmap, extend='both', zlev=0,
                    transform=ccrs.PlateCarree());
                axes[1].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes[1].set_title(labelfesom, size = 20)

                cb = fig.colorbar(
                                im1, orientation="horizontal", ax=axes.ravel().tolist(), pad=0.01, shrink=0.9
                            )
                cb.ax.tick_params(labelsize=15)
                cb.set_label(unitfesom, size=20)


                fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True,figsize=(15,10),
                                         subplot_kw=dict(projection=ccrs.PlateCarree()))
                axes.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
                im = axes.contourf(xx, yy, tphc3 - tfesom, levels = levels_diff,cmap='RdBu_r', extend='both', zlev=0,
                    transform=ccrs.PlateCarree());
                axes.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
                axes.set_title('PHC3 - FESOM', size = 20)

                cb = fig.colorbar(
                                im, orientation="horizontal", ax=axes, pad=0.01, shrink=0.9
                            )
                cb.ax.tick_params(labelsize=15)
                cb.set_label(unitfesom, size=20)

       
            if(self.savefig==True): print('\n***\n***Too many figures to export...\n***')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            ii = pf.ind_for_depth(depth_limit,mesh)
            iz, xx, yy, zz_temp = pf.fesom2clim(np.nanmean(Tempfesom[:,:ii],axis=1), depth_limit, mesh, w, verbose=False)
            
            # mean over depth                       
            tfesom = zz_temp
            tphc3 = np.mean(w.S[:iz,:,:], axis = 0) 
            
            fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(15,15),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))

             
            axes[0].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im0 = axes[0].contourf(xx, yy, tphc3, levels = levels,cmap=cmap, extend='both', zlev=0,
                transform=ccrs.PlateCarree());
            axes[0].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes[0].set_title(labelphc3, size = 20)
            
            axes[1].set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im1 = axes[1].contourf(xx, yy, tfesom, levels = levels,cmap=cmap, extend='both', zlev=0,
                transform=ccrs.PlateCarree());
            axes[1].add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes[1].set_title(labelfesom, size = 20)
        
            cb = fig.colorbar(
                            im1, orientation="horizontal", ax=axes.ravel().tolist(), pad=0.01, shrink=0.9
                        )
            cb.ax.tick_params(labelsize=15)
            cb.set_label(unitfesom, size=20)
            
            
            fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True,figsize=(15,10),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))
            axes.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
            im = axes.contourf(xx, yy, tphc3 - tfesom, levels = levels_diff,cmap='RdBu_r', extend='both', zlev=0,
                transform=ccrs.PlateCarree());
            axes.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='lightgray')
            axes.set_title('PHC3 - FESOM', size = 20)
            
            cb = fig.colorbar(
                            im, orientation="horizontal", ax=axes, pad=0.01, shrink=0.9
                        )
            cb.ax.tick_params(labelsize=15)
            cb.set_label(unitfesom, size=20)
            
            plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'Temp_PHC3'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight') 

            
class plot_maps_pco2:
    '''
    class pCO2SOCATcomp(resultpath,savepath,meshpath,ncpath,first_year,last_year,
                mapproj='pc',savefig=False,layerwise=False, runname='fesom')
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilepCO2,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='pc',
                 SOCATvar='TAlk',
                 cmap='viridis',
                 savefig=False,
                 cmap_extension='both',
                 verbose=False,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.cmap = cmap
        self.SOCATvar = SOCATvar
        self.cmap_extension = cmap_extension
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        self.verbose = verbose
        
        # derive SOCAT mapping projection -------------------------------------------------------------------------------------
        if((self.mapproj != 'pc') & (self.mapproj != 'rob')):
            print('Projection for plotting SOCAT data is not supported! Choose "pc" or "rob".\nprojection set to "pc"')
            self.mapproj == 'rob'
        
        box=[-180, 180, -90, 90]
        self.mapproj = pf.get_proj(self.mapproj)
 
        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        

        # load FESOM data -------------------------------------------------------------------------------------
        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        unitfesom = 'pCO$_2$ [$\mu$atm]' # equals to mumol/L
        
        # interpolate Glodap data -------------------------------------------------------------------------------------
        SOCAT_input = load_socat_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.SOCATvar, 
                                self.fyear, self.lyear, get_overview=False)
        socat_int = SOCAT_input.socat_int
        print(np.shape)
        time_soccat = SOCAT_input.time
            
        int = np.intersect1d(years, np.arange(1970,2020))
        
        if np.size(int) != 0:
            previous_year=-1
            previous_month=-1
            FESOM = np.empty((len(mesh.x2),len(time_soccat)))
            SOCAT = np.empty((len(mesh.x2),len(time_soccat)))
            DIFF = np.empty((len(mesh.x2),len(time_soccat)))
            for i in range(len(time_soccat)):
                current_time = time_soccat[i]
                current_year = current_time.year
                current_month = current_time.month
                
                if current_year != previous_year:
                    pCO2fesom = pf.get_data(resultpath, "pCO2s", current_year, mesh, 
                                   how=None, compute=True, runid=self.runname, silent=True)

                if current_month != previous_month:
                    fesom_monthly = pCO2fesom[current_month-1,:]

                # apply sea mask to Glodap as in FESOM ----------------------------------------------------------------------------------
                # assumption: there is no ocean where value in FESOM == 0
                socat_int_ma = np.copy(socat_int[:,i])
                socat_int_ma[fesom_monthly == 0] = 0

                diff = fesom_monthly - socat_int_ma

                DIFF[:,i]=diff
                SOCAT[:,i]=socat_int_ma
                FESOM[:,i]=fesom_monthly

                previous_year = current_year
                previous_month = current_month

            DIFF_mean = np.nanmean(DIFF, axis= 1)
            SOCAT_mean = np.nanmean(SOCAT, axis= 1)
            FESOM_mean = np.nanmean(FESOM, axis= 1)
        elif np.size(int) == 0:
            SOCAT_mean = SOCAT_input.socat_int_mean
            FESOM_mean = pf.get_data(resultpath, "pCO2s", years, mesh, 
                                   how='mean', compute=True, runid=self.runname, silent=True)
            DIFF_mean = FESOM_mean - SOCAT_mean
        
        # ==============================================================================
        # plot FESOM and SOCAT   
        if(self.verbose):
            print('\nPlotting pCO2\nFESOM min = {0}, max = {1}'.format(
                        np.nanmin(FESOM_mean),np.nanmax(FESOM_mean)))
            print('SOCAT min = {0}, max = {1}'.format(
                    np.nanmin(SOCAT_mean),np.nanmax(SOCAT_mean)))
        
        if ((self.fyear < 1970) and (self.lyear > 2020)):
            labelsocat = 'SOCAT (1970-2017)'
            label_diff = 'FESOM - SOCAT (1970-2020)'
        elif ((self.fyear < 1970) and (self.lyear <= 2020) and (self.lyear >= 1970)):
            labelsocat = 'SOCAT (1970-{0})'.format(last_year)
            label_diff = 'FESOM - SOCAT (1970-{0})'.format(last_year)
        elif ((self.fyear >= 1970) and (self.lyear > 2020)):
            labelsocat = 'SOCAT ({0}-2020)'.format(first_year)
            label_diff = 'FESOM - SOCAT ({0}-2020)'.format(first_year)
        elif ((self.lyear < 1970)):
            labelsocat = 'SOCAT (1970-2020)'
            label_diff = 'FESOM - SOCAT (no overlap)'
        elif ((self.fyear > 2020)):
            labelsocat = 'SOCAT (1970-2020)'
            label_diff = 'FESOM - SOCAT (no overlap)'
        else:
            labelsocat = 'SOCAT ({0}-{1})'.format(self.fyear,self.lyear)
            label_diff = 'FESOM - SOCAT ({0}-{1})'.format(self.fyear,self.lyear)
            
        unitsocat = 'pCO$_2$ [$\mu$atm]' 
            
        if plotting:
            fig = plt.figure(figsize=(15,12), constrained_layout=False)
            axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
            m1 = axes['A']
            levels = np.arange(140,520,20)
            f1 = pf.subplot(mesh, fig, m1, [FESOM_mean],
                            levels = levels,
                            units=unitsocat, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension='both',
                            titles=labelfesom,
                            box=box, interp = 'linear',
                           )

            print(np.shape(SOCAT_mean))
            m2 = axes['B']
            f2 = pf.subplot(mesh, fig, m2, SOCAT_mean, 
                            levels = levels,
                            units=unitsocat, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension='both',
                            titles=labelsocat,
                            box=box, interp = 'linear',
                           )
                    
            cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
            cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
            cbar1.set_label(unitfesom, fontsize=18)
            cbar1.ax.tick_params(labelsize=18)
        
            m3 = axes['C']

            levels_diff = np.arange(-120,130,10)
            f3 = pf.subplot(mesh, fig, m3, [DIFF_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitsocat, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles=label_diff,
                            box=box, interp = 'linear',
                           )
            
            m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
            m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                        size=30, weight='bold')
            m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                        size=30, weight='bold')
            
            fig.subplots_adjust(bottom=0.02)
            cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
            cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
            cbar2.set_label(unitfesom, fontsize=18)
            cbar2.ax.tick_params(labelsize=18)

                
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'pCO2_SOCAT'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'pCO2_SOCAT'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
            plt.show(block=False)
            
            
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(SOCAT_mean)): print('WARNING: The interpolated Glodap field contains NaNs')
                if np.isnan(np.min(FESOM_mean)): print('WARNING: The interpolated FESOM field contains NaNs')

                aux = np.where(np.isfinite(SOCAT_mean))

                title = 'Taylor Diagram for pCO$_2$'
                plot_taylor_norm(SOCAT_mean[aux],FESOM_mean[aux],mask=True,title=title)

                # fig export  
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'pCO2_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                                dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'pCO2_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                                bbox_inches='tight')
                plt.show(block=False)
                
            if output:
                    self.pco2_fesom = FESOM_mean
                    self.pco2_socat = SOCAT_mean
                    self.pco2_diff  = DIFF_mean



class plot_maps_co2f_Takahashi:
    '''
    class CO2f_Takahashi_comp(resultpath,savepath,meshpath,txtfile,first_year,last_year,
                 mapproj='pc',savefig=False,layerwise=False,runname='fesom')
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilepCO2,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='rob',
                 cmap='RdBu_r',
                 savefig=False,
                 cmap_extension='both',
                 verbose=False,
                 plotting=True,
                 output=False,
                 Taylor=True,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.txtfile = txtfile
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.cmap = cmap
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        self.cmap_extension = cmap_extension
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        years = np.arange(self.fyear, self.lyear+1,1)

        # load FESOM data -------------------------------------------------------------------------------------
        CO2ffesom = pf.get_data(resultpath, "CO2f", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)    

        CO2ffesom = -365.*CO2ffesom/1000. # converting to molC/m2/year
    
    
        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        unitfesom = 'air-sea CO$_2$ flux [mol C m$^{-2}$ yr$^{-1}$]' 

        # Reading CO2 flux data from Takahashi =========================================
        labeltakashi = 'Takahashi et al.'
        path = self.txtfile
        header = 109                                          # Number of lines in file to skip (with text)
        allvalues = []
        with open(path, 'r') as f:
            for _ in range(header):                             # Skipping text lines
                f.readline()
            for line in f:                                      # Saving all data in matrix "allvalues"
                allvalues.append(line.split())

        allvalues     = np.array(allvalues)                   # Converting list to array
        allvalues     = allvalues.astype(float)            # Converting string to float
        lat           = allvalues[:,0]                        # Saving all latitude values in array "lat"
        lat_matrix    = np.unique(lat)                           # 
        lon           = allvalues[:,1]                        # Saving all longitude values in array "lon"
        lon[lon>180.] = lon[lon>180.]-360.                    # Converting lon to values from -180 to 180
        lon_matrix    = np.unique(lon)                           #
        month         = allvalues[:,2]                        # Saving number of month in array "month"      
        flux          = allvalues[:,17]                       # CO2 flux in units [moles C/m2/month] saved in array "flux"

        matrix = np.zeros(shape=(72,40,12))                   # Initializing matrix [lon,lat,month] for gridded sorting of flux values
        for i in range(0,len(flux)):                          # Stepping through all values and adding them to proper position in the matrix
            n = np.where(lat_matrix == lat[i])[0] 
            m = np.where(lon_matrix == lon[i])[0]
            matrix[m,n,int(month[i]-1)] = flux[i]
        matrix[matrix==0] = np.nan                                 # Converting missing values to NaN
        CO2matrix = np.sum(matrix,axis=2)                     # Yearly CO2 flux [moles C/m2/yr]
        CO2matrix = CO2matrix.transpose()                     # Transposing 

        # ==============================================================================
        # Interpolation to fesom's grid

        lonbefore, latbefore = np.meshgrid(lon_matrix, lat_matrix)               # Matrix lon and lat prepared for interpolation

        CO2ftakashi = griddata((lonbefore.ravel(), latbefore.ravel()), CO2matrix.ravel(), (mesh.x2, mesh.y2), method='nearest')
        CO2ftakashi = np.ma.filled(CO2ftakashi, np.nan)

        # apply sea mask to Takashi as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        CO2ftakashi_ma = np.copy(CO2ftakashi)
        CO2ftakashi_ma[np.isnan(CO2ffesom)] = np.nan
        
        
        # Plotting -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nPlotting CO2 flux \nFESOM min = {0}, max = {1}\nTakashi min = {2}, max = {3}'.format(
                    np.nanmin(CO2ffesom),np.nanmax(CO2ffesom),
                    np.nanmin(CO2ftakashi_ma),np.nanmax(CO2ftakashi_ma)))
        
        if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(-9,9.2,.2)
                f1 = pf.subplot(mesh, fig, m1, [CO2ffesom],
                            levels = levels,
                            units=unitfesom, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom,
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [CO2ftakashi_ma], 
                            levels = levels,
                            units=unitfesom, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labeltakashi,
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-5,5.2,.2)
                f3 = pf.subplot(mesh, fig, m3, [CO2ffesom-CO2ftakashi_ma], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitfesom, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - Takahashi',
                            box=box,
                           )
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')

                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'CO2f_Taka'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'CO2f_Taka'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)

        # statistics  -------------------------------------------------------------------------------------            
        # preparation of datasets

        if Taylor:
            #get statistics only from valid Takashi points
            ind_stat = np.where(np.isfinite(CO2ftakashi_ma))

            if np.isnan(np.min(CO2ftakashi_ma[ind_stat])): print('WARNING: The interpolated Takashi field contains NaNs at depth')
            if np.isnan(np.min(CO2ffesom[ind_stat])): print('WARNING: The FESOM field contains NaNs at depth')

            title = 'Taylor Diagram for CO$_2$ flux'
            plot_taylor_norm(CO2ftakashi_ma[ind_stat],CO2ffesom[ind_stat],mask=True,title=title)

            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):                
                plt.savefig(self.savepath+self.runname+'_'+'CO2flux_Takashi_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'CO2flux_Takashi_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
                plt.show(block=False)
            
        if output:
                    self.co2f_fesom = CO2ffesom
                    self.co2f_taka  = CO2ftakashi_ma


class plot_maps_pco2_chau:
    '''
    class pCO2SOCATcomp(resultpath,savepath,meshpath,ncpath,first_year,last_year,
                mapproj='pc',savefig=False,layerwise=False,runname='fesom')
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileCO2f,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='rob',
                 SOCATvar='fgco2',
                 cmap='RdBu_r',
                 savefig=False,
                 cmap_extension='both',
                 verbose=False,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.cmap = cmap
        self.SOCATvar = SOCATvar
        self.cmap_extension = cmap_extension
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        self.verbose = verbose
        
        # derive SOCAT mapping projection -------------------------------------------------------------------------------------
        if((self.mapproj != 'pc') & (self.mapproj != 'rob')):
            print('Projection for plotting Chau et al. data is not supported! Choose "pc" or "rob".\nprojection set to "pc"')
            self.mapproj == 'rob'
        
        box=[-180, 180, -90, 90]

        self.mapproj = pf.get_proj(self.mapproj)
 
        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # load FESOM data -------------------------------------------------------------------------------------
        labelfesom = 'FESOM-REcoM'
        
        if SOCATvar == 'fgco2':
            unit = 'air-sea CO$_2$ flux [mol C m$^{-2}$ yr$^{-1}$]' 
        elif SOCATvar == 'spco2':
            unit = 'pCO$_2$ [$\mu$atm]'
        
        # interpolate Glodap data -------------------------------------------------------------------------------------
        SOCAT_input = load_chau_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.SOCATvar, 
                                self.fyear, self.lyear, get_overview=False)
        socat_int = SOCAT_input.socat_int
        
        print(np.shape)
        time_soccat = SOCAT_input.time
            
        int = np.intersect1d(years, np.arange(1970,2017))
        
        if np.size(int) != 0:
            previous_year=-1
            previous_month=-1
            FESOM = np.empty((len(mesh.x2),len(time_soccat)))
            SOCAT = np.empty((len(mesh.x2),len(time_soccat)))
            DIFF = np.empty((len(mesh.x2),len(time_soccat)))
            for i in range(len(time_soccat)):
                current_time = time_soccat[i]
                current_year = current_time.year
                current_month = current_time.month
                
                if current_year != previous_year:
                    sicfesom = pf.get_data(resultpath, "a_ice", current_year, mesh, 
                                       how=None, compute=True, runid=self.runname, silent=True)
                    if SOCATvar == 'fgco2':
                        pCO2fesom = pf.get_data(resultpath, "CO2f", current_year, mesh, 
                                       how=None, compute=True, runid=self.runname, silent=True)
                        pCO2fesom = -365.*pCO2fesom/1000. # converting to molC/m2/year
                    elif SOCATvar == 'spco2':
                        pCO2fesom = pf.get_data(resultpath, "pCO2s", current_year, mesh, 
                                       how=None, compute=True, runid=self.runname, silent=True)
                        
                if current_month != previous_month:
                    fesom_monthly = pCO2fesom[current_month-1,:]
                    sic_monthly = sicfesom[current_month-1,:]
                    
                # apply sea mask to Glodap as in FESOM ----------------------------------------------------------------------------------
                # assumption: there is no ocean where value in FESOM == 0
                socat_int_ma = np.copy(socat_int[:,i])
                socat_int_ma[fesom_monthly == 0] = 0

                socat_int_ma[sic_monthly>0.15]= np.nan
                fesom_monthly[sic_monthly>0.15]= np.nan
                
                diff = fesom_monthly - socat_int_ma
                
                DIFF[:,i]=diff
                SOCAT[:,i]=socat_int_ma
                FESOM[:,i]=fesom_monthly

                previous_year = current_year
                previous_month = current_month

            DIFF_mean = np.nanmean(DIFF, axis= 1)
            SOCAT_mean = np.nanmean(SOCAT, axis= 1)
            FESOM_mean = np.nanmean(FESOM, axis= 1)

            
        elif np.size(int) == 0:
            SOCAT_mean = SOCAT_input.socat_int_mean

            sicfesom = pf.get_data(resultpath, "a_ice", years, mesh, 
                                       how='mean', compute=True, runid=self.runname, silent=True)
            
            if SOCATvar == 'fgco2':
                FESOM_mean = pf.get_data(resultpath, "CO2f", years, mesh, 
                                       how='mean', compute=True, runid=self.runname, silent=True)
                FESOM_mean = -365.*FESOM_mean/1000. # converting to molC/m2/year
            elif SOCATvar == 'spco2':
                FESOM_mean = pf.get_data(resultpath, "pCO2s", years, mesh, 
                                       how='mean', compute=True, runid=self.runname, silent=True)
            
            FESOM_mean[sicfesom>0.15]= np.nan
            SOCAT_mean[sicfesom>0.15]= np.nan

            DIFF_mean = FESOM_mean - SOCAT_mean
            
        # ==============================================================================
        # plot FESOM and SOCAT   
        if(self.verbose):
            print('\nPlotting pCO2\nFESOM min = {0}, max = {1}'.format(
                        np.nanmin(FESOM_mean),np.nanmax(FESOM_mean)))
            print('Chau et al. min = {0}, max = {1}'.format(
                    np.nanmin(SOCAT_mean),np.nanmax(SOCAT_mean)))
        
#         if ((self.fyear < 1970) and (self.lyear > 2017)):
#             labelsocat = 'Chau et al. (1985-2020)'
#             label_diff = 'FESOM-REcoM - Chau et al. (1985-2020)'
#         elif ((self.fyear < 1970) and (self.lyear <= 2017) and (self.lyear >= 1970)):
#             labelsocat = 'Chau et al. (1985-{0})'.format(last_year)
#             label_diff = 'FESOM-REcoM - Chau et al. (1985-{0})'.format(last_year)
#         elif ((self.fyear >= 1970) and (self.lyear > 2017)):
#             labelsocat = 'Chau et al. ({0}-2020)'.format(first_year)
#             label_diff = 'FESOM-REcoM - Chau et al. ({0}-2020)'.format(last_year)
#         elif ((self.lyear < 1970)):
#             labelsocat = 'Chau et al. (1985-2020)'
#             label_diff = 'FESOM-REcoM - Chau et al. (no overlap)'
#         elif ((self.fyear > 2017)):
#             labelsocat = 'Chau et al. (1985-2020)'
#             label_diff = 'FESOM-REcoM - Chau et al. (no overlap)'
#         else:
#             labelsocat = 'Chau et al. ({0}-{1})'.format(self.fyear,self.lyear)
#             label_diff = 'FESOM-REcoM - Chau et al. ({0}-{1})'.format(self.fyear,self.lyear)
             
        labelsocat = 'Chau et al.'
        label_diff = 'FESOM-REcoM - Chau et al.'
        
        if plotting:
            fig = plt.figure(figsize=(15,12), constrained_layout=False)
            axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
            m1 = axes['A']
            
            if SOCATvar == 'fgco2':
                levels = np.arange(-6,6.5,.5)
            elif SOCATvar == 'spco2':
                levels = np.arange(180,620,20)
                
            f1 = pf.subplot(mesh, fig, m1, [FESOM_mean],
                            levels = levels,
                            units=unit, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension='both',
                            titles=labelfesom,
                            box=box,
                           )
                    
            m2 = axes['B']
            f2 = pf.subplot(mesh, fig, m2, [SOCAT_mean], 
                            levels = levels,
                            units=unit, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension='both',
                            titles=labelsocat,
                            box=box,
                           )
                    
            cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
            cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
            cbar1.set_label(unit, fontsize=18)
            cbar1.ax.tick_params(labelsize=18)
        
            m3 = axes['C']
            
            if SOCATvar == 'fgco2':
                levels_diff = np.arange(-5,5.5,.5)
            elif SOCATvar == 'spco2':
                levels_diff = np.arange(-120,130,10)
                
            f3 = pf.subplot(mesh, fig, m3, [FESOM_mean-SOCAT_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unit, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles=label_diff,
                            box=box,
                           )
                
            
            m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
            m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                        size=30, weight='bold')
            m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                        size=30, weight='bold')
            
            #fig.subplots_adjust(bottom=0.02)
            cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
            cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
            cbar2.set_label(unit, fontsize=18)
            cbar2.ax.tick_params(labelsize=18)

            #plt.tight_layout()
            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+str(SOCATvar)+'_CHAU'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+str(SOCATvar)+'_CHAU'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
            plt.show(block=False)
            
            
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(SOCAT_mean)): print('WARNING: The interpolated Glodap field contains NaNs')
                if np.isnan(np.min(FESOM_mean)): print('WARNING: The interpolated FESOM field contains NaNs')

                aux = np.where(np.isfinite(SOCAT_mean))
                
                if SOCATvar == 'fgco2':
                    title = 'Taylor Diagram for fCO$_2$'
                elif SOCATvar == 'spco2':
                    title = 'Taylor Diagram for pCO$_2$'
                    
                plot_taylor_norm(SOCAT_mean[aux],FESOM_mean[aux],mask=True,title=title)

                # fig export  
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+str(SOCATvar)+'_CHAU_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                                dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+str(SOCATvar)+'_CHAU_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                                bbox_inches='tight')
                plt.show(block=False)
                
            if output:
                    self.pco2_fesom = FESOM_mean
                    self.pco2_socat = SOCAT_mean
                    self.pco2_diff  = DIFF_mean


class plot_maps_npp_global:
    '''
    class NPPsurf_OCNPP_comp(runname,resultpath,savepath,mesh,matfileNPPsurf,first_year,last_year,
                 mapproj='rob',savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
                 
    n_levels = 1: number of mesh levels used for FESOM surface mean
    
    self.NPPnfesom_interp contains 2D dataset of 1x1 interpolated nanophytoplankton NPP
    self.NPPdfesom_interp contains 2D dataset of 1x1 interpolated diatom NPP
    self.NPPtfesom_interp contains 2D dataset of 1x1 interpolated of total NPP
    self.unitfesom contains str of FESOM NPP unit
    self.OCNPP contains 2D dataset of 1x1 interpolated of remotely sensed NPP
    self.lon longitude
    self.lat latitude
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 matfileNPPsurf=matfileNPPvgpm,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='rob',runname='fesom', 
                 savefig=False,output=False,
                 plotting=True,verbose=False,Taylor=True):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.matfileNPPsurf=matfileNPPsurf
        self.verbose = verbose
        self.taylor = Taylor
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
        
        # load OCCCI CHl.a data -------------------------------------------------------------------------------------
        matNPP = spio.loadmat(self.matfileNPPsurf, squeeze_me=True)
        
        lat         = np.arange(-90,90.-.1,1.)
        lon         = np.arange(-179.5,180.-.1,1.)
        latdic, londic = np.meshgrid(lat, lon)
        
        #annualchl   = np.log10(matChl['x'])
        npp = matNPP['NPP_CLIM']
        npp = np.nanmean(npp,axis=2) # For now, only take annual mean, seasonnal evaluation will be implemented later
        npp = npp.T
        
        OCNPPlabel = matfileNPPsurf[-1-12:-9]
        OCNPPunit = 'NPP [mg C m$^{-2}$ d$^{-1}$]'
        print(' !!! Satellite data for the 1998-2019 period only !!!')
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        
        lon_fesom = mesh.x2
        lat_fesom = mesh.y2        
        
        NPPnfesom = pf.get_data(self.resultpath, "NPPn", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        
        labelfesomNano = 'FESOM Nanophyto NPP {0}-{1}'.format(self.fyear,self.lyear)        

        
        NPPdfesom = pf.get_data(self.resultpath, "NPPd", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
        
        labelfesomDia = 'FESOM Diatom NPP {0}-{1}'.format(self.fyear,self.lyear)
        
        from pathlib import Path
        cocco_path = Path(self.resultpath + '/NPPc.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/NPPp.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        diah_path = Path(self.resultpath + '/NPPDiaH.fesom.'+str(years[0])+'.nc')
        
        
        if cocco_path.is_file():
            NPPcfesom = pf.get_data(self.resultpath, "NPPc", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
            labelfesomCocco = 'FESOM Cocco NPP {0}-{1}'.format(self.fyear,self.lyear)
            
            if phaeo_path.is_file():
                NPPpfesom = pf.get_data(self.resultpath, "NPPp", years, mesh, 
                                   how="mean", compute=True, runid=self.runname, silent=True)
                labelfesomPhaeo = 'FESOM Phaeo NPP {0}-{1}'.format(self.fyear,self.lyear)
                print('4-phytoplankton model is used')

                if diah_path.is_file():
                    NPPhfesom = pf.get_data(self.resultpath, "NPPdiaH", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    labelfesomDiaH = 'FESOM DiaH NPP {0}-{1}'.format(self.fyear,self.lyear)
                    print('5-phytoplankton model is used')
                
            else:
                print('3-phytoplankton model is used')
                
        else:
            print('2-phytoplankton model is used')
        
        
        # convert FESOM CHl.a data -------------------------------------------------------------------------------------
        # #########################
        # FESOM outputs the data in mg/m3 already, wrong units set in netcdf !?!?!
        # #########################
        if True:
            #C : 12.01 g/mol
            conv = 12.01
            NPPdfesom = NPPdfesom * conv # mmol/m2/day to mg/m2/day
            NPPnfesom = NPPnfesom * conv
            if cocco_path.is_file():
                NPPcfesom = NPPcfesom * conv
            if phaeo_path.is_file():
                NPPpfesom = NPPpfesom * conv
            unitfesom = 'NPP [mg C m$^{-2}$ d$^{-1}$]'
        else: 
            print('***\nFESOM data in not converted...\n***')
            unitfesom = 'NPP [mmol C m$^{-2}$ d$^{-1}$]'
         
        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        
        # interpolate FESOM CHl.a to regular -------------------------------------------------------------------------------------
        NPPn_interp = pf.fesom2regular(
                data = NPPnfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        NPPd_interp = pf.fesom2regular(
                data = NPPdfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        if cocco_path.is_file():
            NPPc_interp = pf.fesom2regular(
                data = NPPcfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if phaeo_path.is_file():
            NPPp_interp = pf.fesom2regular(
                data = NPPpfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if diah_path.is_file():
            NPPh_interp = pf.fesom2regular(
                data = NPPhfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        # Nanophyto + Diatoms (+ Coccos + Phaeo): TOTAL NPP -------------------------------------------------------------------------------------
        
        NPPt_interp = NPPn_interp + NPPd_interp
        if cocco_path.is_file():
            NPPt_interp = NPPt_interp + NPPc_interp
        if phaeo_path.is_file():
            NPPt_interp = NPPt_interp + NPPp_interp
        if diah_path.is_file():
            NPPt_interp = NPPt_interp + NPPh_interp
        


        # apply sea mask to OCCCI as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        OCNPP_ma = np.copy(npp)
        OCNPP_ma[~np.isfinite(NPPt_interp)] = np.nan
        
        # Derive regular grid area
        Aarea = pf.get_regulargrid_area(lon,lat)
        
        # Compute areal NPP
        NPP_OC_so = np.nansum(OCNPP_ma[:-1,lat<-50] * Aarea[:,lat[:-1]<-50])
        NPP_FE_so = np.nansum(NPPt_interp[:-1,lat<-50] * Aarea[:,lat[:-1]<-50])
        NPP_OC_glo = np.nansum(OCNPP_ma[:-1,:-1] * Aarea[:,:])
        NPP_FE_glo = np.nansum(NPPt_interp[:-1,:-1] * Aarea[:,:])
        print(' NPP SAT Southern Ocean [PgC / yr] = ', 365*NPP_OC_so/1e18)
        print(' NPP REcoM Southern Ocean [PgC / yr] = ', 365*NPP_FE_so/1e18)
        print(' NPP SAT Global [PgC / yr] = ', 365*NPP_OC_glo/1e18)
        print(' NPP REcoM Global [PgC / yr] = ', 365*NPP_FE_glo/1e18)
        
        # Convert to log10 scale
        OCNPP_ma_log10 = np.log10(OCNPP_ma)
        NPPn_log10     = np.log10(NPPn_interp)
        NPPd_log10     = np.log10(NPPd_interp)
        if cocco_path.is_file():
            NPPc_log10     = np.log10(NPPc_interp)
        if phaeo_path.is_file():
            NPPp_log10     = np.log10(NPPp_interp)
        NPPt_log10     = np.log10(NPPt_interp)
        
        # check CHl.a data -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nNPP\nOC nmin = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f}'.format(
                    np.nanmin(NPPt_interp),np.nanmax(NPPt_interp),
                    np.nanmin(OCNPP_ma),np.nanmax(OCNPP_ma)))
        
        if (self.plotting):
            # plot each PP dataset -------------------------------------------------------------------------------------        
            levels = 1000*np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                               0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                               1,2,3]) #+,4,5,7?

            ticks = np.array([0,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1,3])*1000

            def mygrid(m):
                #m.coastlines(resolution='110m', color='black', linewidth=1)
                m.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='gray')
                #m.set_xlim(box[0], box[1])
                #m.set_ylim(box[2], box[3])


            # if phaeos and coccos are used ----------------------------------------------------------------------------------------
            if cocco_path.is_file() and phaeo_path.is_file() and diah_path.is_file():
                #print('5-phytoplankton model is printed')
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        ABC
                        CDE
                        FFF
                        HHH
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, NPPn_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_extent(box, ccrs.PlateCarree())
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, NPPd_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_extent(box, ccrs.PlateCarree())
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)
                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, NPPc_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_extent(box, ccrs.PlateCarree())
                m3.set_title('FESOM-REcoM Coccolithophores', fontsize=16)
                
                # FESOM phaeocystis
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, NPPp_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_extent(box, ccrs.PlateCarree())
                m4.set_title('FESOM-REcoM Phaeocystis', fontsize=16)

                # FESOM DiaH
                m5 = axes['E']
                f5 = m5.pcolormesh(londic, latdic, NPPh_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m5)
                m5.set_extent(box, ccrs.PlateCarree())
                m5.set_title('FESOM-REcoM highly silicified Diatioms', fontsize=16)

                # FESOM
                m6 = axes['F']
                f6 = m6.pcolormesh(londic, latdic, NPPt_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m6)
                m6.set_extent(box, ccrs.PlateCarree())
                m6.set_title('FESOM-REcoM Total', fontsize=16)

                # OC-CCI
                m7 = axes['G']
                f7 = m7.pcolormesh(londic, latdic, OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m7)
                m7.set_extent(box, ccrs.PlateCarree())
                m7.set_title(OCNPPlabel, fontsize=16)
                

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label(unitfesom, fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-600,620,20)
                m8 = axes['H']
                f8 = m8.pcolormesh(londic, latdic, NPPt_interp - OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
                #f3.set_clim([-2, 2])

                mygrid(m8)
                m8.set_extent(box, ccrs.PlateCarree())
                m8.set_title('FESOM-REcoM - '+matfileNPPsurf[-1-12:-9], fontsize=16)


                 # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f7,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                extend = 'both',
                                #location ='bottom',
                                ticks = [-600,-400,-200,0,200,400,600]) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label(unitfesom, fontsize=16)

                
                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                m7.text(-0.12, 1.05, 'G', transform=m7.transAxes,
                            size=30, weight='bold')
                m8.text(-0.12, 1.05, 'H', transform=m8.transAxes,
                            size=30, weight='bold')
                

                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())
                m7.set_extent(box, ccrs.PlateCarree())
                m8.set_extent(box, ccrs.PlateCarree())

            
            elif cocco_path.is_file() and phaeo_path.is_file() and not diah_path.is_file():
                #print('4-phytoplankton model is printed')
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        AB
                        CD
                        EF
                        GG
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, NPPn_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_extent(box, ccrs.PlateCarree())
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, NPPd_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_extent(box, ccrs.PlateCarree())
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)
                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, NPPc_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_extent(box, ccrs.PlateCarree())
                m3.set_title('FESOM-REcoM Coccolithophores', fontsize=16)
                
                # FESOM phaeocystis
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, NPPp_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_extent(box, ccrs.PlateCarree())
                m4.set_title('FESOM-REcoM Phaeocystis', fontsize=16)

                # FESOM
                m5 = axes['E']
                f5 = m5.pcolormesh(londic, latdic, NPPt_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m5)
                m5.set_extent(box, ccrs.PlateCarree())
                m5.set_title('FESOM-REcoM Total', fontsize=16)

                # OC-CCI
                m6 = axes['F']
                f6 = m6.pcolormesh(londic, latdic, OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m6)
                m6.set_extent(box, ccrs.PlateCarree())
                m6.set_title(OCNPPlabel, fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label(unitfesom, fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-600,620,20)
                m7 = axes['G']
                f7 = m7.pcolormesh(londic, latdic, NPPt_interp - OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
                #f3.set_clim([-2, 2])

                mygrid(m7)
                m7.set_extent(box, ccrs.PlateCarree())
                m7.set_title('FESOM-REcoM - '+matfileNPPsurf[-1-12:-9], fontsize=16)


                 # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f7,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                extend = 'both',
                                #location ='bottom',
                                ticks = [-600,-400,-200,0,200,400,600]) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label(unitfesom, fontsize=16)

                
                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                m7.text(-0.12, 1.05, 'G', transform=m7.transAxes,
                            size=30, weight='bold')
                

                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())
                m7.set_extent(box, ccrs.PlateCarree())
            
            # if coccos are used ---------------------------------------------------------------------------------------------   
            
            elif cocco_path.is_file() and not phaeo_path.is_file() and not diah_path.is_file():
                #print('3-phytoplankton model is printed')
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        ABC
                        DEF
                        GGG
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, NPPn_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_extent(box, ccrs.PlateCarree())
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, NPPd_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_extent(box, ccrs.PlateCarree())
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)
                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, NPPc_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_extent(box, ccrs.PlateCarree())
                m3.set_title('FESOM-REcoM Coccolithophores', fontsize=16)

                # FESOM
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, NPPt_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_extent(box, ccrs.PlateCarree())
                m4.set_title('FESOM-REcoM Total', fontsize=16)
                
                
                # Placeholder
                m7 = axes['E']
                axes['E'].remove()

                # OC-CCI
                m5 = axes['F']
                f5 = m5.pcolormesh(londic, latdic, OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m5)
                m5.set_extent(box, ccrs.PlateCarree())
                m5.set_title(OCNPPlabel, fontsize=16)

                

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label(unitfesom, fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-600,620,20)
                m6 = axes['G']
                f6 = m6.pcolormesh(londic, latdic, NPPt_interp - OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
                #f3.set_clim([-2, 2])

                mygrid(m6)
                m6.set_extent(box, ccrs.PlateCarree())
                m6.set_title('FESOM-REcoM - '+matfileNPPsurf[-1-12:-9], fontsize=16)


                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f6,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                extend = 'both',
                                #location ='bottom',
                                ticks = [-600,-400,-200,0,200,400,600]) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label(unitfesom, fontsize=16)

                
                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                m7.text(-0.12, 1.05, '', transform=m7.transAxes,
                            size=30, weight='bold')

                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())
                
                
            # if coccos are not used ------------------------------------------------------------------------------------    
            
            else:
                #print('2-phytoplankton model is printed')
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        AB
                        CD
                        EE
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, NPPn_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_extent(box, ccrs.PlateCarree())
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, NPPd_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_extent(box, ccrs.PlateCarree())
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)

                # OC-CCI
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m4)
                m4.set_extent(box, ccrs.PlateCarree())
                m4.set_title(OCNPPlabel, fontsize=16)


                # FESOM
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, NPPt_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_extent(box, ccrs.PlateCarree())
                m3.set_title('FESOM-REcoM Total', fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label(unitfesom, fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-600,620,20)
                m5 = axes['E']
                f5 = m5.pcolormesh(londic, latdic, NPPt_interp - OCNPP_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
                #f3.set_clim([-2, 2])

                mygrid(m5)
                m5.set_extent(box, ccrs.PlateCarree())
                m5.set_title('FESOM-REcoM - '+matfileNPPsurf[-1-12:-9], fontsize=16)

                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f5,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                extend = 'both',
                                #location ='bottom',
                                ticks = [-600,-400,-200,0,200,400,600]) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label(unitfesom, fontsize=16)
                
                
                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')

                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())

            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'OCNPP_'+matfileNPPsurf[-1-12:-9]+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'OCNPP_'+matfileNPPsurf[-1-12:-9]+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  

        if(self.Taylor):
            # statistics  -------------------------------------------------------------------------------------            
            # preparation of datasets
            if np.isnan(np.min(OCNPP_ma_log10)): print('WARNING: OCNPP field contains NaNs')
            if np.isnan(np.min(NPPt_log10)): print('WARNING: FESOM field contains NaNs')

            # get statistics only from valid OCCCI gridpoints 
            ind_stat = np.where(np.isfinite(OCNPP_ma_log10))

            title = 'log10 surface NPP'
            print('\nStatistics for '+title)
            plot_taylor_norm(OCNPP_ma_log10[ind_stat],NPPt_log10[ind_stat],
                                        mask=True,title=title)
            
            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):                
                plt.savefig(self.savepath+self.runname+'_'+'OCNPP_Taylor_'+matfileNPPsurf[-1-12:-9]+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'OCNPP_Taylor_'+matfileNPPsurf[-1-12:-9]+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
            plt.show(block=False) 
        
        # store interpolated datasets ----------------------------------------------------------------------------------
        if output:
            self.NPPd_interp = NPPd_interp
            self.NPPn_interp = NPPn_interp
            if cocco_path.is_file():
                self.NPPc_interp = NPPc_interp
            if phaeo_path.is_file():
                self.NPPp_interp = NPPp_interp
            self.NPPt_interp = NPPt_interp
            self.unit = unitfesom
            self.NPPt_OC = OCNPP_ma

class plot_maps_npp_arctic:
    '''
    class NPPsurf_OCNPP_comp(resultpath,savepath,mesh,npfile,first_year,last_year,
                 savefig=False, verbose=False, output=False, plotting=True, Taylor=True)
                 
    self.NPPnfesom_interp contains 2D dataset of 1x1 interpolated nanophytoplankton NPP
    self.NPPdfesom_interp contains 2D dataset of 1x1 interpolated diatom NPP
    self.NPPtfesom_interp contains 2D dataset of 1x1 interpolated of total NPP
    self.unitfesom contains str of FESOM NPP unit
    self.OCNPP contains 2D dataset of 1x1 interpolated of remotely sensed NPP
    self.lon longitude
    self.lat latitude
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 npfile=npfileNPPArcLewis,
                 first_year=first_year,
                 last_year=last_year,
                 savefig=False,
                 output=False,plotting=True, 
                 verbose = False, Taylor=True,runname = 'fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.npfile = npfile
        self.verbose = verbose
        self.taylor = Taylor
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor

        box=[-180, 180, 60, 90]
        mapproj = pf.get_proj('np')

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
        
        # load OCCCI CHl.a data -------------------------------------------------------------------------------------
        NPP_SAT = np.load(npfile)
        
        londic, latdic = np.meshgrid(np.arange(-179.25,180.,0.25), np.arange(-90,90.25,0.25))
        unit = 'NPP [mg C m$^{-2}$ d$^{-1}$]'

        if 'ARRIGO' in npfile:
            tag = 'LEWIS'
            print('Lewis et al. dataset selected')
            label = 'Lewis et al. [2003-2018]'
            print(' !!! Only MODIS Satellite for the 2003-2018 period !!!')
        elif 'CMEMS' in npfile:
            NPP_SAT[latdic<=60]=np.nan
            tag = 'CMEMS'
            print('Globcolour dataset selected')    
            label = 'Globcolour [2003-2018]'
            print(' !!! Only MODIS Satellite for the 2003-2018 period !!!')
        else:
            print('wrong dataset selected, please select either {Lewis} or {OCCCI}')
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        NPPn = pf.get_data(resultpath, "NPPn", years, mesh, how='mean', compute=True, silent=True)
        NPPd = pf.get_data(resultpath, "NPPd", years, mesh, how='mean', compute=True, silent=True)
        
        cocco_path = Path(self.resultpath + '/NPPc.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/NPPp.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        diah_path = Path(self.resultpath + '/NPPdiaH.fesom.'+str(years[0])+'.nc')
        
        if cocco_path.is_file():
            NPPc = pf.get_data(self.resultpath, "NPPc", years, mesh, how="mean", compute=True, silent=True)
            
            if phaeo_path.is_file():
                NPPp = pf.get_data(self.resultpath, "NPPp", years, mesh, how="mean", compute=True, silent=True)

                if phaeo_path.is_file():
                    NPPh = pf.get_data(self.resultpath, "NPPp", years, mesh, how="mean", compute=True, silent=True)

                    print('5-phytoplankton model is used')
                    NPP_MODEL = 365* (NPPh+NPPd+NPPn+NPPc+NPPp)*12.01 /1e3

                else:
                    
                    print('4-phytoplankton model is used')
                    NPP_MODEL = 365* (NPPd+NPPn+NPPc+NPPp)*12.01 /1e3
            else: 
                print('3-phytoplankton model is used')
                NPP_MODEL = 365* (NPPd+NPPn+NPPc)*12.01 /1e3 
            
        else:
            print('2-phytoplankton model is used')
            NPP_MODEL = 365* (NPPd+NPPn)*12.01 /1e3

        # interpolate FESOM CHl.a to regular -------------------------------------------------------------------------------------
        NPPfesom_interp = pf.fesom2regular(
                data = NPP_MODEL,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        NPPfesom_interp_log10 = np.log10(NPPfesom_interp)
        fesom_label = 'FESOM-REcoM NPP {0}-{1}'.format(self.fyear,self.lyear)        

        # apply sea mask to OCCCI as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        NPPsat_ma = np.copy(NPP_SAT)
        NPPsat_ma[~np.isfinite(NPPfesom_interp)] = np.nan
        NPPsat_ma_log10 = np.log10(NPPsat_ma)

        # plotting -------------------------------------------------------------------------------------
        levels = 100*np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                                   0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                                   1,2,3,4,5])
        ticks = [0,1,3,5,7,10,30,50,70,100,300,500]
        ticks_label = ['0','1','3','5','7','10','30','50','70','100','300','500']

        def mygrid(m):
            m.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='gray')
                
        fig, axes = plt.subplots(1,3, 
                                     subplot_kw=dict(projection=mapproj),
                                     gridspec_kw={'hspace': 0.01, 'wspace': 0.1},
                                     figsize=(20,7), constrained_layout=False)     

        # REcoM
        m1 = axes[0]
        f1 = m1.pcolormesh(londic, latdic, NPPfesom_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
        mygrid(m1)
        m1.set_title(fesom_label, fontsize=16)


        # Satellite
        m2 = axes[1]
        f2 = m2.pcolormesh(londic, latdic, NPPsat_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
        mygrid(m2)
        m2.set_title(label, fontsize=16)

        # add one colorbar for first row plots below figure
        cbar = fig.colorbar(f1,
                                ax = axes[:2], 
                                location ='bottom',
                                extend = 'max',
                                ticks = ticks,
                                fraction=0.046, pad=0.04) 
        #cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_xticklabels(ticks_label, fontsize=16) 
        cbar.set_label(unit, fontsize=16)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
                
        # REcoM - Satellite
        levels_diff = 100*np.arange(-3,3,0.125)
        m3 = axes[2]
        f3 = m3.pcolormesh(londic, latdic, NPPfesom_interp - NPPsat_ma, 
                                   transform = ccrs.PlateCarree(),cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
        mygrid(m3)
        m3.set_title('FESOM-REcoM - '+label, fontsize=16)

        # add one colorbar for difference plot below figure
        cbar = fig.colorbar(f3,
                            ax = axes[2], 
                            orientation = 'horizontal',
                            #location ='bottom',
                            ticks = [-300,-200,-100,0,100,200,300],
                            extend = 'both',
                            fraction=0.046, pad=0.04) 
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(unit, fontsize=16)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
                
        m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
        m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
        m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                    
        m1.set_extent(box, ccrs.PlateCarree())
        m2.set_extent(box, ccrs.PlateCarree())
        m3.set_extent(box, ccrs.PlateCarree())

        # fig export  -------------------------------------------------------------------------------------
        if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'ArcNPP_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'ArcNPP_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
        plt.show(block=False)

        if(self.Taylor):
            # statistics  -------------------------------------------------------------------------------------            
            # preparation of datasets
            if np.isnan(np.min(NPPsat_ma_log10)): print('WARNING: Satellite field contains NaNs')
            if np.isnan(np.min(NPPfesom_interp_log10)): print('WARNING: FESOM field contains NaNs')

            # get statistics only from valid OCCCI gridpoints 
            ind_stat = np.where(np.isfinite(NPPsat_ma_log10))

            title = 'log10 NPP'
            print('\nStatistics for '+title)
            plot_taylor_norm(NPPsat_ma_log10[ind_stat],NPPfesom_interp_log10[ind_stat],
                                    mask=True,title=title)

            
            #plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'ArcNPP_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'_Taylor.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'ArcNPP_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'_Taylor.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  
        
        if(self.output):
            self.lon = londic
            self.lat = latdic
            self.chl_oc = NPPsat_ma
            self.chl_fesom = NPPfesom_interp
            self.unit = unit


class plot_maps_mld:
    '''
    class MLD_comp(runname,resultpath,savepath,mesh,matfileMLD,first_year,last_year,
                 mapproj='pc',savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
                 
    n_levels = 1: number of mesh levels used for FESOM surface mean
    
    self.MLDfesom_interp contains 2D dataset of 1x1 interpolated MLD
    self.MLDobs contains 2D dataset of 1x1 interpolated of in situ MLD climatology
    self.lon longitude
    self.lat latitude
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfilepCO2,
                 first_year=first_year,
                 last_year=last_year,
                 ptype="cflog",
                 mapproj='rob',cmap_extension='max',cmap = cmo.cm.deep_r,
                 savefig=False,runname='fesom',
                 output=False,plotting=True, verbose = False, Taylor=True):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.matfileMLD=matfileMLD
        self.verbose = verbose
        self.taylor = Taylor
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor
        self.cmap = cmap
        self.cmap_extension = cmap_extension
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
        
        # load Atlas MLD data -------------------------------------------------------------------------------------
        
        MLDinput = load_mld_data(runname,resultpath,mesh,matfileMLD)
        MLD_sept_int = np.array(MLDinput.mld_sept_int)
        MLD_marc_int = np.array(MLDinput.mld_marc_int)
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(self.meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)      
        
        MLDfesom = pf.get_data(self.resultpath, "MLD2", years, mesh, 
                               how=None, compute=False, runid=self.runname, silent=True)
        #MLD = MLD.resample(time='MS').mean(dim='time').compute()
        MLDfesom = -MLDfesom.groupby('time.month').mean('time')
        
        MLDfesom_marc = np.array(MLDfesom[2,:])
        MLDfesom_sept = np.array(MLDfesom[8,:])
        
        #labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        labelfesom = 'FESOM'
        labelAtlas = 'Atlas'
        unit = 'MLD [m]'
        
        
        # apply sea mask to MLD as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        MLD_sept_int_ma = np.copy(MLD_sept_int)
        MLD_sept_int_ma[~np.isfinite(MLDfesom_sept)] = np.nan
        
        MLD_marc_int_ma = np.copy(MLD_marc_int)
        MLD_marc_int_ma[~np.isfinite(MLDfesom_marc)] = np.nan
        
        
        # check CHl.a data -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nWinter MLD\nAtlas nmin = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f}'.format(
                    np.nanmin(MLD_marc_int_ma),np.nanmax(MLD_marc_int_ma),
                    np.nanmin(MLDfesom_marc),np.nanmax(MLDfesom_marc)))
            print('\nSummer MLD\nAtlas nmin = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f}'.format(
                    np.nanmin(MLD_sept_int_ma),np.nanmax(MLD_sept_int_ma),
                    np.nanmin(MLDfesom_sept),np.nanmax(MLDfesom_sept)))
        
        if plotting:

            levels = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                               1,2,3,4,5,6,7,8])*100
            ticks = [0.05,0.1,0.3,0.5,0.7,1,3,5,6,7,8]*100
            ticks_label = ['0.05','0.1','0.3','0.5','0.7','1','3','7'] # +7 ?
            # plot each dataset -------------------------------------------------------------------------------------        
            #levels = np.arange(0,620,20)
            levels_diff = np.arange(-400,420,20)

            fig = plt.figure(figsize=(12,5), constrained_layout=True)
            axes = fig.subplot_mosaic(
                    """
                    ABC
                    DEF
                    """,
                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01}, 
                    subplot_kw=dict(projection=self.mapproj))             
            fig.get_layout_engine().set(w_pad=0.2)
            
            # FESOM 
            m1 = axes['A']
            f1 = pf.subplot(mesh, fig, m1, [MLDfesom_marc],
                                levels = levels,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom,
                                box= box,
                                ptype=ptype,
                               )

            # Atlas 
            m2 = axes['B']
            f2 = pf.subplot(mesh, fig, m2, [MLD_marc_int_ma],
                                levels = levels,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelAtlas,
                                box= box,
                                ptype=ptype,
                               )

            # FESOM - Atlas
            m3 = axes['C']
            f3 = pf.subplot(mesh, fig, m3, [MLDfesom_marc - MLD_marc_int_ma],
                                levels = levels_diff,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - Atlas',
                                box= box,
                               )
            
            


            # FESOM
            m4 = axes['D']
            f4 = pf.subplot(mesh, fig, m4, [MLDfesom_sept],
                                levels = levels,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                box= box,
                                ptype=ptype,
                               )

            # Atlas 

            m5 = axes['E']
            f5 = pf.subplot(mesh, fig, m5, [MLD_sept_int_ma],
                                levels = levels,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                box= box,
                                ptype=ptype,
                               )
            
            # FESOM - Atlas 
            m6 = axes['F']
            f6 = pf.subplot(mesh, fig, m6, [MLDfesom_sept - MLD_sept_int_ma],
                                levels = levels_diff,
                                units=unit, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                box= box,
                               )
            
            fig.subplots_adjust(bottom=0.2)
            
            cbar1_ax = fig.add_axes([0.10, 0.0001, 0.52, 0.02])
            cbar1 = fig.colorbar(f5,
                            cax = cbar1_ax, 
                            orientation = 'horizontal',
                            #ticks = ticks,
                            fraction=0.1, pad=0.1) 
            cbar1.set_label(unit, fontsize=16)
            cbar1.ax.tick_params(labelsize=16)
            

            cbar2_ax = fig.add_axes([0.75, 0.0001, 0.21, 0.02])
            cbar2 = fig.colorbar(f6,
                            cax = cbar2_ax, 
                            orientation = 'horizontal',
                            #location ='bottom',
                            ticks = [-400,-200,0,200,400],
                                )
            cbar2.ax.tick_params(labelsize=16)
            cbar2.set_label(unit, fontsize=16)
            
            
            m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
            m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                        size=30, weight='bold')
            m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                        size=30, weight='bold')
            m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                        size=30, weight='bold')
            m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                        size=30, weight='bold')
            m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                        size=30, weight='bold')
            # add one colorbar for difference plot below figure

            m1.text(-0.18, .3, 'March', transform=m1.transAxes,
                        size=18, rotation=90)
            m4.text(-0.18, .05, 'September', transform=m4.transAxes,
                        size=18, rotation=90)
            
            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'MLD'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'MLD'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  
            

#        if(self.Taylor):
#             # statistics  -------------------------------------------------------------------------------------            
#             # preparation of datasets
#             if np.isnan(np.min(OCNPP_ma_log10)): print('WARNING: OCNPP field contains NaNs')
#             if np.isnan(np.min(NPPt_log10)): print('WARNING: FESOM field contains NaNs')

#             # get statistics only from valid OCCCI gridpoints 
#             ind_stat = np.where(np.isfinite(OCNPP_ma_log10))

#             title = 'MLD'
#             print('\nStatistics for '+title)
#             plot_taylor_norm(OCNPP_ma_log10[ind_stat],NPPt_log10[ind_stat],
#                                         mask=True,title=title)

#             # fig export  -------------------------------------------------------------------------------------
#             if(self.savefig==True):                
#                 plt.savefig(self.savepath+self.runname+'_'+'OCNPP_Taylor'+'_'+str(years[0])+'to'+str(years[1])+'.png', 
#                             dpi = 300, bbox_inches='tight')
#             plt.show(block=False) 
        
#        # store interpolated datasets ----------------------------------------------------------------------------------
#        if output:
#             self.NPPd_interp = NPPd_interp
#             self.NPPn_interp = NPPn_interp
#             self.NPPt_interp = NPPt_interp
#             self.unit = unitfesom
#             self.NPPt_OC = OCNPP_ma
  
def plot_maps_maredat_depths(model, figlabel, savelabel, lon_maredat, lat_maredat, 
                      mapproj, first_year, last_year, savefig=False, savepath=''):
    '''
    Plot log10 FESOM layered means
    
    Input:
    model: FESOM data array with 4 data layers, i.e. DiaC_interp_all as output of fesom_to_maredat_levels()
    figlabel: used for xlabel
    savelabel: used to save the produced graph ('FESOM_depths_asMarEDat_'+label...)
    lon_maredat
    lat_maredat
    mapproj: as ccrs format
    savefig=False
    savepath
    '''
    
    if(mapproj == 'pc'): mapproj = ccrs.PlateCarree()
    elif(mapproj == 'rob'): mapproj = ccrs.Robinson()
    else: raise ValueError('Map projecction not supported')   
        
    if(savepath == ''):
        if(savefig == True):
            raise ValueError('Input for saving graph insufficient')
        
    print(figlabel)
    
    fig, axes = plt.subplots(2,2, 
                             subplot_kw=dict(projection=mapproj),
                             gridspec_kw={'hspace': 0.001, 'wspace': 0.1},
                             figsize=(20,15))
    fig.tight_layout()

    # 0-5 m
    m1 = axes[0,0]
    f1 = m1.pcolormesh(lon_maredat, lat_maredat, np.log10(model[0]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m1)
    m1.set_title('FESOM 0-5 m', fontsize=20)


    # 5-25 m
    m2 = axes[0,1]
    f2 = m2.pcolormesh(lon_maredat, lat_maredat, np.log10(model[1]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m2)
    m2.set_title('FESOM 5-25 m', fontsize=20)

    # 25-100 m
    m3 = axes[1,0]
    f3 = m3.pcolormesh(lon_maredat, lat_maredat, np.log10(model[2]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m3)
    m3.set_title('FESOM 25-100 m', fontsize=20)

    # 100-bottom m
    m4 = axes[1,1]
    f4 = m4.pcolormesh(lon_maredat, lat_maredat, np.log10(model[3]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m4)
    m4.set_title('FESOM 100 m - bottom', fontsize=20)

    # add one colorbar for all plots below figure
    cbar = fig.colorbar(f1,
                        ax = axes[1,:2], 
                        location ='bottom',
                        ticks=[np.log10(0.001),np.log10(0.01), np.log10(0.1),np.log10(0.5), np.log10(1),np.log10(10),np.log10(50),np.log10(100),np.log10(200)],
                        fraction=0.1, pad=0.1) 
    cbar.ax.set_xticklabels(['0.001','0.01','0.1','0.5','1','10','50','100','200'], fontsize=20) 
    cbar.set_label(figlabel+', Diatom Biomass [mg C m$^{-3}$]', fontsize=20)
    #cbar.set_label('Log$_{10}$ Biomass [$\mu$g L $^{-1}$]', fontsize=20)

    if(savefig == True):
        fig.savefig(savepath+'FESOM_depths_asMarEDat_'+savelabel+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
    plt.show()

def plot_maps_maredat_overview(maredat_layered_mean,lon_maredat, lat_maredat,mapproj = 'rob'):
    '''
    Input: 
    maredat_layered_mean: maredat data as array[4] layered into 0-5, 5-25, 25-100, 100m-bottom (mean in each depth interval)
    lon_maredat
    lat_maredat
    mapproj = 'rob': mapprojection abbreviation as in pyfesom2
    
    Output:
    figure
    '''
    
    if(mapproj == 'pc'): mapproj = ccrs.PlateCarree()
    elif(mapproj == 'rob'): mapproj = ccrs.Robinson()
    else: raise ValueError('Map projecction not supported')
    
    fig, axes = plt.subplots(2,2, 
                             subplot_kw=dict(projection=mapproj),
                             gridspec_kw={'hspace': 0.001, 'wspace': 0.1},
                             figsize=(20,15))
    fig.tight_layout()

    # 0-5 m
    m1 = axes[0,0]
    f1 = m1.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_mean[0]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(100), 
                        cmap=plt.cm.viridis)
    mygrid(m1)
    m1.set_title('MarEDAT 0-5 m', fontsize=20)


    # 5-25 m
    m2 = axes[0,1]
    f2 = m2.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_mean[1]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(100), 
                        cmap=plt.cm.viridis)
    mygrid(m2)
    m2.set_title('MarEDAT 5-25 m', fontsize=20)

    # 25-100 m
    m3 = axes[1,0]
    f3 = m3.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_mean[2]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(100), 
                        cmap=plt.cm.viridis)
    mygrid(m3)
    m3.set_title('MarEDAT 25-100 m', fontsize=20)

    # 100-bottom m
    m4 = axes[1,1]
    f4 = m4.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_mean[3]), 
                       transform = ccrs.PlateCarree(),
                        shading='flat', vmin=np.log10(0.001), vmax=np.log10(100), 
                        cmap=plt.cm.viridis)
    mygrid(m4)
    m4.set_title('MarEDAT 100 m - bottom', fontsize=20)

    # add one colorbar for all plots below figure
    cbar = fig.colorbar(f1,
                        ax = axes[1,:2], 
                        location ='bottom',
                        ticks=[np.log10(0.001),np.log10(0.01), np.log10(0.1),np.log10(0.5), np.log10(1),np.log10(10),np.log10(50)],
                        fraction=0.1, pad=0.1) 
    cbar.ax.set_xticklabels(['0.001','0.01','0.1','0.5','1','10','50'], fontsize=20) 
    cbar.set_label('Diatom Biomass [mg C m$^{-3}$]', fontsize=20)
    #cbar.set_label('Log$_{10}$ Biomass [$\mu$g L $^{-1}$]', fontsize=20)

    plt.show()
    return fig

def plot_maps_maredat_2cols(fesom_layered_sum, figlabel, savelabel, maredat_layered_sum, lon_maredat, lat_maredat, 
                      mapproj, years, savefig=False, savepath=''):
    '''
    Plot MarEDAT and FESOM data side by side in two columns, seperated by depth ranges
    
    Input:
    model: FESOM data array with 4 data layers, i.e. fesom_layered_sum as output of fesom_to_maredat_levels()
    figlabel: part of xlabel for plot
    savelabel: used to save the produced graph ('MarEDAT_FESOM_depths_'+label...)
    maredat_layered_sum: maredat data as array[4] layered into 0-5, 5-25, 25-100, 100m-bottom, sum over each depth layer
    lon_maredat
    lat_maredat
    mapproj: as ccrs format 
    savefig=False
    savepath
    '''
    
    if(mapproj == 'pc'): mapproj = ccrs.PlateCarree()
    elif(mapproj == 'rob'): mapproj = ccrs.Robinson()
    else: raise ValueError('Map projecction not supported')
        
    if(savepath == ''):
        if(savefig == True):
            raise ValueError('Input for saving graph insufficient')
    
    print(figlabel)
    
    fig, axes = plt.subplots(4,2, 
                             subplot_kw=dict(projection=mapproj),
                             gridspec_kw={'hspace': 0.001, 'wspace': 0.1},
                             figsize=(30,45))
    #fig.tight_layout()

    # 0-5 m maredat ---------------------------------------------------------------------------------------------
    m1 = axes[0,0]
    f1 = m1.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_sum[0]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m1)
    m1.set_title('Maredat 0-5 m', fontsize=20)

    # 0-5 m FESOM
    m1 = axes[0,1]
    f1 = m1.pcolormesh(lon_maredat, lat_maredat, np.log10(fesom_layered_sum[0]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m1)
    m1.set_title('FESOM 0-5 m', fontsize=20)

    # 5-25 m maredat ---------------------------------------------------------------------------------------------
    m2 = axes[1,0]
    f2 = m2.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_sum[1]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m2)
    m2.set_title('Maredat 5-25 m', fontsize=20)

    # 5-25 m FESOM
    m2 = axes[1,1]
    f2 = m2.pcolormesh(lon_maredat, lat_maredat, np.log10(fesom_layered_sum[1]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m2)
    m2.set_title('FESOM 5-25 m', fontsize=20)

    # 25-100 m maredat ---------------------------------------------------------------------------------------------
    m3 = axes[2,0]
    f3 = m3.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_sum[2]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m3)
    m3.set_title('Maredat 25-100 m', fontsize=20)

    # 25-100 m FESOM
    m3 = axes[2,1]
    f3 = m3.pcolormesh(lon_maredat, lat_maredat, np.log10(fesom_layered_sum[2]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m3)
    m3.set_title('FESOM 25-100 m', fontsize=20)

    # 100-bottom m maredat ---------------------------------------------------------------------------------------------
    m4 = axes[3,0]
    f4 = m4.pcolormesh(lon_maredat, lat_maredat, np.log10(maredat_layered_sum[2]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m4)
    m4.set_title('Maredat 100 m - bottom', fontsize=20)

    # 100-bottom m
    m4 = axes[3,1]
    f4 = m4.pcolormesh(lon_maredat, lat_maredat, np.log10(fesom_layered_sum[3]), 
                       transform = ccrs.PlateCarree(),
                        shading='auto', vmin=np.log10(0.001), vmax=np.log10(200), 
                        cmap=plt.cm.viridis)
    mygrid(m4)
    m4.set_title('FESOM 100 m - bottom', fontsize=20)

    # ---------------------------------------------------------------------------------------------
    # add one colorbar for all plots below figure
    cbar = fig.colorbar(f1,
                        ax = axes[3,:2], 
                        location ='bottom',
                        ticks=[np.log10(0.001),np.log10(0.01), np.log10(0.1),np.log10(0.5), np.log10(1),np.log10(10),np.log10(50),np.log10(100)],
                        fraction=0.1, pad=0.1) 
    cbar.ax.set_xticklabels(['0.001','0.01','0.1','0.5','1','10','50','100'], fontsize=20) 
    cbar.set_label(PF+'[mg C m$^{-2}$]', fontsize=20)
    #cbar.set_label('Log$_{10}$ Biomass [$\mu$g L $^{-1}$]', fontsize=20)

    if(savefig == True):
        fig.savefig(savepath+'Maredat_FESOM_maps_'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
    plt.show()

class plot_maps_all_pfts:
    '''
    class Bio_map_all(runname,resultpath,savepath,meshpath,first_year,last_year,
                 mapproj='pc',cmap,cmap_extension,savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='pc',
                 cmap='viridis',
                 cmap_extension='max',
                 savefig=False,
                 output=False,
                 plotting=True,
                 verbose=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.meshpath = meshpath
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.cmap = cmap
        self.cmap_extension = cmap_extension
        self.savefig = savefig
        self.verbose = verbose
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
            
            
        years = np.arange(self.fyear, self.lyear+1,1)
        num_plots = 3 # minimum number of PFTs for 2p1z1d
        
        # Load biomass data and compute layersum and volsum -------------------------------------------------------------------------------
            
        # Diatoms  
        DiaC = pf.get_data(resultpath, 'DiaC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
        DiaC = DiaC * 12.01
        DiaC_layersum = pf.layersum_data(DiaC,mesh,uplow=[0, 6250])
        DiaC_layersum = DiaC_layersum/1e3 # mg/m2 -> g/m2
        DiaC_volsum   = pf.volsum_data(DiaC,mesh,uplow=[0, 6250])
        DiaC_volsum   = DiaC_volsum/1e18 # mg -> Pg
        
        # Small phytoplankton
        SPC = pf.get_data(resultpath, 'PhyC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
        SPC = SPC * 12.01
        SPC_layersum = pf.layersum_data(SPC,mesh,uplow=[0, 6250])
        SPC_layersum = SPC_layersum/1e3 # mg/m2 -> g/m2
        SPC_volsum   = pf.volsum_data(SPC,mesh,uplow=[0, 6250])
        SPC_volsum   = SPC_volsum/1e18 # mg -> Pg
        
        # Coccolithophores
        cocco_path = Path(self.resultpath + '/CoccoC.fesom.'+str(years[0])+'.nc') 
        if cocco_path.is_file():
            num_plots = num_plots + 1
            CoccoC = pf.get_data(resultpath, 'CoccoC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
            CoccoC = CoccoC * 12.01
            CoccoC_layersum = pf.layersum_data(CoccoC,mesh,uplow=[0, 6250])
            CoccoC_layersum = CoccoC_layersum/1e3 # mg/m2 -> g/m2
            CoccoC_volsum   = pf.volsum_data(CoccoC,mesh,uplow=[0, 6250])
            CoccoC_volsum   = CoccoC_volsum/1e18 # mg -> Pg

        # Phaeocystis 
        phaeo_path = Path(self.resultpath + '/PhaeoC.fesom.'+str(years[0])+'.nc') 
        if phaeo_path.is_file():
            num_plots = num_plots + 1
            PhaeoC = pf.get_data(resultpath, 'PhaeoC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
            PhaeoC = PhaeoC * 12.01
            PhaeoC_layersum = pf.layersum_data(PhaeoC,mesh,uplow=[0, 6250])
            PhaeoC_layersum = PhaeoC_layersum/1e3 # mg/m2 -> g/m2
            PhaeoC_volsum   = pf.volsum_data(PhaeoC,mesh,uplow=[0, 6250])
            PhaeoC_volsum   = PhaeoC_volsum/1e18 # mg -> Pg

        # DiaH
        diah_path = Path(self.resultpath + '/DiaHC.fesom.'+str(years[0])+'.nc') 
        if diah_path.is_file():
            num_plots = num_plots + 1
            DiaHC = pf.get_data(resultpath, 'DiaHC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
            DiaHC = DiaHC * 12.01
            DiaHC_layersum = pf.layersum_data(DiaHC,mesh,uplow=[0, 6250])
            DiaHC_layersum = DiaHC_layersum/1e3 # mg/m2 -> g/m2
            DiaHC_volsum   = pf.volsum_data(DiaHC,mesh,uplow=[0, 6250])
            DiaHC_volsum   = DiaHC_volsum/1e18 # mg -> Pg

        
        # Microzooplankton
        micro_path = Path(self.resultpath + '/Zoo3C.fesom.'+str(years[0])+'.nc') 
        if micro_path.is_file():
            num_plots = num_plots + 1
            MicroC = pf.get_data(resultpath, 'Zoo3C', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
            MicroC = MicroC * 12.01
            MicroC_layersum = pf.layersum_data(MicroC,mesh,uplow=[0, 6250])
            MicroC_layersum = MicroC_layersum/1e3 # mg/m2 -> g/m2
            MicroC_volsum   = pf.volsum_data(MicroC,mesh,uplow=[0, 6250])
            MicroC_volsum   = MicroC_volsum/1e18 # mg -> Pg
        
        # Mesozooplankton
        MesoC = pf.get_data(resultpath, 'HetC', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
        MesoC = MesoC * 12.01
        MesoC_layersum = pf.layersum_data(MesoC,mesh,uplow=[0, 6250])
        MesoC_layersum = MesoC_layersum/1e3 # mg/m2 -> g/m2
        MesoC_volsum   = pf.volsum_data(MesoC,mesh,uplow=[0, 6250])
        MesoC_volsum   = MesoC_volsum/1e18 # mg -> Pg

        # Macrozooplankton
        macro_path = Path(self.resultpath + '/Zoo2C.fesom.'+str(years[0])+'.nc') 
        if macro_path.is_file():
            num_plots = num_plots + 1
            MacroC = pf.get_data(resultpath, 'Zoo2C', years, mesh, runid=self.runname, how="mean", compute=True, silent=True)
            MacroC = MacroC * 12.01
            MacroC_layersum = pf.layersum_data(MacroC,mesh,uplow=[0, 6250])
            MacroC_layersum = MacroC_layersum/1e3 # mg/m2 -> g/m2
            MacroC_volsum   = pf.volsum_data(MacroC,mesh,uplow=[0, 6250])
            MacroC_volsum   = MacroC_volsum/1e18 # mg -> Pg
            
            
            
        # Plot ---------------------------------------------------------------------------------------------------------
        
        levels_diatoms            = np.arange(0,6,0.3)
        levels_diah               = np.arange(0,6,0.3)
        levels_smallphytoplankton = np.arange(0,6,0.3)
        levels_coccolithophores   = np.arange(0,4,0.4)
        levels_phaeocystis        = np.arange(0,4,0.4)
        levels_microzooplankton   = np.arange(0,1,0.1)
        levels_mesozooplankton    = np.arange(0,0.5,0.05)
        levels_macrozooplankton   = np.arange(0,0.5,0.05)
        unit_phytoplankton        = 'Biomass [g m$^{-2}$]'
        unit_zooplankton          = 'Biomass [g m$^{-2}$]'
        cmap_extension            = 'max'
        cmap                      = self.cmap
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        mapproj = pf.get_proj(self.mapproj)
        
        # Automatically calculate rows (max 2 columns)
        cols = 2
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.1},
                             subplot_kw=dict(projection=mapproj))
        axes = axes.flatten()

        # Diatoms
        f1 = pf.subplot(mesh, fig, axes[0], [DiaC_layersum[:]], levels = levels_diatoms,
                        units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
        axes[0].set_title('Diatoms', fontsize=12)

        pos = axes[0].get_position()
        cbar1_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
        #cbar1_ax = fig.add_axes([0.135, 0.715, 0.35, 0.01])  # left, bottom, width, height
        cbar1 = fig.colorbar(f1,
                        cax = cbar1_ax, 
                        orientation = 'horizontal',
                        extend = 'max',
                        #ticks = [0,4,8,12,16,20,24,28,32,36,40]
                            ) 
        cbar1.ax.tick_params(labelsize=10)
        cbar1.set_label(unit_phytoplankton, fontsize=10)
        
        DiaC_volsum_text = str(round(DiaC_volsum[0],2))
        DiaC_text_all = r'$\Sigma$ '+DiaC_volsum_text+' Pg C'
        axes[0].text(0.7,0.03,DiaC_text_all, transform=axes[0].transAxes, color='white', fontsize=10, fontweight='bold')
        
        
        # Small phytoplankton
        f2 = pf.subplot(mesh, fig, axes[1], [SPC_layersum[:]], levels = levels_smallphytoplankton,
                        units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
        axes[1].set_title('Small phytoplankton', fontsize=12)

        pos = axes[1].get_position()
        cbar2_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
        cbar2 = fig.colorbar(f2,
                        cax = cbar2_ax, 
                        orientation = 'horizontal',
                        extend = 'max',
                        #ticks = [0,4,8,12,16,20,24,28,32,36]
                            ) 
        cbar2.ax.tick_params(labelsize=10)
        cbar2.set_label(unit_phytoplankton, fontsize=10)
        
        SPC_volsum_text = str(round(SPC_volsum[0],2))
        SPC_text_all = r'$\Sigma$ '+SPC_volsum_text+' Pg C'
        axes[1].text(0.7,0.03,SPC_text_all, transform=axes[1].transAxes, color='white', fontsize=10, fontweight='bold')
        num_axes = 1
        
        # Coccolithophores
        if cocco_path.is_file():
            num_axes = num_axes + 1
            f3 = pf.subplot(mesh, fig, axes[num_axes], [CoccoC_layersum[:]], levels = levels_coccolithophores,
                            units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
            axes[num_axes].set_title('Coccolithophores', fontsize=12)


            pos = axes[num_axes].get_position()
            cbar3_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])  
            cbar3 = fig.colorbar(f3,
                            cax = cbar3_ax, 
                            orientation = 'horizontal',
                            extend = 'max',
                            #ticks = [0,4,8,12,16,20,24,28,32,36]
                                ) 
            cbar3.ax.tick_params(labelsize=10)
            cbar3.set_label(unit_phytoplankton, fontsize=10)
            
            CoccoC_volsum_text = str(round(CoccoC_volsum[0],2))
            CoccoC_text_all = r'$\Sigma$ '+CoccoC_volsum_text+' Pg C'
            axes[num_axes].text(0.7,0.03,CoccoC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')


        # Phaeocystis
        if phaeo_path.is_file():
            num_axes = num_axes + 1
            f7 = pf.subplot(mesh, fig, axes[num_axes], [PhaeoC_layersum[:]], levels = levels_phaeocystis,
                            units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
            axes[num_axes].set_title('Phaeocystis', fontsize=12)
            
            pos = axes[num_axes].get_position()
            cbar7_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
            cbar7 = fig.colorbar(f7,
                            cax = cbar7_ax, 
                            orientation = 'horizontal',
                            extend = 'max',
                            #ticks = [0,4,8,12,16,20,24,28,32,36]
                                ) 
            cbar7.ax.tick_params(labelsize=10)
            cbar7.set_label(unit_phytoplankton, fontsize=10)
            
            PhaeoC_volsum_text = str(round(PhaeoC_volsum[0],2))
            PhaeoC_text_all = r'$\Sigma$ '+PhaeoC_volsum_text+' Pg C'
            axes[num_axes].text(0.7,0.03,PhaeoC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')

        # DiaH
        if diah_path.is_file():
            num_axes = num_axes + 1
            f8 = pf.subplot(mesh, fig, axes[num_axes], [DiaHC_layersum[:]], levels = levels_diah,
                            units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
            axes[num_axes].set_title('Highly silicyfied Diatoms', fontsize=12)
            
            pos = axes[num_axes].get_position()
            cbar8_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
            cbar8 = fig.colorbar(f8,
                            cax = cbar8_ax, 
                            orientation = 'horizontal',
                            extend = 'max',
                            #ticks = [0,4,8,12,16,20,24,28,32,36]
                                ) 
            cbar8.ax.tick_params(labelsize=10)
            cbar8.set_label(unit_phytoplankton, fontsize=10)
            
            DiaHC_volsum_text = str(round(DiaHC_volsum[0],2))
            DiaHC_text_all = r'$\Sigma$ '+DiaHC_volsum_text+' Pg C'
            axes[num_axes].text(0.7,0.03,DiaHC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')
         
                
        # Microzooplankton
        if micro_path.is_file():
            num_axes = num_axes + 1
            f4 = pf.subplot(mesh, fig, axes[num_axes], [MicroC_layersum[:]], levels = levels_microzooplankton,
                            units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
            axes[num_axes].set_title('Microzooplankton', fontsize=12)
            
            pos = axes[num_axes].get_position()
            cbar4_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
            cbar4 = fig.colorbar(f4,
                            cax = cbar4_ax, 
                            orientation = 'horizontal',
                            extend = 'max',
                            #ticks = [0,4,8,12,16,20,24,28,32,36]
                                ) 
            cbar4.ax.tick_params(labelsize=10)
            cbar4.set_label(unit_zooplankton, fontsize=10)
            
            MicroC_volsum_text = str(round(MicroC_volsum[0],2))
            MicroC_text_all = r'$\Sigma$ '+MicroC_volsum_text+' Pg C'
            axes[num_axes].text(0.7,0.03,MicroC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')
            

            
        num_axes = num_axes + 1
        f5 = pf.subplot(mesh, fig, axes[num_axes], [MesoC_layersum[:]], levels = levels_mesozooplankton,
                        units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
        axes[num_axes].set_title('Mesozooplankton', fontsize=12)

        pos = axes[num_axes].get_position()
        cbar5_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
        cbar5 = fig.colorbar(f5,
                        cax = cbar5_ax, 
                        orientation = 'horizontal',
                        extend = 'max',
                        #ticks = [0,4,8,12,16,20,24,28,32,36]
                            ) 
        cbar5.ax.tick_params(labelsize=10)
        cbar5.set_label(unit_zooplankton, fontsize=10)
        
        MesoC_volsum_text = str(round(MesoC_volsum[0],2))
        MesoC_text_all = r'$\Sigma$ '+MesoC_volsum_text+' Pg C'
        axes[num_axes].text(0.7,0.03,MesoC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')

        
        # Macrozooplankton
        if macro_path.is_file():
            num_axes = num_axes + 1
            f6 = pf.subplot(mesh, fig, axes[num_axes], [MacroC_layersum[:]], levels = levels_macrozooplankton,
                            units = unit_phytoplankton, mapproj = mapproj, cmap = cmap, cmap_extension = cmap_extension, box = box)
            axes[num_axes].set_title('Macrozooplankton', fontsize=12)

            pos = axes[num_axes].get_position()
            cbar6_ax = fig.add_axes([pos.x0, pos.y0-0.02, 0.35, 0.01])
            cbar6 = fig.colorbar(f6,
                            cax = cbar6_ax, 
                            orientation = 'horizontal',
                            extend = 'max',
                            #ticks = [0,4,8,12,16,20,24,28,32,36]
                                ) 
            cbar6.ax.tick_params(labelsize=10)
            cbar6.set_label(unit_zooplankton, fontsize=10)
            
            MacroC_volsum_text = str(round(MacroC_volsum[0],2))
            MacroC_text_all = r'$\Sigma$ '+MacroC_volsum_text+' Pg C'
            axes[num_axes].text(0.7,0.03,MacroC_text_all, transform=axes[num_axes].transAxes, color='white', fontsize=10, fontweight='bold')
        
        
        if(self.savefig == True):
            plt.savefig(self.savepath+self.runname+'_Biomass_maps_'+'_'+str(self.fyear)+'to'+str(self.lyear)+'.png', 
                    dpi = 300, bbox_inches='tight')

        plt.show()
        

           
class plot_maps_alk:
    '''
    class Alkcomp(resultpath,savepath,meshpath,ncpath,first_year,last_year,
                 GLODAPvar='TAlk',mapproj='pc',savefig=False,layerwise=False,runname)
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileAlk,
                 first_year=first_year,
                 last_year=last_year,
                 GLODAPvar='TAlk_mmol',
                 mapproj='rob',
                 cmap='viridis',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='both',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 get_overview = False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.GLODAPvar = GLODAPvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.resultpath+'/'+self.runname+'.mesh.diag.nc'
        labelfesom = 'FESOM'
        unitfesom = 'Alkalinity [mmol m$^{-3}$]' 

        # load FESOM data -------------------------------------------------------------------------------------
        Alkfesom = pf.get_data(resultpath, "Alk", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load raw Glodap data -------------------------------------------------------------------------------------
        
        unitglodap = 'Alkalinity [mmol m$^{-3}$]'
        labelglodap = 'GLODAP'

        
        # interpolate Glodap data -------------------------------------------------------------------------------------
        GLODAP_input = load_glodap_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.GLODAPvar, get_overview=False)
        glodap_int = GLODAP_input.glodap_int 
        
        # apply sea mask to Glodap as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        glodap_int_ma = np.copy(glodap_int)
        glodap_int_ma[Alkfesom == 0] = 0

        # plot Glodap and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):  
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(GLODAP_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]


                Alkfesom_mean = pf.layermean_data(Alkfesom, mesh, uplow = uplow)
                glodap_int_ma_mean = pf.layermean_data(glodap_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(Alkfesom_mean),np.nanstd(Alkfesom_mean),np.nanmin(Alkfesom_mean),np.nanmax(Alkfesom_mean),
                        np.nanmean(glodap_int_ma_mean),np.nanstd(glodap_int_ma_mean),np.nanmin(glodap_int_ma_mean),np.nanmax(glodap_int_ma_mean)))
                

                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = 20
                    levels = np.arange(2100,2520,20)
                    f1 = pf.subplot(mesh, fig, m1, [Alkfesom_mean],
                                levels = levels,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [glodap_int_ma_mean], 
                                levels = levels,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-100,110,10)
                    f3 = pf.subplot(mesh, fig, m3, [Alkfesom_mean-glodap_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - GLODAP\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    #fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Alk_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'Alk_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
  
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(glodap_int)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(Alkfesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Taylor Diagram for Alkalinity'
                    plt_taylor_comp(glodap_int_ma,Alkfesom,mask=True,title=title, depth_array=depth_array, mesh=mesh)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'Alk_GLODAP_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                        dpi = 300, bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(GLODAP_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            Alkfesom_mean = pf.layermean_data(Alkfesom, mesh, uplow = uplow)
            glodap_int_ma_mean = pf.layermean_data(glodap_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPISCES mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(Alkfesom_mean),np.nanstd(Alkfesom_mean),np.nanmin(Alkfesom_mean),np.nanmax(Alkfesom_mean),
                np.nanmean(glodap_int_ma_mean),np.nanstd(glodap_int_ma_mean),np.nanmin(glodap_int_ma_mean),np.nanmax(glodap_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(2100,2520,20)
                f1 = pf.subplot(mesh, fig, m1, [Alkfesom_mean],
                            levels = levels,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [glodap_int_ma_mean], 
                            levels = levels,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelglodap+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-100,110,10)
                f3 = pf.subplot(mesh, fig, m3, [Alkfesom_mean-glodap_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - GLODAP'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                
                #fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'Alk_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'Alk_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(glodap_int_ma_mean)): print('WARNING: The interpolated Glodap field contains NaNs')
                if np.isnan(np.min(Alkfesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs')

                title = 'Taylor Diagram for Alk \n(mean over depth, max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(glodap_int_ma_mean,Alkfesom_mean,mask=True,title=title)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'Alk_Glodap_Taylor'+'_'+str(years[0])+'to'+str(years[1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                plt.show(block=False)  
                
            if output:
                    self.fesom = Alkfesom_mean
                    self.glodap   = glodap_int_ma_mean   


class plot_maps_dic:
    '''
    class DICcomp
    
    Compare DIC to Glodap
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDIC,
                 first_year=first_year,
                 last_year=last_year,
                 GLODAPvar='TCO2_mmol',
                 mapproj='pc',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=False,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 get_overview = False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.GLODAPvar = GLODAPvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        #print('Processing {0}'.format(self.resultpath))
        self.mapproj = pf.get_proj(self.mapproj)
        years = np.arange(self.fyear, self.lyear+1,1)

        # check variables
        #NCfesom = self.resultpath + '/DFe.'+self.runname+'.'+str(self.fyear)+'.nc'
        #!ncdump -h $NCfesom

        labelfesom = 'FESOM'
        unitfesom = 'DIC [mmol C m$^{-3}$]' # equals to mumol/L

        # load FESOM data -------------------------------------------------------------------------------------
        DICfesom = pf.get_data(resultpath, "DIC", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent = True)
        
        unitglodap = 'DIC [mmol C m$^{-3}$]'
        labelglodap = 'GLODAP'
        
        # interpolate Glodap data -------------------------------------------------------------------------------------
        GLODAP_input = load_glodap_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.GLODAPvar, get_overview=False)
        glodap_int = GLODAP_input.glodap_int   

        # apply sea mask to Glodap as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        glodap_int_ma = np.copy(glodap_int)
        glodap_int_ma[DICfesom == 0] = 0

        # plot Glodap and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(GLODAP_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]


                DICfesom_mean = pf.layermean_data(DICfesom, mesh, uplow = uplow)
                glodap_int_ma_mean = pf.layermean_data(glodap_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(DICfesom_mean),np.nanstd(DICfesom_mean),np.nanmin(DICfesom_mean),np.nanmax(DICfesom_mean),
                        np.nanmean(glodap_int_ma_mean),np.nanstd(glodap_int_ma_mean),np.nanmin(glodap_int_ma_mean),np.nanmax(glodap_int_ma_mean)))
                
                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(1600,2300,10)
                    f1 = pf.subplot(mesh, fig, m1, [DICfesom_mean],
                                levels = levels,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [glodap_int_ma_mean], 
                                levels = levels,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelglodap+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-150,160,10)
                    f3 = pf.subplot(mesh, fig, m3, [DICfesom_mean-glodap_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitglodap, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - GLODAP ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )
                    
                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')

                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                    
                    
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(glodap_int)): print('WARNING: The interpolated GLODAP field contains NaNs at depth')
                    if np.isnan(np.min(DICfesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Normalized Taylor Diagram for DIC'
                    plt_taylor_comp(dic_int_ma,DICfesom,mask=True,title=title, depth_array=depth_array, mesh=mesh)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.png', 
                                        dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.pdf', 
                                        bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    #self.fesom.{str(d)} = DICfesom_mean
                    #self.glodap.{str(d)}   = glodap_int_ma_mean
                    print('returning layerwise output not supported yet')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(GLODAP_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            DICfesom_mean = pf.layermean_data(DICfesom, mesh, uplow = uplow)
            glodap_int_ma_mean = pf.layermean_data(glodap_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPISCES mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(DICfesom_mean),np.nanstd(DICfesom_mean),np.nanmin(DICfesom_mean),np.nanmax(DICfesom_mean),
                np.nanmean(glodap_int_ma_mean),np.nanstd(glodap_int_ma_mean),np.nanmin(glodap_int_ma_mean),np.nanmax(glodap_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(1600,2300,10)
                f1 = pf.subplot(mesh, fig, m1, [DICfesom_mean],
                            levels = levels,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [glodap_int_ma_mean], 
                            levels = levels,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelglodap+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-150,160,10)
                f3 = pf.subplot(mesh, fig, m3, [DICfesom_mean-glodap_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitglodap, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - GLODAP'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DIC_GLODAP'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(glodap_int_ma_mean)): print('WARNING: The interpolated GLODAP field contains NaNs at depth')
                if np.isnan(np.min(DICfesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')


                title = 'Taylor Diagram for DIC \n(mean over depth, max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(glodap_int_ma_mean,DICfesom_mean,mask=True,title=title)
                
                
                # fig export  
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'DIC_Glodap_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DIC_Glodap_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if output:
                    self.fesom = DICfesom_mean
                    self.glodap   = glodap_int_ma_mean


class plot_maps_din:
    '''
    class DINcomp
    
    compare FESOM-REcoM DIN data to World Ocean Atlas 
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDIN,
                 first_year=first_year,
                 last_year=last_year,
                 WOAvar='n_an',
                 mapproj='rob',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 get_overview = False,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.WOAvar = WOAvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.get_overview = get_overview
        self.verbose = verbose
        self.plotting = plotting
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        years = np.arange(self.fyear, self.lyear+1,1)


        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'DIN [mmol m$^{-3}$]' # equals to mumol/L

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(self.resultpath, "DIN", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load WOA data  -------------------------------------------------------------------------------------
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.WOAvar, get_overview=False)
        woa_int = WOA_input.woa_int    
        
        labelwoa = 'WOA'
        unitwoa = 'DIN [mmol m$^{-3}$]'

        # apply sea mask to WOA as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        woa_int_ma = np.copy(woa_int)
        woa_int_ma[fesom == 0] = 0

        # plot WOA and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)

            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))

                if (self.plotting):
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(0,25,1)
                    f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-15,16,2)
                    f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - WOA ({0}-{1} m)'.format(uplow[0],uplow[1]),
                               )

                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')

                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                    
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(din_int)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(DINfesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Normalized Taylor Diagram for DIN'
                    plt_taylor_comp(din_int_ma,DINfesom,mask=True,title=title, depth_array=depth_array, mesh=mesh)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.png', 
                                        dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.pdf', 
                                        bbox_inches='tight')
                    plt.show(block=False) 
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(0,25,1)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0], uplow[1]),
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0], uplow[1]),
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-15,16,1)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - WOA'+' ({0}-{1} m)'.format(uplow[0], uplow[1]),
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(woa_int_ma_mean)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for DIN \n(mean over depth, max = {0}-{1}m)'.format(uplow[0], uplow[1]),
                plot_taylor_norm(woa_int_ma_mean,fesom_mean,mask=True,title=title,verbose=True)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DIN_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if output:
                    self.fesom = fesom_mean
                    self.woa   = woa_int_ma_mean


class plot_maps_do2:
    '''
    class DO2comp
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDO2,
                 first_year=first_year,
                 last_year=last_year,
                 WOAvar='oxygen_mmol',
                 mapproj='rob',
                 cmap = cmo.cm.oxy,
                 savefig=False,
                 layerwise=False,depth_array=[],
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.WOAvar = WOAvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        
        # load FESOM data -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'DO2 [mmol O2 m$^{-3}$]' 
        
        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "O2", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)


        # load WOA data  -------------------------------------------------------------------------------------
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.WOAvar, get_overview=False)
        WOA_int = WOA_input.woa_int     
        
        labelwoa = 'WOA'
        unitwoa = 'DO2 [mmol O2 m$^{-3}$]' #??? ## o_an:units = "micromoles_per_kilogram"

        # apply sea mask to WOA as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        woa_int_ma = np.copy(WOA_int)
        woa_int_ma[fesom == 0] = 0

        # plot WOA and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)

            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))

                if (self.plotting):
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    m1 = axes['A']
                    levels = np.arange(0,341,10)
                    f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-60,61,5)
                    f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - WOA ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')

                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]),  
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(O2_int_ma)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(DO2fesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Normalized Taylor Diagram for DO2'
                    plot_taylor_comp(O2_int_ma,DO2fesom,mask=True,title=title, depth_array=depth_array, mesh=mesh)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                        dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                        bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')

        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(0,341,10)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-60,61,5)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - WOA'+' ({0}-{1} m)'.format(uplow[0],uplow[1]),
                            box=box,
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if Taylor:
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(woa_int_ma_mean)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for DO$_2$ \n(mean over depth, max = {0}-{1}m)'.format(uplow[0],uplow[1]),
                plot_taylor_norm(woa_int_ma_mean,fesom_mean,mask=True,title=title,verbose = True)
                
                
                # Taylor fig export  
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DO2_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)  
                
            if output:
                    self.fesom = fesom_mean
                    self.woa   = woa_int_ma_mean
        

class plot_maps_chl_arctic:
    '''
    class ChlArctic(runname,resultpath,savepath,meshpath,dataset, observation_file,first_year,last_year,
                 savefig=False, verbose=False, output=False, plotting=True, Taylor=True)
                 
    This routine compare FESOM Chlorophyll outputs with state-of-the-art Arctic adapted satellite products:
    1) From OC-CCI: OCEANCOLOUR_ARC_BGC_L4_MY_009_124, https://doi.org/10.48670/moi-00293 => select dataset = 'OCCCI'
    2) From Stanford: Lewis and Arrigo, 2020. => select dataset = 'Lewis'
    
    self.ChlAfesom_surf_interp contains 2D dataset of 1x1 interpolated nanophytoplankton Chl.a
    self.DiaChlfesom_surf_interp contains 2D dataset of 1x1 interpolated diatom Chl.a
    self.Chl_total_interp = Chl_total contains 2D dataset of 1x1 interpolated sum of Chl.a
    self.unitfesom contains str of FESOM Chl.a unit
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncfile=ncfileChlArcLewis,
                 first_year=first_year,
                 last_year=last_year,
                 savefig=False,output=False,
                 plotting=True,verbose=False,
                 Taylor=True,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.ncfile = ncfile
        self.verbose = verbose
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor
        
        mapproj = pf.get_proj('np')

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))

        # resolution can be increased below if necessary
        londic, latdic = np.meshgrid(np.arange(-180,181,1), np.arange(65,90.5,0.5))
            
        # load OCCCI CHl.a data -------------------------------------------------------------------------------------
        Chlsat = xr.open_dataset(ncfile, decode_times=True)
        if 'LEWIS' in ncfile:
            tag = 'LEWIS'
            print('Lewis et al. dataset selected')
            label = 'Lewis et al. [2003-2021]'
            print(' !!! Only MODIS Satellite for the 2003-2021 period !!!')
            Chlsat = Chlsat.where(Chlsat.month.isin([5,6,7,8,9]), drop=True).mean(dim='month').compute()
            # interpolate Satellite Chl to the same regular grid -------------------------------------------------------------------------------------
            Chlsat_interp = griddata((Chlsat.longitude.data.ravel(), Chlsat.latitude.data.ravel()),
                                Chlsat.TChl.data.ravel(), (londic, latdic), method = 'nearest')
        elif 'OCCCI' in ncfile:
            tag = 'OCCCI'
            print('OCCCI. dataset selected')    
            label = 'OC-CCI [2000-2019]'
            print(' !!! Only Satellite data for the 2000-2019 period !!!')
            Chlsat = Chlsat.where(Chlsat.time.isin([5,6,7,8,9]), drop=True).mean(dim='time').compute()
            # interpolate Satellite Chl to the same regular grid -------------------------------------------------------------------------------------
            Chlsat_interp = griddata((Chlsat.longitude.data.ravel(), Chlsat.latitude.data.ravel()),
                                Chlsat.CHL.data.ravel(), (londic, latdic), method = 'nearest')
        else:
            print('wrong dataset selected, please select either {Lewis} or {OCCCI}')
                                   
        
        box=[-180, 180, 65, 90]
        unit = 'Chl.a [mg Chl m$^{-3}$]'          
        
        # load FESOM Chl.a data -------------------------------------------------------------------------------------        
        years = np.arange(self.fyear, self.lyear+1,1)
        fesom_label = 'FESOM-REcoM Chl.a {0}-{1}'.format(self.fyear,self.lyear)   
        
        PhyChlfesom = pf.get_data(self.resultpath, "PhyChl", years, mesh, 
                               how=None, compute=False, runid=self.runname, silent=True)
        DiaChlfesom = pf.get_data(self.resultpath, "DiaChl", years, mesh, 
                               how=None, compute=False, runid=self.runname, silent=True)

        PhyChlfesom = PhyChlfesom.where(PhyChlfesom.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='YS').mean(dim='time').compute()
        DiaChlfesom = DiaChlfesom.where(DiaChlfesom.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='YS').mean(dim='time').compute() 
        
        # load FESOM Cocco/Phaeo Chl.a data -------------------------------------------------------------------------------------------
        from pathlib import Path
        cocco_path = Path(self.resultpath + '/CoccoChl.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/PhaeoChl.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        diah_path = Path(self.resultpath + '/DiaHChl.fesom.'+str(years[0])+'.nc') 
        
        
        if cocco_path.is_file():
            CoccoChlfesom = pf.get_data(self.resultpath, "CoccoChl", years, mesh, 
                               how=None, compute=False, runid=self.runname, silent=True)
            CoccoChlfesom = CoccoChlfesom.where(CoccoChlfesom.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='YS').mean(dim='time').compute()

            if phaeo_path.is_file():
                print('4-phytoplankton model is used')
                PhaeoChlfesom = pf.get_data(self.resultpath, "PhaeoChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                PhaeoChlfesom = PhaeoChlfesom.where(PhaeoChlfesom.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='YS').mean(dim='time').compute()
                Chlfesom = np.nanmean(PhyChlfesom[:,:,0] + DiaChlfesom[:,:,0] + CoccoChlfesom[:,:,0] + PhaeoChlfesom[:,:,0], axis =0) # select only surface
                if diah_path.is_file():
                    print('5-phytoplankton model is used')
                    DiaHChlfesom = pf.get_data(self.resultpath, "DiaHChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    DiaHChlfesom = DiaHChlfesom.where(DiaHChlfesom.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='YS').mean(dim='time').compute()
                    Chlfesom = np.nanmean(DiaHChlfesom[:,:,0] + PhyChlfesom[:,:,0] + DiaChlfesom[:,:,0] + CoccoChlfesom[:,:,0] + PhaeoChlfesom[:,:,0], axis =0) # select only surface
            else:
                print('3-phytoplankton model is used')
                Chlfesom = np.nanmean(PhyChlfesom[:,:,0] + DiaChlfesom[:,:,0] + CoccoChlfesom[:,:,0], axis =0) # select only surface

        
        else:
            print('2-phytoplankton model is used')
            Chlfesom = np.nanmean(PhyChlfesom[:,:,0] + DiaChlfesom[:,:,0], axis =0) # select only surface
            
        # interpolate FESOM CHl.a to regular -------------------------------------------------------------------------------------
        Chlfesom_interp = pf.fesom2regular(
                data = Chlfesom,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        Chlfesom_interp_log10 = np.log10(Chlfesom_interp)

        # apply sea mask to OCCCI as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        Chlsat_ma = np.copy(Chlsat_interp)
        Chlsat_ma[~np.isfinite(Chlfesom_interp)] = np.nan
        Chlsat_ma_log10 = np.log10(Chlsat_ma)
        
        # check CHl.a data -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nChl.a\nOCCCI min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f}'.format(
                    np.nanmin(Chlfesom_interp),np.nanmax(Chlfesom_interp),
                    np.nanmin(Chlsat_ma),np.nanmax(Chlsat_ma)))

            print('\nlog10(Chl.a)\nOCCCI min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f}'.format(
                    np.nanmin(Chlfesom_interp_log10),np.nanmax(Chlfesom_interp_log10),
                    np.nanmin(Chlsat_ma_log10),np.nanmax(Chlsat_ma_log10)))
        
        if(self.plotting):
            # plot each CHl.a dataset -------------------------------------------------------------------------------------        
            
            levels = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                                   0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                                   1,2,3,4,5,7])
            ticks = [0,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1,3,5,7]
            ticks_label = ['0','0.01','0.03','0.05','0.07','0.1','0.3','0.5','0.7','1','3','5','7'] # +7 ?

            def mygrid(m):
                m.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='gray')
                
            fig, axes = plt.subplots(1,3, 
                                     subplot_kw=dict(projection=mapproj),
                                     gridspec_kw={'hspace': 0.01, 'wspace': 0.1},
                                     figsize=(20,7), constrained_layout=False)     

            # REcoM
            m1 = axes[0]
            f1 = m1.pcolormesh(londic, latdic, Chlfesom_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
            mygrid(m1)
            m1.set_title(fesom_label, fontsize=16)


            # Satellite
            m2 = axes[1]
            f2 = m2.pcolormesh(londic, latdic, Chlsat_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
            mygrid(m2)
            m2.set_title(label, fontsize=16)

            # add one colorbar for first row plots below figure
            cbar = fig.colorbar(f1,
                                ax = axes[:2], 
                                location ='bottom',
                                extend = 'max',
                                ticks = ticks,
                                fraction=0.046, pad=0.04) 
            #cbar.ax.tick_params(labelsize=14)
            cbar.ax.set_xticklabels(ticks_label, fontsize=16) 
            cbar.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
                
            # REcoM - Satellite
            levels_diff = np.arange(-3,3,0.125)
            m3 = axes[2]
            f3 = m3.pcolormesh(londic, latdic, Chlfesom_interp - Chlsat_ma, 
                                   transform = ccrs.PlateCarree(),cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256))
            mygrid(m3)
            m3.set_title('FESOM-REcoM - '+label, fontsize=16)

            # add one colorbar for difference plot below figure
            cbar = fig.colorbar(f3,
                            ax = axes[2], 
                            orientation = 'horizontal',
                            #location ='bottom',
                            ticks = [-3,-2,-1,0,1,2,3],
                            extend = 'both',
                            fraction=0.046, pad=0.04) 
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
                
            m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
            m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
            m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                    
            m1.set_extent(box, ccrs.PlateCarree())
            m2.set_extent(box, ccrs.PlateCarree())
            m3.set_extent(box, ccrs.PlateCarree())

            
            #plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'ArcChla_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'ArcChla_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  

        if(self.Taylor):
            # statistics  -------------------------------------------------------------------------------------            
            # preparation of datasets
            if np.isnan(np.min(Chlsat_ma_log10)): print('WARNING: Satellite field contains NaNs')
            if np.isnan(np.min(Chlfesom_interp_log10)): print('WARNING: FESOM field contains NaNs')

            # get statistics only from valid OCCCI gridpoints 
            ind_stat = np.where(np.isfinite(Chlsat_ma_log10))

            title = 'log10 surface Chlorophyll'
            print('\nStatistics for '+title)
            plot_taylor_norm(Chlsat_ma_log10[ind_stat],Chlfesom_interp_log10[ind_stat],
                                    mask=True,title=title)

            
            #plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'ArcChla_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'_Taylor.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'ArcChla_'+tag+'_'+str(years[0])+'to'+str(years[-1])+'_Taylor.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  
        
        if(self.output):
            self.lon = londic
            self.lat = latdic
            self.chl_oc = Chlsat_ma
            self.chl_fesom = Chlfesom_interp
            self.unit = unitfesom


class plot_maps_chl_global:
    '''
    class Chlsurf_OCCCI_comp(runname,resultpath,savepath,meshpath,matfileChlsurf,first_year,last_year,
                 mapproj='pc',savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
                 
    
    self.ChlAfesom_surf_interp contains 2D dataset of 1x1 interpolated nanophytoplankton Chl.a
    self.DiaChlfesom_surf_interp contains 2D dataset of 1x1 interpolated diatom Chl.a
    self.Chl_total_interp = Chl_total contains 2D dataset of 1x1 interpolated sum of Chl.a
    self.unitfesom contains str of FESOM Chl.a unit
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 matfileChlsurf=matfileChlGloOCCCI,
                 first_year=first_year,
                 last_year=last_year,
                 mapproj='rob',runname='fesom',
                 savefig=False,output=False,plotting=True,verbose=False,Taylor=True):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.matfileChlsurf=matfileChlsurf
        self.verbose = verbose
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)

        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
        
        # load OCCCI CHl.a data -------------------------------------------------------------------------------------
        matChl = spio.loadmat(self.matfileChlsurf, squeeze_me=True)
        
        lat         = np.arange(-89.5,90.,1.)
        lon         = np.arange(-179.5,180.,1.)
        latdic, londic = np.meshgrid(lat, lon)
        
        #annualchl   = np.log10(matChl['x'])
        OCCCIchla = matChl['x']
        OCCCIchla_log10 = np.log10(OCCCIchla)
        
        OCCCIlabel = 'OC-CCI'
        OCCCIunit = 'Chl.a [mg Chl m$^{-3}$]'          
        print(' !!! Satellite data only for 2012-2015 period !!!')
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(self.meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        lon_fesom = mesh.x2
        lat_fesom = mesh.y2        
        
        # load FESOM Nanophyto Chl.a data -------------------------------------------------------------------------------------        
        
        #ncFESOMChl = self.resultpath + '/PhyChl.fesom.1948.nc'  # Miriam: not needed?
        #!ncdump -h $ncFESOMChl
        
        PhyChlfesom = pf.get_data(self.resultpath, "PhyChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
        
        PhyChlfesom_surf = PhyChlfesom[:,0]
        
        labelfesomNano = 'FESOM-REcoM Nanophyto Chl.a {0}-{1}'.format(self.fyear,self.lyear)        
        #unitfesomNano = 'Chl.a [mmol m$^{-3}$]' 

        # load FESOM Diatom Chl.a data -------------------------------------------------------------------------------------
        
        #ncFESOMDiaChl = self.resultpath + '/DiaChl.fesom.1948.nc'   # Miriam: not needed?
        #!ncdump -h $ncFESOMDiaChl
        
        DiaChlfesom = pf.get_data(self.resultpath, "DiaChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
        
        labelfesomDia = 'FESOM-REcoM Diatom Chl.a {0}-{1}'.format(self.fyear,self.lyear)
        
        DiaChlfesom_surf = DiaChlfesom[:,0]
    
        unitfesom = 'Chl.a [mg m$^{-3}$]'
        labelfesom = 'FESOM-REcoM ({0}-{1})'.format(self.fyear,self.lyear)
        
        # load FESOM Cocco Chl.a data -------------------------------------------------------------------------------------------
        
        from pathlib import Path
        
        cocco_path = Path(self.resultpath + '/CoccoChl.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/PhaeoChl.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        diah_path = Path(self.resultpath + '/DiaHChl.fesom.'+str(years[0])+'.nc') 
        
        if cocco_path.is_file():
            
            CoccoChlfesom = pf.get_data(self.resultpath, "CoccoChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
            
            labelfesomCocco = 'FESOM-REcoM Cocco Chl.a {0}-{1}'.format(self.fyear,self.lyear)
            
            CoccoChlfesom_surf = CoccoChlfesom[:,0]

            # load FESOM Phaeo Chl.a data ---------------------------------------------------------------------------------------

            if phaeo_path.is_file():
            
                PhaeoChlfesom = pf.get_data(self.resultpath, "PhaeoChl", years, mesh, 
                                   how="mean", compute=True, runid=self.runname, silent=True)
                
                labelfesomPhaeo = 'FESOM-REcoM Phaeo Chl.a {0}-{1}'.format(self.fyear,self.lyear)
                
                PhaeoChlfesom_surf = PhaeoChlfesom[:,0]
                
                print('4-phytoplankton model is used')
                
                if diah_path.is_file():
            
                    DiaHChlfesom = pf.get_data(self.resultpath, "DiaHChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    labelfesomDiaH = 'FESOM-REcoM DiaH Chl.a {0}-{1}'.format(self.fyear,self.lyear)
                    
                    DiaHChlfesom_surf = DiaHChlfesom[:,0]
                    
                    print('5-phytoplankton model is used')

            else:
            
                print('3-phytoplankton model is used')

        
        else:
            
            print('2-phytoplankton model is used')
        
        
        # interpolate FESOM CHl.a to regular -------------------------------------------------------------------------------------
        PhyChlfesom_surf_interp = pf.fesom2regular(
                data = PhyChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        DiaChlfesom_surf_interp = pf.fesom2regular(
                data = DiaChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        if cocco_path.is_file():
            CoccoChlfesom_surf_interp = pf.fesom2regular(
                data = CoccoChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if phaeo_path.is_file():
            PhaeoChlfesom_surf_interp = pf.fesom2regular(
                data = PhaeoChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if diah_path.is_file():
            DiaHChlfesom_surf_interp = pf.fesom2regular(
                data = DiaHChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        # Nanophyto + Diatoms (+ Coccos / + Phaeos): TOTAL CHLOROPHYLL -------------------------------------------------------------------------------------
        PhyChlfesom_surf_interp_log10 = np.log10(PhyChlfesom_surf_interp)
        DiaChlfesom_surf_interp_log10 = np.log10(DiaChlfesom_surf_interp)
        if cocco_path.is_file():
            CoccoChlfesom_surf_interp_log10 = np.log10(CoccoChlfesom_surf_interp)

        if phaeo_path.is_file():
            PhaeoChlfesom_surf_interp_log10 = np.log10(PhaeoChlfesom_surf_interp)
        
        Chl_total = PhyChlfesom_surf_interp + DiaChlfesom_surf_interp
        if cocco_path.is_file():
            Chl_total = Chl_total + CoccoChlfesom_surf_interp

        if phaeo_path.is_file():
            Chl_total = Chl_total + PhaeoChlfesom_surf_interp

        if diah_path.is_file():
            Chl_total = Chl_total + DiaHChlfesom_surf_interp
            
        Chl_total_log10 = np.log10(Chl_total)
        
        if False: # interpolation check
            Chl_total_preinterp = ChlAfesom_surf + DiaChlfesom_surf
            
            print('\nFESOM interpolation check:\noriginal min {0:5.4f} max {1:5.4f} mean {2:5.4f} \ninterpol min {3:5.4f} max {4:5.4f} mean {5:5.4f}'.format(
                np.nanmin(Chl_total_preinterp),np.nanmax(Chl_total_preinterp),np.nanmean(Chl_total_preinterp),
                np.nanmin(Chl_total),np.nanmax(Chl_total),np.nanmean(Chl_total)))
        
            fig = plt.figure(figsize=(10,10))
            ax1 = plt.subplot(projection = ccrs.Robinson())

            m1 = plt.pcolormesh(londic, latdic, Chl_total, 
                transform = ccrs.PlateCarree(),
                norm=colors.LogNorm(vmin=np.nanmin(Chl_total), 
                                    vmax=np.nanmax(Chl_total)),
                cmap='viridis')

            ax1.coastlines(resolution='110m', color='black', linewidth=1)

            cbar = fig.colorbar(m1,orientation = 'horizontal',fraction=0.1, pad=0.1) 
            cbar.set_label('Interpolated '+unitfesom, fontsize=20)

        # apply sea mask to OCCCI as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        OCCCIchla_ma = np.copy(OCCCIchla)
        OCCCIchla_ma[~np.isfinite(Chl_total)] = np.nan
        
        OCCCIchla_ma_log10 = np.log10(OCCCIchla_ma)
        
        # check CHl.a data -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nChl.a\nOCCCI min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f} (Mean over 0 to {4} m)'.format(
                    np.nanmin(Chl_total),np.nanmax(Chl_total),
                    np.nanmin(OCCCIchla_ma),np.nanmax(OCCCIchla_ma),
                    f_depth))

            print('\nlog10(Chl.a)\nOCCCI min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f} (Mean over 0 to {4} m)'.format(
                    np.nanmin(Chl_total_log10),np.nanmax(Chl_total_log10),
                    np.nanmin(OCCCIchla_ma_log10),np.nanmax(OCCCIchla_ma_log10),
                    f_depth))
        
        if(self.plotting):
            # plot each CHl.a dataset -------------------------------------------------------------------------------------        
            
            levels = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                                   0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                                   1,2,3,4,5])
            ticks = [0,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1,3,5]
            ticks_label = ['0','0.01','0.03','0.05','0.07','0.1','0.3','0.5','0.7','1','3','5'] # +7 ?

            def mygrid(m):
                m.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='gray')


            # if phaeos and coccos are used ----------------------------------------------------------------------------------------
            if cocco_path.is_file() and phaeo_path.is_file() and not diah_path.is_file() :
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        AB
                        CD
                        EF
                        GG
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, PhyChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, DiaChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)

                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, CoccoChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_title('FESOM-REcoM Coccos', fontsize=16)

                # FESOM phaeocystis
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, PhaeoChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_title('FESOM-REcoM Phaeo', fontsize=16)
                
                # FESOM
                m5 = axes['E']
                f5 = m5.pcolormesh(londic, latdic, Chl_total, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m5)
                m5.set_title('FESOM-REcoM Total', fontsize=16)
            
                
                # OC-CCI
                m6 = axes['F']
                f6 = m6.pcolormesh(londic, latdic, OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m6)
                m6.set_title('OC-CCI', fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label('Chl.a [mg m$^{-3}$]', fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-2,2.125,0.125)
                m7 = axes['G']
                f7 = m7.pcolormesh(londic, latdic, Chl_total - OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256),
                                   #vmin=-3, vmax=3
                                   )
                #f3.set_clim([-2, 2])

                mygrid(m7)
                m7.set_title('FESOM-REcoM - OC-CCI', fontsize=16)

                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f7,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                #location ='bottom',
                                extend = 'both',
                                ticks = [-3,-2,-1,0,1,2,3],
                                ) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                m7.text(-0.12, 1.05, 'G', transform=m7.transAxes,
                            size=30, weight='bold')

                    
                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())
                m7.set_extent(box, ccrs.PlateCarree())

                
            elif cocco_path.is_file() and phaeo_path.is_file() and diah_path.is_file():
                
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        AB
                        CD
                        EF
                        GG
                        HH
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, PhyChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, DiaChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)

                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, CoccoChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_title('FESOM-REcoM Coccos', fontsize=16)

                # FESOM phaeocystis
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, PhaeoChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_title('FESOM-REcoM Phaeo', fontsize=16)

                # FESOM DiaH
                m5 = axes['E']
                f5 = m5.pcolormesh(londic, latdic, DiaHChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m5)
                m5.set_title('FESOM-REcoM DiaH', fontsize=16)
                
                # FESOM
                m6 = axes['F']
                f6 = m6.pcolormesh(londic, latdic, Chl_total, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m6)
                m6.set_title('FESOM-REcoM Total', fontsize=16)
            
                
                # OC-CCI
                m7 = axes['G']
                f7 = m7.pcolormesh(londic, latdic, OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m7)
                m7.set_title('OC-CCI', fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label('Chl.a [mg m$^{-3}$]', fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-2,2.125,0.125)
                m8 = axes['H']
                f8 = m7.pcolormesh(londic, latdic, Chl_total - OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256),
                                   #vmin=-3, vmax=3
                                   )
                #f3.set_clim([-2, 2])

                mygrid(m8)
                m8.set_title('FESOM-REcoM - OC-CCI', fontsize=16)

                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f7,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                #location ='bottom',
                                extend = 'both',
                                ticks = [-3,-2,-1,0,1,2,3],
                                ) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                m7.text(-0.12, 1.05, 'G', transform=m7.transAxes,
                            size=30, weight='bold')
                m8.text(-0.12, 1.05, 'H', transform=m8.transAxes,
                            size=30, weight='bold')
                    
                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())
                m7.set_extent(box, ccrs.PlateCarree())
                m8.set_extent(box, ccrs.PlateCarree())
            
            # if coccos are used ----------------------------------------------------------------------------------------
            elif cocco_path.is_file() and not phaeo_path.is_file() and not diah_path.is_file():
                
                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        ABC
                        DEF
                        GGG
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, PhyChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, DiaChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)

                
                # FESOM coccolithophores
                m3 = axes['C']
                f3 = m3.pcolormesh(londic, latdic, CoccoChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_title('FESOM-REcoM Coccos', fontsize=16)
                
                
                
            
                
                # OC-CCI
                m5 = axes['F']
                f1 = m5.pcolormesh(londic, latdic, OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m5)
                m5.set_title('OC-CCI', fontsize=16)


                
                # Filling gap
                m7 = axes['E']
                axes['E'].remove()
                
                
                
                # FESOM
                m4 = axes['D']
                f4 = m4.pcolormesh(londic, latdic, Chl_total, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m4)
                m4.set_title('FESOM-REcoM Total', fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label('Chl.a [mg m$^{-3}$]', fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-2,2.125,0.125)
                m6 = axes['G']
                f6 = m6.pcolormesh(londic, latdic, Chl_total - OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256),
                                   #vmin=-3, vmax=3
                                   )
                #f3.set_clim([-2, 2])

                mygrid(m6)
                m6.set_title('FESOM-REcoM - OC-CCI', fontsize=16)

                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f6,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                #location ='bottom',
                                extend = 'both',
                                ticks = [-3,-2,-1,0,1,2,3],
                                ) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')
                m6.text(-0.12, 1.05, 'F', transform=m6.transAxes,
                            size=30, weight='bold')
                    
                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
                m6.set_extent(box, ccrs.PlateCarree())

                
            # if coccos and phaeos are not used -------------------------------------------------------------------------------------------    
            else:

                fig = plt.figure(figsize=(15,15), constrained_layout=False)
                axes = fig.subplot_mosaic(
                        """
                        AB
                        CD
                        EE
                        """,
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, 
                        subplot_kw=dict(projection=self.mapproj))             

                # FESOM nanophyto
                m1 = axes['A']
                f1 = m1.pcolormesh(londic, latdic, PhyChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m1)
                m1.set_title('FESOM-REcoM small phytoplankton', fontsize=16)


                # FESOM diatom
                m2 = axes['B']
                f2 = m2.pcolormesh(londic, latdic, DiaChlfesom_surf_interp, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m2)
                m2.set_title('FESOM-REcoM Diatom', fontsize=16)

                # OC-CCI
                m4 = axes['D']
                f1 = m4.pcolormesh(londic, latdic, OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                                   #vmin=1e-3,vmax=5e3)
                mygrid(m4)
                m4.set_title('OC-CCI', fontsize=16)


                # FESOM
                m3 = axes['C']
                f2 = m3.pcolormesh(londic, latdic, Chl_total, 
                                   transform = ccrs.PlateCarree(),
                                   norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                mygrid(m3)
                m3.set_title('FESOM-REcoM Total', fontsize=16)

                cbar1_ax = fig.add_axes([0.92, 0.44, 0.02, 0.4])

                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'vertical',
                                ticks = ticks,
                                fraction=0.1, pad=0.1,
                                extend = 'max') 
                cbar1.set_label('Chl.a [mg m$^{-3}$]', fontsize=14)
                cbar1.ax.tick_params(labelsize=14)



                # OC-CCI - FESOM
                levels_diff = np.arange(-2,2.125,0.125)
                m5 = axes['E']
                f3 = m5.pcolormesh(londic, latdic, Chl_total - OCCCIchla_ma, 
                                   transform = ccrs.PlateCarree(),
                                   cmap = 'RdBu_r',
                                   norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256),
                                   #vmin=-3, vmax=3
                                   )
                #f3.set_clim([-2, 2])

                mygrid(m5)
                m5.set_title('FESOM-REcoM - OC-CCI', fontsize=16)

                # add one colorbar for difference plot below figure

                #fig.subplots_adjust(right=0.8)
                cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.2])

                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'vertical',
                                #location ='bottom',
                                extend = 'both',
                                ticks = [-3,-2,-1,0,1,2,3],
                                ) 
                cbar2.ax.tick_params(labelsize=14)
                cbar2.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                            size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                m4.text(-0.12, 1.05, 'D', transform=m4.transAxes,
                            size=30, weight='bold')
                m5.text(-0.12, 1.05, 'E', transform=m5.transAxes,
                            size=30, weight='bold')

                m1.set_extent(box, ccrs.PlateCarree())
                m2.set_extent(box, ccrs.PlateCarree())
                m3.set_extent(box, ccrs.PlateCarree())
                m4.set_extent(box, ccrs.PlateCarree())
                m5.set_extent(box, ccrs.PlateCarree())
            
            #plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'Chla_OCCCI'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'Chla_OCCCI'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  

        if(self.Taylor):
            # statistics  -------------------------------------------------------------------------------------            
            # preparation of datasets
            if np.isnan(np.min(OCCCIchla_ma_log10)): print('WARNING: OCCCI field contains NaNs')
            if np.isnan(np.min(Chl_total_log10)): print('WARNING: FESOM field contains NaNs')

            # get statistics only from valid OCCCI gridpoints 
            ind_stat = np.where(np.isfinite(OCCCIchla_ma_log10))

            title = 'log10 surface Chlorophyll'
            print('\nStatistics for '+title)
            plot_taylor_norm(OCCCIchla_ma_log10[ind_stat],Chl_total_log10[ind_stat],
                                    mask=True,title=title)

            
            #plt.show(block=False) 
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):                
                plt.savefig(self.savepath+self.runname+'_'+'Chla_OCCCI_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'Chla_OCCCI_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False) 
        
        if(self.output):
            self.lon = londic
            self.lat = latdic
            self.chl_oc = OCCCIchla_ma
            self.chl_fesom = Chl_total
            self.chld_fesom = DiaChlfesom_surf_interp
            self.chln_fesom = PhyChlfesom_surf_interp
            if cocco_path.is_file():
                self.chlc_fesom = CoccoChlfesom_surf_interp
            if phaeo_path.is_file():
                self.chlp_fesom = PhaeoChlfesom_surf_interp
            
            self.unit = unitfesom

class plot_maps_chl_southern:
    '''
    class Chlsurf_SO_comp(runname,resultpath,savepath,meshpath,ncfileJohnson2013,first_year,last_year,
                 savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
                 
    pathJohnson2013: path to numpy output files of Johnson2013 mean data 
    folder should contain:
    -Johnson2013_MEAN_1x1_Chl_mg_m3.npy
    assuming 1x1 grid as 
    ilat         = np.arange(-89.5,-29.5,1.)
    ilon         = np.arange(-179.5,180.,1.)
    
    n_levels = 1: number of mesh levels used for FESOM surface mean
    '''
    
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncfileJohnson2013=ncfileChlSouthernJohnson,
                 first_year=first_year,
                 last_year=last_year,
                 savefig=False,
                 n_levels = 1,
                 verbose=False,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.savefig = savefig
        self.n_levels = n_levels
        self.ncfileJohnson2013 = ncfileJohnson2013
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        mapproj = pf.get_proj('sp')
        
        if(self.verbose):
            print('Processing {0}'.format(self.resultpath))
        
        # load Johnson2013 CHl.a data -------------------------------------------------------------------------------------
        chlJohnson = np.load(self.ncfileJohnson2013+'')
        
        #lat = np.load(pathJohnson2013+'Johnson2013_lat.npy')
        #lon = np.load(pathJohnson2013+'Johnson2013_lon.npy')
        
        
        chlJohnson_log10 = np.log10(chlJohnson)
        
        chlJohnsonlabel = 'Johnson et al. (1998-2019)'
        chlJohnsonunit = 'Chl.a [mg Chl m$^{-3}$]'
                
        # define regular mesh -------------------------------------------------------------------------------------
        lat = np.arange(-89.5,90,1)
        lat_SO = np.arange(-89.5,-29.5,1.)
        lon = np.arange(-179.5,180.,1.)
        
        latdic, londic = np.meshgrid(lat, lon)
        
        latdic_SO, londic_SO = np.meshgrid(lat_SO, lon)
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(self.meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        lon_fesom = mesh.x2
        lat_fesom = mesh.y2        
        
        # surface: mean over n mesh levels
        #self.n_levels
        f_depth = mesh.zlev[self.n_levels]
        if(self.verbose):
            print('***\nUsing upper {0} layers to depth {1} m for surface FESOM data!\n***'.format(
                n_levels,f_depth))
        
        # load FESOM Nanophyto Chl.a data -------------------------------------------------------------------------------------        
        #ncFESOMChl = self.resultpath + '/PhyChl.fesom.1948.nc'
        #!ncdump -h $ncFESOMChl
        
        ChlAfesom = pf.get_data(self.resultpath, "PhyChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
        
        
        ChlAfesom_surf = np.nanmean(ChlAfesom[:,:n_levels],axis=1)
        
        # mean over whole water column
        #ChlAfesom_mean = np.nanmean(ChlAfesom, axis=1)
        
        labelfesomNano = 'FESOM Nanophyto Chl.a {0}-{1}'.format(self.fyear,self.lyear)        
        #unitfesomNano = 'Chl.a [mmol m$^{-3}$]' 

        # load FESOM Diatom Chl.a data -------------------------------------------------------------------------------------
        
        #ncFESOMDiaChl = self.resultpath + '/DiaChl.fesom.1948.nc'
        #!ncdump -h $ncFESOMDiaChl
        
        DiaChlfesom = pf.get_data(self.resultpath, "DiaChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
        
        labelfesomDia = 'FESOM Diatom Chl.a {0}-{1}'.format(self.fyear,self.lyear)
        
        # surface: mean over n mesh levels
        DiaChlfesom_surf = np.nanmean(DiaChlfesom[:,:n_levels],axis=1)
    
        unitfesom = 'Chl.a [mg m$^{-3}$]'
        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        
        
        # load FESOM Coccolithophore Chl.a data -------------------------------------------------------------------------------------

        from pathlib import Path
        cocco_path = Path(self.resultpath + '/CoccoChl.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/PhaeoChl.fesom.'+str(years[0])+'.nc') # assuming that phaeo was used for the entire simulation if they were used in the first year of simulation
        diah_path = Path(self.resultpath + '/DiaHChl.fesom.'+str(years[0])+'.nc')
        
        if cocco_path.is_file():
            
            CoccoChlfesom = pf.get_data(self.resultpath, "CoccoChl", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)
            
            labelfesomCocco = 'FESOM-REcoM Cocco Chl.a {0}-{1}'.format(self.fyear,self.lyear)
            
            CoccoChlfesom_surf = np.nanmean(CoccoChlfesom[:,:n_levels],axis=1)

            # load FESOM Phaeo Chl.a data ---------------------------------------------------------------------------------------

            if phaeo_path.is_file():
            
                PhaeoChlfesom = pf.get_data(self.resultpath, "PhaeoChl", years, mesh, 
                                   how="mean", compute=True, runid=self.runname, silent=True)
                
                labelfesomPhaeo = 'FESOM-REcoM Phaeo Chl.a {0}-{1}'.format(self.fyear,self.lyear)
                
                PhaeoChlfesom_surf = PhaeoChlfesom[:,0]
                
                print('4-phytoplankton model is used')
                
                if diah_path.is_file():
            
                    DiaHChlfesom = pf.get_data(self.resultpath, "DiaHChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    labelfesomPhaeo = 'FESOM-REcoM DiaH Chl.a {0}-{1}'.format(self.fyear,self.lyear)
                    
                    DiaHChlfesom_surf = DiaHChlfesom[:,0]
                    
                    print('5-phytoplankton model is used')

            else:
            
                print('3-phytoplankton model is used')
        
        else:
            
            print('2-phytoplankton model is used')
        
        # interpolate FESOM CHl.a to regular -------------------------------------------------------------------------------------
        ChlAfesom_surf_interp = pf.fesom2regular(
                data = ChlAfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        DiaChlfesom_surf_interp = pf.fesom2regular(
                data = DiaChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
        if cocco_path.is_file():
            CoccoChlfesom_surf_interp = pf.fesom2regular(
                data = CoccoChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if phaeo_path.is_file():
            PhaeoChlfesom_surf_interp = pf.fesom2regular(
                data = PhaeoChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)

        if diah_path.is_file():
            DiaHChlfesom_surf_interp = pf.fesom2regular(
                data = DiaHChlfesom_surf,
                mesh = mesh,
                lons = londic, 
                lats = latdic)
        
         
        # Nanophyto + Diatoms (+ Coccos): TOTAL CHLOROPHYLL -------------------------------------------------------------------------------------
        ChlAfesom_surf_interp_log10 = np.log10(ChlAfesom_surf_interp)
        DiaChlfesom_surf_interp_log10 = np.log10(DiaChlfesom_surf_interp)
        if cocco_path.is_file():
            CoccoChlfesom_surf_interp_log10 = np.log10(CoccoChlfesom_surf_interp)
        if phaeo_path.is_file():
            PhaeoChlfesom_surf_interp_log10 = np.log10(PhaeoChlfesom_surf_interp)
        if diah_path.is_file():
            DiaHChlfesom_surf_interp_log10 = np.log10(DiaHChlfesom_surf_interp)
        
        Chl_total = ChlAfesom_surf_interp + DiaChlfesom_surf_interp
        if cocco_path.is_file():
            Chl_total = Chl_total + CoccoChlfesom_surf_interp
        if phaeo_path.is_file():
            Chl_total = Chl_total + PhaeoChlfesom_surf_interp
        if diah_path.is_file():
            Chl_total = Chl_total + DiaHChlfesom_surf_interp
        Chl_total_log10 = np.log10(Chl_total)
        
        # cut to Southern Ocean -------------------------------------------------------------------------------------
        ilat = np.where(lat < -30)
        
        Chl_total_SO = np.squeeze(Chl_total[:,ilat])
        Chl_total_SO_log10 = np.squeeze(Chl_total_log10[:,ilat]) 
        
        #print(np.shape(Chl_total_SO))
        
        if False: # interpolation check
            Chl_total_preinterp = ChlAfesom_surf + DiaChlfesom_surf + CoccoChlfesom_surf + PhaeoChlfesom_surf
            
            print('\nFESOM interpolation check:\noriginal min {0:5.4f} max {1:5.4f} mean {2:5.4f} \ninterpol min {3:5.4f} max {4:5.4f} mean {5:5.4f}'.format(
                np.nanmin(Chl_total_preinterp),np.nanmax(Chl_total_preinterp),np.nanmean(Chl_total_preinterp),
                np.nanmin(Chl_total),np.nanmax(Chl_total),np.nanmean(Chl_total)))
        
            fig = plt.figure(figsize=(20,10))
            ax1 = plt.subplot(projection = ccrs.PlateCarree())

            m1 = plt.pcolormesh(londic_SO, latdic_SO, Chl_total_SO, 
                transform = ccrs.PlateCarree(),
                norm=colors.LogNorm(vmin=np.nanmin(Chl_total_SO), 
                                    vmax=np.nanmax(Chl_total_SO)),
                cmap='viridis')

            ax1.coastlines(resolution='110m', color='black', linewidth=1)

            cbar = fig.colorbar(m1,orientation = 'horizontal',fraction=0.1, pad=0.1, extend = 'max') 
            cbar.set_label('Interpolated '+unitfesom, fontsize=20)

        
        # apply sea mask to Johnson as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        #chlJohnson_ma = np.copy(chlJohnson)
        #chlJohnson_ma[~np.isfinite(ChlAfesom_surf_interp)] = np.nan
        
        #chlJohnson_ma_log10 = np.log10(chlJohnson_ma)
        
        # check CHl.a data -------------------------------------------------------------------------------------
        if(self.verbose):
            print('\nChl.a\nJohnson2013 min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f} (Mean over 0 to {4} m)'.format(
                    np.nanmin(Chl_total_SO),np.nanmax(Chl_total_SO),
                    np.nanmin(chlJohnson),np.nanmax(chlJohnson),
                    f_depth))

            print('\nlog10(Chl.a)\nJohnson2013 min = {2:5.4f}, max = {3:5.4f}\nFESOM min = {0:5.4f}, max = {1:5.4f} (Mean over 0 to {4} m)'.format(
                    np.nanmin(Chl_total_SO_log10),np.nanmax(Chl_total_SO_log10),
                    np.nanmin(chlJohnson_log10),np.nanmax(chlJohnson_log10),
                    f_depth))
        
        if(self.plotting):
            # plot each CHl.a dataset -------------------------------------------------------------------------------------        
            levels = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                               0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                               1,2,3,4,5,7])
            ticks = [0,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7,1,3,5,7]
            ticks_label = ['0','0.01','0.03','0.05','0.07','0.1','0.3','0.5','0.7','1','3','5','7']

            def mygrid(m,grid=False):
                #m.coastlines(resolution='110m', color='black', linewidth=1)
                m.add_feature(cfeature.LAND, zorder=1, edgecolor='none', facecolor='gray')
                if grid:
                    g1 = m.gridlines(draw_labels = True)
                    g1.xlabels_top = False
                    g1.xlabel_style = {'size': 16, 'color': 'gray'}
                    g1.ylabel_style = {'size': 16, 'color': 'gray'}


            # plot Johnson2013 and SUM CHl.a data -------------------------------------------------------------------------------------            
            fig, axes = plt.subplots(1,3, 
                                     subplot_kw=dict(projection=mapproj),
                                     gridspec_kw={'hspace': 0.01, 'wspace': 0.1},
                                     figsize=(20,7), constrained_layout=False)
            # Johnson
            m1 = axes[1]
            f1 = m1.pcolormesh(londic_SO, latdic_SO, chlJohnson, 
                               transform = ccrs.PlateCarree(),
                               norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
                               #vmin=1e-3,vmax=5e3)
            mygrid(m1)
            m1.set_title(chlJohnsonlabel, fontsize=16)


            # FESOM
            m2 = axes[0]
            f2 = m2.pcolormesh(londic_SO, latdic_SO, Chl_total_SO, 
                               transform = ccrs.PlateCarree(),
                               norm=colors.BoundaryNorm(boundaries=levels, ncolors=256))
            mygrid(m2)
            m2.set_title('{0}\n(0 to {1} m)'.format(labelfesom,f_depth), fontsize=16)

            # add one colorbar for first row plots below figure
            cbar = fig.colorbar(f1,
                                ax = axes[:2], 
                                location ='bottom',
                                #orientation = 'horizontal',
                                ticks = ticks,
                                fraction=0.046, pad=0.04) 
            cbar.ax.set_xticklabels(ticks_label, fontsize=16) 
            cbar.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
            # Johnson - FESOM
            levels_diff = np.arange(-3,3,0.125)
            m3 = axes[2]
            f3 = m3.pcolormesh(londic_SO, latdic_SO, Chl_total_SO - chlJohnson, 
                               transform = ccrs.PlateCarree(),
                               cmap = 'RdBu',
                               norm=colors.BoundaryNorm(boundaries=levels_diff, ncolors=256)
                               )
            mygrid(m3,grid=False)
            m3.set_title('FESOM - Johnson et al.(2013)', fontsize=16)
            
            m1.text(-0.12, 1.05, 'B', transform=m1.transAxes,
                        size=30, weight='bold')
            m2.text(-0.12, 1.05, 'A', transform=m2.transAxes,
                        size=30, weight='bold')
            m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                        size=30, weight='bold')
            
            # add one colorbar for difference plot below figure
            cbar = fig.colorbar(f3,
                            ax = axes[2], 
                            orientation = 'horizontal',
                            #location ='bottom',
                            ticks = [-3,-2,-1,0,1,2,3],
                            extend = 'both',
                            fraction=0.046, pad=0.04) 
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Chl.a [mg m$^{-3}$]', fontsize=16)
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):
                plt.savefig(self.savepath+self.runname+'_'+'Chla_Johnson'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'Chla_Johnson'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)  

        if(self.Taylor):
            # statistics  -------------------------------------------------------------------------------------            
            # preparation of datasets
            if np.isnan(np.min(chlJohnson_log10)): print('WARNING: Johnson field contains NaNs')
            if np.isnan(np.min(Chl_total_SO_log10)): print('WARNING: FESOM field contains NaNs')

            # get statistics only from valid OCCCI gridpoints 
            ind_stat = np.where(np.isfinite(chlJohnson_log10))

            title = 'log10 surface Chlorophyll'
            print('\nStatistics for '+title)
            plot_taylor_norm(chlJohnson_log10[ind_stat],Chl_total_SO_log10[ind_stat],
                                    mask=True,title=title)

            
            # fig export  -------------------------------------------------------------------------------------
            if(self.savefig==True):                
                plt.savefig(self.savepath+self.runname+'_'+'Chla_Johnson_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
                plt.savefig(self.savepath+self.runname+'_'+'Chla_Johnson_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                        bbox_inches='tight')
            plt.show(block=False)
            
        if(self.output):
            self.lon = londic_SO
            self.lat = latdic_SO
            self.Chl_johnson = chlJohnson
            self.Chl_fesom = Chl_total_SO


class plot_maps_dfe:
    '''
    class DFecomp
    
    Compare DFe to PISCES
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDFe_pisces,
                 first_year=first_year,
                 last_year=last_year,
                 PISCESvar='Fe',
                 mapproj='rob',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 verbose=True,
                 plotting=True,
                 output=False,
                 Taylor=True,
                 get_overview = False,runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.PISCESvar = PISCESvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        self.output = output
        self.Taylor = Taylor
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        labelfesom = 'FESOM ({0}-{1})'.format(self.fyear,self.lyear)
        unitfesom = 'DFe [mmol m$^{-3}$]' # equals to mumol/L
        labelpisces = 'PISCES'
        unitpisces = 'DFe [mmol m$^{-3}$]'

        # load FESOM data -------------------------------------------------------------------------------------
        DFefesom = pf.get_data(resultpath, "DFe", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runname+'.mesh.diag.nc'
        
        # load PISCES data  -------------------------------------------------------------------------------------
        PISCES_input = load_pisces_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.PISCESvar, get_overview=False)
        pisces_int = PISCES_input.pisces_int    

        # apply sea mask to PISCES as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        pisces_int_ma = np.copy(pisces_int)
        pisces_int_ma[DFefesom == 0] = 0
        
        # plot PISCES and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] > np.max(PISCES_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(d,np.max(PISCES_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]
                    
                DFefesom_mean = pf.layermean_data(DFefesom, mesh, uplow = uplow)
                DFepisces_mean = pf.layermean_data(pisces_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPISCES mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(DFefesom_mean),np.nanstd(DFefesom_mean),np.nanmin(DFefesom_mean),np.nanmax(DFefesom_mean),
                        np.nanmean(DFepisces_mean),np.nanstd(DFepisces_mean),np.nanmin(DFepisces_mean),np.nanmax(DFepisces_mean)))
                
                if plotting:
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(0,3.1,.1)
                    f1 = pf.subplot(mesh, fig, m1, [DFefesom_mean],
                                levels = levels,
                                units=unitpisces, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [DFepisces_mean], 
                                levels = levels,
                                units=unitpisces, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelpisces+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-3,3.1,0.1)
                    f3 = pf.subplot(mesh, fig, m3, [DFefesom_mean-DFepisces_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitpisces, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - PISCES ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')
                    
                    #fig.subplots_adjust(bottom=0.1)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DFe'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DFe'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.pdf'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(DFepisces_mean)): print('WARNING: The interpolated PISCES field contains NaNs at depth')
                    if np.isnan(np.min(DFefesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Normalized Taylor Diagram for DFe'
                    plt_taylor_comp(pisces_int_ma,DFefesom,mask=True,title=title, depth_array=depth_array, mesh=mesh,verbose = self.verbose)
                    
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                        dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                        bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] > np.max(PISCES_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(PISCES_input.layer_depths)))

            DFefesom_mean = pf.layermean_data(DFefesom, mesh, uplow = uplow)
            DFepisces_mean = pf.layermean_data(pisces_int_ma, mesh, uplow = uplow)
            
            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nPISCES mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(DFefesom_mean),np.nanstd(DFefesom_mean),np.nanmin(DFefesom_mean),np.nanmax(DFefesom_mean),
                np.nanmean(DFepisces_mean),np.nanstd(DFepisces_mean),np.nanmin(DFepisces_mean),np.nanmax(DFepisces_mean)))
                
        
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                            """
                            AB
                            CC
                            """,
                            gridspec_kw={'hspace': 0.1, 'wspace': 0.01, 'bottom': 0.03}, 
                            subplot_kw=dict(projection=self.mapproj))
                
                
                m1 = axes['A']
                levels = np.arange(0,3.1,.1)
                f1 = pf.subplot(mesh, fig, m1, [DFefesom_mean],
                            levels = levels,
                            units=unitpisces, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n (0-{0} m)'.format(uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [DFepisces_mean], 
                            levels = levels,
                            units=unitpisces, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelpisces+'\n (0-{0} m)'.format(uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-3,3.1,0.1)
                f3 = pf.subplot(mesh, fig, m3, [DFefesom_mean-DFepisces_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitpisces, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - PISCES (0-{0} m)'.format(uplow[1]),
                            box=box,
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                
                #fig.subplots_adjust(bottom=0.3)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
                
            if Taylor:  
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(DFepisces_mean)): print('WARNING: The interpolated PISCES field contains NaNs at depth')
                if np.isnan(np.min(DFefesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for DFe \n(mean over depth, max = {0}-{1}m)'.format(uplow[0],uplow[1])
                plot_taylor_norm(DFepisces_mean,DFefesom_mean,mask=True,title=title)
                
                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DFe_PISCES_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)  
            
            if output:
                    self.DFefesom_mean = DFefesom_mean
                    self.DFepisces_mean   = DFepisces_mean


class plot_maps_dsi:
    '''
    class DSicomp
    
    -use layerwise = True to compare a set of depth
    -use depth_limit to specify maximum depth for mean-over-depth comparison
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDSi,
                 first_year=first_year,
                 last_year=last_year,
                 WOAvar='i_an',
                 mapproj='rob',
                 cmap = 'viridis',
                 savefig=False,
                 layerwise=False,
                 depth_array=(0,50,200,1000,2000,4000),
                 uplow=[0, 100],
                 cmap_extension='max',
                 output=False,
                 Taylor=True,
                 get_overview = False,
                 verbose = False,
                 plotting = True,
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.ncpath = ncpath
        self.fyear = first_year
        self.lyear = last_year
        self.WOAvar = WOAvar
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.layerwise = layerwise
        self.depth_array = depth_array
        self.uplow = uplow
        self.cmap_extension = cmap_extension
        self.verbose = verbose
        self.plotting = plotting
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        
        # load FESOM data -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        NCfesom = self.resultpath + '/DSi.'+self.runname+'.'+str(self.fyear)+'.nc'
        labelfesom = 'FESOM {0}-{1}'.format(self.fyear,self.lyear)
        unitfesom = 'DSi [mmol Si m$^{-3}$]' 

        # load data -------------------------------------------------------------------------------------
        fesom = pf.get_data(resultpath, "DSi", years, mesh, 
                               how="mean", compute=True, runid=self.runname, silent=True)

        # load WOA data  -------------------------------------------------------------------------------------
        WOA_input = load_woa_data(self.runname,self.resultpath,self.mesh,self.ncpath,self.WOAvar, get_overview=False)
        woa_int = WOA_input.woa_int    
        
        labelwoa = 'WOA'
        unitwoa = 'DSi [mmol Si m$^{-3}$]' #i_an:units = "micromoles_per_liter" = mmol/m3

        # apply sea mask to WOA as in FESOM ----------------------------------------------------------------------------------
        # assumption: there is no ocean where value in FESOM == 0
        woa_int_ma = np.copy(woa_int)
        woa_int_ma[fesom == 0] = 0

        # plot WOA and FESOM ----------------------------------------------------------------------------------
        if(self.layerwise):
            if(self.depth_array == []):
                depth_array = (0,50,200,1000,2000,4000)
                
            for i in np.arange(len(depth_array)-1):
                if depth_array[i+1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))
                
                uplow = [depth_array[i], depth_array[i+1]]

                fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
                woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)
                
                if(self.verbose):
                    print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                        np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                        np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))

                if (self.plotting):
                    fig = plt.figure(figsize=(15,12), constrained_layout=False)
                    axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))

                    m1 = axes['A']
                    levels = np.arange(0,62,2)
                    f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    m2 = axes['B']
                    f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                                levels = levels,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = self.cmap,
                                cmap_extension=self.cmap_extension,
                                titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )

                    cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                    cbar1 = fig.colorbar(f1,
                                    cax = cbar1_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar1.set_label(unitfesom, fontsize=18)
                    cbar1.ax.tick_params(labelsize=18)

                    m3 = axes['C']

                    levels_diff = np.arange(-30,31,1)
                    f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                                rowscol= (1,1),
                                levels = levels_diff,
                                units=unitwoa, 
                                mapproj=self.mapproj, # robinson projection takes more time!
                                cmap = 'RdBu_r',
                                cmap_extension='both',
                                titles='FESOM - WOA ({0}-{1} m)'.format(uplow[0],uplow[1]),
                                box=box,
                               )
                    
                    m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                                size=30, weight='bold')
                    m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                                size=30, weight='bold')
                    m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                                size=30, weight='bold')

                    fig.subplots_adjust(bottom=0.02)
                    cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                    cbar2 = fig.colorbar(f3,
                                    cax = cbar2_ax, 
                                    orientation = 'horizontal',
                                    fraction=0.046, pad=0.04) 
                    cbar2.set_label(unitfesom, fontsize=18)
                    cbar2.ax.tick_params(labelsize=18)
                    
                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'_{0}-{1}m.png'.format(uplow[0], uplow[1]), 
                                bbox_inches='tight')
                    plt.show(block=False)
                
                if Taylor:
                    # statistics  -------------------------------------------------------------------------------------
                    # preparation of datasets
                    if np.isnan(np.min(dsi_int)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                    if np.isnan(np.min(DSifesom)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                    title = 'Taylor Diagram for DSi at {0} m'.format(plot_depth)
                    plt_taylor_comp(dsi_int,DSifesom,mask=True,title=title, depth_array=depth_array, mesh=mesh)


                    # fig export  -------------------------------------------------------------------------------------
                    if(self.savefig==True):
                        plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.png', 
                                        dpi = 300, bbox_inches='tight')
                        plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'_layerwise.png', 
                                        dpi = 300, bbox_inches='tight')
                    plt.show(block=False)
                    
                if output:
                    print('Only return non-layerwise output')
        
        # mean over depth  -------------------------------------------------------------------------------------
        else:
            if uplow[1] < -np.max(WOA_input.layer_depths):
                    print('{0}m is too deep for climatology.\nPlease consider choosing max depth at {1}!\n***'.format(uplow[1],np.max(GLODAP_input.layer_depths)))

            fesom_mean = pf.layermean_data(fesom, mesh, uplow = uplow)
            woa_int_ma_mean = pf.layermean_data(woa_int_ma, mesh, uplow = uplow)

            if(self.verbose):
                print('\nFESOM mean = {0:4.4f}, std = {1:4.4f}, min = {2:4.4f}, max = {3:4.4f}\nWOA mean = {4:4.4f}, std = {5:4.4f}, min = {6:4.4f}, max = {7:4.4f}'.format(
                np.nanmean(fesom_mean),np.nanstd(fesom_mean),np.nanmin(fesom_mean),np.nanmax(fesom_mean),
                np.nanmean(woa_int_ma_mean),np.nanstd(woa_int_ma_mean),np.nanmin(woa_int_ma_mean),np.nanmax(woa_int_ma_mean)))
            
            if plotting:
                fig = plt.figure(figsize=(15,12), constrained_layout=False)
                axes = fig.subplot_mosaic(
                                """
                                AB
                                CC
                                """,
                                gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'bottom': 0.03}, 
                                subplot_kw=dict(projection=self.mapproj))
                    
                m1 = axes['A']
                levels = np.arange(0,62,2)
                f1 = pf.subplot(mesh, fig, m1, [fesom_mean],
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelfesom+'\n ({0}-{1} m)'.format(uplow[0], uplow[1]),
                            box=box,
                           )
                    
                m2 = axes['B']
                f2 = pf.subplot(mesh, fig, m2, [woa_int_ma_mean], 
                            levels = levels,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = self.cmap,
                            cmap_extension=self.cmap_extension,
                            titles=labelwoa+'\n ({0}-{1} m)'.format(uplow[0], uplow[1]),
                            box=box,
                           )
                    
                cbar1_ax = fig.add_axes([0.13, 0.53, 0.76, 0.02])
                cbar1 = fig.colorbar(f1,
                                cax = cbar1_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04)
                cbar1.set_label(unitfesom, fontsize=18)
                cbar1.ax.tick_params(labelsize=18)
        
                m3 = axes['C']

                levels_diff = np.arange(-30,31,1)
                f3 = pf.subplot(mesh, fig, m3, [fesom_mean-woa_int_ma_mean], 
                            rowscol= (1,1),
                            levels = levels_diff,
                            units=unitwoa, 
                            mapproj=self.mapproj, # robinson projection takes more time!
                            cmap = 'RdBu_r',
                            cmap_extension='both',
                            titles='FESOM - WOA'+' ({0}-{1} m)'.format(uplow[0], uplow[1]),
                            box=box,
                           )

                m1.text(-0.12, 1.05, 'A', transform=m1.transAxes,
                        size=30, weight='bold')
                m2.text(-0.12, 1.05, 'B', transform=m2.transAxes,
                            size=30, weight='bold')
                m3.text(-0.12, 1.05, 'C', transform=m3.transAxes,
                            size=30, weight='bold')
                
                fig.subplots_adjust(bottom=0.02)
                cbar2_ax = fig.add_axes([0.13, 0.001, 0.76, 0.02])
                cbar2 = fig.colorbar(f3,
                                cax = cbar2_ax, 
                                orientation = 'horizontal',
                                fraction=0.046, pad=0.04) 
                cbar2.set_label(unitfesom, fontsize=18)
                cbar2.ax.tick_params(labelsize=18)

                
                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):
                    plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)
            
            if Taylor: 
                # statistics  -------------------------------------------------------------------------------------            
                # preparation of datasets
                if np.isnan(np.min(woa_int_ma_mean)): print('WARNING: The interpolated WOA field contains NaNs at depth')
                if np.isnan(np.min(fesom_mean)): print('WARNING: The interpolated FESOM field contains NaNs at depth')

                title = 'Taylor Diagram for DSi \n(mean over depth, max = {0}-{1}m)'.format(uplow[0], uplow[1]),
                plot_taylor_norm(woa_int_ma_mean,fesom_mean,mask=True,title=title)

                # fig export  -------------------------------------------------------------------------------------
                if(self.savefig==True):                
                    plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                            dpi = 300, bbox_inches='tight')
                    plt.savefig(self.savepath+self.runname+'_'+'DSi_WOA_Taylor'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                            bbox_inches='tight')
                plt.show(block=False)  
        
            if output:
                    self.fesom = fesom_mean
                    self.woa   = woa_int_ma_mean


class plot_maps_limfact:
    '''
    class LimFact
    
    Derive and plot most limiting factor (nutrient vs. light) at surface
    
    '''
    def __init__(self,
                 mesh,
                 resultpath=resultpath,
                 savepath=savepath,
                 ncpath=ncfileDSi,
                 first_year=first_year,
                 last_year=last_year,
                 Tfunction = 'New_functions',
                 mapproj='rob',
                 cmap = cmo.cm.thermal,
                 savefig=False,
                 verbose=False,
                 plotting=True,
                 output=True,
                 frequency='yearly',
                 runname='fesom'):

        self.runname = runname
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.fyear = first_year
        self.lyear = last_year
        self.Tfunction = Tfunction
        self.mapproj = mapproj
        self.cmap = cmap
        self.savefig = savefig
        self.verbose = verbose
        self.plotting = plotting
        self.frequency = frequency
        self.output = output
        
        if self.mapproj == 'rob':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'pc':
            box=[-180, 180, -90, 90]
        elif self.mapproj == 'sp':
            box=[-180, 180, -90, -30]
        elif self.mapproj == 'np':
            box=[-180, 180, 60, 90]
            
        self.mapproj = pf.get_proj(self.mapproj)
        
        # load FESOM data -------------------------------------------------------------------------------------
        #print('Processing ' + self.resultpath)
        
        #mesh       = pf.load_mesh(self.meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # Check if it is a version with or without coccolithophores
        cocco_path  = Path(self.resultpath + '/CoccoC.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation 
        
        # model parameters
        if Tfunction == 'Arrhenius': # Values for the model version without coccolithophores
            print("Version with diatoms, small phytoplankton, ")
            alfa_phy, alfa_dia, Pcm, Pcm_d, C2K, rTref, Ae = 0.14, 0.19, 3.0, 3.5, 273.15, 1/288.15, 4500.0
            k_din_phy, k_din_dia, k_fe_phy, k_fe_dia, k_si_dia = 0.55, 1.0, 0.04, 0.12, 4.0
            NMinSlope, SiMinSlope, NCmin, SiCmin  = 50, 1000, 0.04, 0.04
            a_co2_phy, a_co2_dia = 1.162, 1.040
            b_co2_phy, b_co2_dia = 48.88, 28.90
            c_co2_phy, c_co2_dia = 0.2255, 0.8778
            d_co2_phy, d_co2_dia = 1.023e+07, 2.640e+06
            Cunits = 976.5625 # Conversion factor between [mol/m3] (model) and [umol/kg] (function): (1000 * 1000) / 1024
            if cocco_path.is_file(): # Values for the model version with coccolithophores
                print("... and coccolithophores")
                alfa_phy, alfa_dia, alfa_cocco, Pcm, Pcm_d, Pcm_c, C2K, rTref, Ae = 0.14, 0.19, 0.10, 3.0, 3.5, 2.8, 273.15, 1/288.15, 4500.0
                k_din_phy, k_din_dia, k_din_cocco, k_fe_phy, k_fe_dia, k_fe_cocco, k_si_dia = 0.55, 1.0, 0.9, 0.04, 0.12, 0.09, 4.0
                NMinSlope, SiMinSlope, NCmin, SiCmin  = 50, 1000, 0.04, 0.04
                a_co2_phy, a_co2_dia, a_co2_cocco = 1.162, 1.040, 1.109
                b_co2_phy, b_co2_dia, b_co2_cocco = 48.88, 28.90, 37.67
                c_co2_phy, c_co2_dia, c_co2_cocco = 0.2255, 0.8778, 0.3912
                d_co2_phy, d_co2_dia, d_co2_cocco = 1.023e+07, 2.640e+06, 9.450e+06
                Cunits = 976.5625 # Conversion factor between [mol/m3] (model) and [umol/kg] (function): (1000 * 1000) / 1024
        elif Tfunction == 'New_functions':
            print("Version with diatoms, small phytoplankton, coccolithophores, and polar Phaeocystis")
            alfa_phy, alfa_dia, alfa_cocco, alfa_phaeo = 0.15, 0.19, 0.10, 0.17
            base_phy, base_dia, base_cocco, base_phaeo = -0.7338, -0.2216, -0.231, -0.2310
            exp_phy, exp_dia, exp_cocco, exp_phaeo = 0.06422, 0.0406, 0.0327, 0.0327
            k_din_phy, k_din_dia, k_din_cocco, k_din_phaeo, k_fe_phy, k_fe_dia, k_fe_cocco, k_fe_phaeo, k_si_dia = 0.55, 1.0, 0.9, 0.7, 0.04, 0.12, 0.09, 0.09, 4.0
            NMinSlope, SiMinSlope, NCmin, SiCmin  = 50, 1000, 0.04, 0.04
            a_co2_phy, a_co2_dia, a_co2_cocco, a_co2_phaeo = 1.162, 1.040, 1.109, 1.077
            b_co2_phy, b_co2_dia, b_co2_cocco, b_co2_phaeo = 48.88, 28.90, 37.67, 33.06
            c_co2_phy, c_co2_dia, c_co2_cocco, c_co2_phaeo = 0.2255, 0.8778, 0.3912, 0.5775
            d_co2_phy, d_co2_dia, d_co2_cocco, d_co2_phaeo = 1.023e+07, 2.640e+06, 9.450e+06, 5.891e+06
            Cunits = 976.5625 # Conversion factor between [mol/m3] (model) and [umol/kg] (function): (1000 * 1000) / 1024
        else:
            print("Please select the temperature function you used (Arrhenius or New_functions)")
            
            # ==============================================================================
            # Loading data
            
        if frequency == 'monthly':
                print('monthly frequency selected')
                T2d_loc             = np.zeros(shape=(12,len(mesh.x2)))
                Phy_Light_limiter   = np.zeros(shape=(12,len(mesh.x2)))
                Phy_CO2_limiter     = np.zeros(shape=(12,len(mesh.x2)))
                Fephy               = np.zeros(shape=(12,len(mesh.x2)))
                Phy_qlimFac1        = np.zeros(shape=(12,len(mesh.x2),2))
                Phy_qlimFac         = np.zeros(shape=(12,len(mesh.x2)))
                Phy_dq              = np.zeros(shape=(12,len(mesh.x2)))
                Phy_recom_limiter   = np.zeros(shape=(12,len(mesh.x2)))
                T_phy               = np.zeros(shape=(12,len(mesh.x2)))
                Pmax_phy            = np.zeros(shape=(12,len(mesh.x2)))
                Dia_Light_limiter   = np.zeros(shape=(12,len(mesh.x2)))
                Dia_CO2_limiter     = np.zeros(shape=(12,len(mesh.x2)))
                Fedia               = np.zeros(shape=(12,len(mesh.x2)))
                Dia_qlimFac1        = np.zeros(shape=(12,len(mesh.x2),3))
                Dia_qlimFac         = np.zeros(shape=(12,len(mesh.x2)))
                Dia_dq              = np.zeros(shape=(12,len(mesh.x2)))
                Dia_recom_limiterN  = np.zeros(shape=(12,len(mesh.x2)))
                T_dia               = np.zeros(shape=(12,len(mesh.x2)))
                Dia_recom_limiterSi = np.zeros(shape=(12,len(mesh.x2)))
                Pmax_dia            = np.zeros(shape=(12,len(mesh.x2)))
                months_limphy       = np.zeros(shape=(12,len(mesh.x2)))
                months_limdia       = np.zeros(shape=(12,len(mesh.x2)))
                if cocco_path.is_file():
                    Cocco_Light_limiter = np.zeros(shape=(12,len(mesh.x2)))
                    Cocco_CO2_limiter   = np.zeros(shape=(12,len(mesh.x2)))
                    Fecocco             = np.zeros(shape=(12,len(mesh.x2)))
                    Cocco_qlimFac1      = np.zeros(shape=(12,len(mesh.x2),2))
                    Cocco_qlimFac       = np.zeros(shape=(12,len(mesh.x2)))
                    Cocco_dq            = np.zeros(shape=(12,len(mesh.x2)))
                    Cocco_recom_limiter = np.zeros(shape=(12,len(mesh.x2)))
                    T_cocco             = np.zeros(shape=(12,len(mesh.x2)))
                    Pmax_cocco          = np.zeros(shape=(12,len(mesh.x2)))
                    months_limcocco     = np.zeros(shape=(12,len(mesh.x2)))
                if Tfunction == 'New_functions':
                    Phaeo_Light_limiter = np.zeros(shape=(12,len(mesh.x2)))
                    Phaeo_CO2_limiter   = np.zeros(shape=(12,len(mesh.x2)))
                    Fephaeo             = np.zeros(shape=(12,len(mesh.x2)))
                    Phaeo_qlimFac1      = np.zeros(shape=(12,len(mesh.x2),2))
                    Phaeo_qlimFac       = np.zeros(shape=(12,len(mesh.x2)))
                    Phaeo_dq            = np.zeros(shape=(12,len(mesh.x2)))
                    Phaeo_recom_limiter = np.zeros(shape=(12,len(mesh.x2)))
                    T_phaeo             = np.zeros(shape=(12,len(mesh.x2)))
                    Pmax_phaeo          = np.zeros(shape=(12,len(mesh.x2)))
                    months_limphaeo     = np.zeros(shape=(12,len(mesh.x2)))


                DIN = pf.get_data(self.resultpath, "DIN", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DIN = DIN.groupby(time='month').mean().compute()
            
                DSi = pf.get_data(self.resultpath, "DSi", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DSi = DSi.groupby(time='month').mean().compute()
            
                DFe = pf.get_data(self.resultpath, "DFe", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DFe = DFe.groupby(time='month').mean().compute()
            
                HCO3 = pf.get_data(self.resultpath, "HCO3", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                HCO3 = HCO3.groupby(time='month').mean().compute()
            
                CO2 = pf.get_data(self.resultpath, "CO2", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                CO2 = CO2.groupby(time='month').mean().compute()
            
                pH = pf.get_data(self.resultpath, "pH", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                pH = pH.groupby(time='month').mean().compute()
            
                PhyC = pf.get_data(self.resultpath, "PhyC", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                PhyC = PhyC.groupby(time='month').mean().compute()
            
                PhyN = pf.get_data(self.resultpath, "PhyN", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                PhyN = PhyN.groupby(time='month').mean().compute()
            
                PhyChl = pf.get_data(self.resultpath, "PhyChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                PhyChl = PhyChl.groupby(time='month').mean().compute()
            
                DiaSi = pf.get_data(self.resultpath, "DiaSi", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DiaSi = DiaSi.groupby(time='month').mean().compute()
            
                DiaC = pf.get_data(self.resultpath, "DiaC", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DiaC = DiaC.groupby(time='month').mean().compute()

                DiaN = pf.get_data(self.resultpath, "DiaN", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DiaN = DiaN.groupby(time='month').mean().compute()
            
                DiaChl = pf.get_data(self.resultpath, "DiaChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                DiaChl = DiaChl.groupby(time='month').mean().compute()
            
                if cocco_path.is_file():
                    
                    CoccoC = pf.get_data(self.resultpath, "CoccoC", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    CoccoC = CoccoC.groupby(time='month').mean().compute()

                    CoccoN = pf.get_data(self.resultpath, "CoccoN", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    CoccoN = CoccoN.groupby(time='month').mean().compute()
                    
                    CoccoChl = pf.get_data(self.resultpath, "CoccoChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    CoccoChl = CoccoChl.groupby(time='month').mean().compute()
                    
                temp = pf.get_data(self.resultpath, "temp", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                temp = temp.groupby(time='month').mean().compute()
            
                PAR = pf.get_data(self.resultpath, "PAR", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                PAR = PAR.groupby(time='month').mean().compute()
            
                if Tfunction == 'New_functions':
                    
                    PhaeoC = pf.get_data(self.resultpath, "PhaeoC", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    PhaeoC = PhaeoC.groupby(time='month').mean().compute()
                    
                    PhaeoN = pf.get_data(self.resultpath, "PhaeoN", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    PhaeoN = PhaeoN.groupby(time='month').mean().compute()
                    
                    PhaeoChl = pf.get_data(self.resultpath, "PhaeoChl", years, mesh, 
                                   how=None, compute=False, runid=self.runname, silent=True)
                    PhaeoChl = PhaeoChl.groupby(time='month').mean().compute()
                    
                print(np.shape(DIN))

                DIN  = DIN[:,:,0]
                DSi  = DSi[:,:,0]
                DFe  = DFe[:,:,0]
                HCO3 = HCO3[:,:,0]
                CO2  = CO2[:,:,0]
                pH   = pH[:,:,0]
                PhyC = PhyC[:,:,0]
                PhyN = PhyN[:,:,0]
                PhyChl = PhyChl[:,:,0]
                DiaSi = DiaSi[:,:,0]
                DiaC = DiaC[:,:,0]
                DiaN = DiaN[:,:,0]
                DiaChl = DiaChl[:,:,0]
                if cocco_path.is_file():
                    CoccoC   = CoccoC[:,:,0]
                    CoccoN   = CoccoN[:,:,0]
                    CoccoChl = CoccoChl[:,:,0]
                temp = temp[:,:,0]
                PAR = PAR[:,:,0]
                if Tfunction == 'New_functions':
                    PhaeoC   = PhaeoC[:,:,0]
                    PhaeoN   = PhaeoN[:,:,0]
                    PhaeoChl = PhaeoChl[:,:,0]
                print(np.shape(DIN))

                for m in range(0,12):
                    DIN2D  = DIN[m,:]
                    DSi2D  = DSi[m,:]
                    DFe2D  = DFe[m,:]
                    HCO32D = HCO3[m,:]
                    CO22D  = CO2[m,:]
                    pH2D   = pH[m,:]
                    phyc2D = PhyC[m,:]
                    phyn2D = PhyN[m,:]
                    phychl2D = PhyChl[m,:]
                    diac2D = DiaC[m,:]
                    dian2D = DiaN[m,:]
                    diasi2D = DiaSi[m,:]
                    diachl2D = DiaChl[m,:]
                    if cocco_path.is_file():
                        coccoc2D = CoccoC[m,:]
                        coccon2D = CoccoN[m,:]
                        coccochl2D = CoccoChl[m,:]
                    par2d = PAR[m,:]
                    T2d = temp[m,:]
                    if Tfunction == 'New_functions':
                        phaeoc2D = PhaeoC[m,:]
                        phaeon2D = PhaeoN[m,:]
                        phaeochl2D = PhaeoChl[m,:]

                    # Quotas
                    phychl2c = phychl2D/phyc2D 
                    phyn2c   = phyn2D/phyc2D
                    diachl2c = diachl2D/diac2D
                    dian2c   = dian2D/diac2D
                    diasi2c  = diasi2D/diac2D
                    
                    if cocco_path.is_file():
                        coccochl2c = coccochl2D/coccoc2D
                        coccon2c   = coccon2D/coccoc2D

                    if Tfunction == 'New_functions':
                        phaeochl2c = phaeochl2D/phaeoc2D
                        phaeon2c   = phaeon2D/phaeoc2D

                    # Temperature dependence
                    if Tfunction == 'Arrhenius':
                        T2d_loc = 1./(T2d + C2K)
                        T_phy   = np.exp( -Ae * (T2d_loc - rTref))
                        T_dia   = np.exp( -Ae * (T2d_loc - rTref))
                        
                        if cocco_path.is_file():
                            T_cocco = max(0.1419 * T2d**0.8151,1e-15)
                            
                    if Tfunction == 'New_functions':
                        T_phy   = np.exp(base_phy + exp_phy * T2d)
                        T_dia   = np.exp(base_dia + exp_dia * T2d)
                        T_cocco = np.exp(base_cocco + exp_cocco * T2d)
                        T_phaeo = np.exp(base_phaeo + exp_phaeo * T2d)
                        

                    # Nutrient growth limitation
                    Phy_dq[m,:]              = NCmin - phyn2c
                    Phy_recom_limiter[m,:]   = 1.0 - np.exp( -NMinSlope*( abs(Phy_dq[m,:])-Phy_dq[m,:] )**2)

                    Dia_dq[m,:]              = NCmin - dian2c
                    Dia_recom_limiterN[m,:]  = 1.0 - np.exp( -NMinSlope*( abs(Dia_dq[m,:])-Dia_dq[m,:] )**2)
                    Dia_dq[m,:]              = SiCmin - diasi2c;
                    Dia_recom_limiterSi[m,:] = 1.0 - np.exp( -SiMinSlope*( abs(Dia_dq[m,:])-Dia_dq[m,:] )**2)
                    
                    if cocco_path.is_file():
                        Cocco_dq[m,:]              = NCmin - coccon2c
                        Cocco_recom_limiter[m,:]   = 1.0 - np.exp( -NMinSlope*( abs(Cocco_dq[m,:])-Cocco_dq[m,:] )**2)
                        
                    if Tfunction == 'New_functions':
                        Phaeo_dq[m,:]              = NCmin - phaeon2c
                        Phaeo_recom_limiter[m,:]   = 1.0 - np.exp( -NMinSlope*( abs(Phaeo_dq[m,:])-Phaeo_dq[m,:] )**2)
                    

                    # Most limiting nutrient 
                    Fephy[m,:]        = DFe2D/(DFe2D+k_fe_phy)
                    Phy_qlimFac1[m,:] = np.column_stack((Phy_recom_limiter[m,:],Fephy[m,:]))
                    Phy_qlimFac[m,:]  = Phy_qlimFac1[m,:].min(axis=1)

                    Fedia[m,:]        = DFe2D/(DFe2D+k_fe_dia)
                    Dia_qlimFac1[m,:] = np.column_stack((Dia_recom_limiterN[m,:],Dia_recom_limiterSi[m,:],Fedia[m,:]))
                    Dia_qlimFac[m,:]  = Dia_qlimFac1[m,:].min(axis=1)
                    
                    if cocco_path.is_file():
                        Fecocco[m,:]        = DFe2D/(DFe2D+k_fe_cocco)
                        Cocco_qlimFac1[m,:] = np.column_stack((Cocco_recom_limiter[m,:],Fecocco[m,:]))
                        Cocco_qlimFac[m,:]  = Cocco_qlimFac1[m,:].min(axis=1)
                        
                    if Tfunction == 'New_functions':
                        Fephaeo[m,:]        = DFe2D/(DFe2D+k_fe_phaeo)
                        Phaeo_qlimFac1[m,:] = np.column_stack((Phaeo_recom_limiter[m,:],Fephaeo[m,:]))
                        Phaeo_qlimFac[m,:]  = Phaeo_qlimFac1[m,:].min(axis=1)
                        
                    # pmax
                    if Tfunction == 'Arrhenius':
                        Pmax_phy[m,:] = Pcm * T_phy * Phy_qlimFac[m,:] 
                        Pmax_dia[m,:] = Pcm_d * T_dia * Dia_qlimFac[m,:]
                        
                        if cocco_path.is_file():
                            Pmax_cocco[m,:] = Pcm_c * T_cocco * Cocco_qlimFac[m,:]
                            
                    elif Tfunction == 'New_functions':
                        Pmax_phy[m,:]   = T_phy * Phy_qlimFac[m,:] 
                        Pmax_dia[m,:]   = T_dia * Dia_qlimFac[m,:]
                        Pmax_cocco[m,:] = T_cocco * Cocco_qlimFac[m,:]
                        Pmax_phaeo[m,:] = T_phaeo * Phaeo_qlimFac[m,:]

                    # Light limitation
                    Phy_Light_limiter = 1 - np.exp((-alfa_phy * phychl2c * par2d)/Pmax_phy)
                    Dia_Light_limiter = 1 - np.exp((-alfa_dia * diachl2c * par2d)/Pmax_dia)
                    if cocco_path.is_file():
                        Cocco_Light_limiter = 1 - np.exp((-alfa_cocco * coccochl2c * par2d)/Pmax_cocco)
                    if Tfunction == 'New_functions':
                        Phaeo_Light_limiter = 1 - np.exp((-alfa_phaeo * phaeochl2c * par2d)/Pmax_phaeo)
                                                
                    # Carbonate system limitation
                    Phy_CO2_limiter[m,:] = a_co2_phy * HCO32D * Cunits / (b_co2_phy + HCO32D * Cunits) - np.exp(-c_co2_phy * CO22D * Cunits) - d_co2_phy * 10.**(-pH2D)
                    Phy_CO2_limiter[m,:] = np.clip(Phy_CO2_limiter[m,:], 0.0, 3.0)                                                             
                    
                    Dia_CO2_limiter[m,:] = a_co2_dia * HCO32D * Cunits / (b_co2_dia + HCO32D * Cunits) - np.exp(-c_co2_dia * CO22D * Cunits) - d_co2_dia * 10.**(-pH2D)
                    Dia_CO2_limiter[m,:] = np.clip(Dia_CO2_limiter[m,:], 0.0, 3.0)                                                              
                    
                    if cocco_path.is_file():
                        Cocco_CO2_limiter[m,:] = a_co2_cocco * HCO32D * Cunits / (b_co2_cocco + HCO32D * Cunits) - np.exp(-c_co2_cocco * CO22D * Cunits) - d_co2_cocco * 10.**(-pH2D)
                        Cocco_CO2_limiter[m,:] = np.clip(Cocco_CO2_limiter[m,:], 0.0, 3.0)                                                             
                        
                    if Tfunction == 'New_functions':
                        Phaeo_CO2_limiter[m,:] = a_co2_phaeo * HCO32D * Cunits / (b_co2_phaeo + HCO32D * Cunits) - np.exp(-c_co2_phaeo * CO22D * Cunits) - d_co2_phaeo * 10.**(-pH2D)
                        Phaeo_CO2_limiter[m,:] = np.clip(Phaeo_CO2_limiter[m,:], 0.0, 3.0)                                                                
                        
                        
                    # Most limiting factors found for nano
                    # 1 = Fe, 2 = DIN, 3 = Light
                    lim_phy=np.column_stack((Fephy[m,:],Phy_recom_limiter[m,:],Phy_Light_limiter[m,:],Phy_CO2_limiter[m,:]))
                    #limphy1=lim.argmin(axis=1)
                    months_limphy[m, :] = np.choose(lim_phy.argmin(axis=1), [1, 2, 4, 5])

                    # Most limiting factors found for dia
                    # 1 = Fe, 2 = DIN, 3 = DSi, 4 = Light, 5 = CO2
                    lim_dia=np.column_stack((Fedia[m,:],Dia_recom_limiterN[m,:],Dia_recom_limiterSi[m,:],Dia_Light_limiter[m,:],Dia_CO2_limiter[m,:]))
                    #limdia1=lim.argmin(axis=1)
                    months_limdia[m, :] = np.choose(lim_dia.argmin(axis=1), [1, 2, 3, 4, 5])
                    
                    if cocco_path.is_file():
                    # Most limiting factors found for cocco
                    # 1= Fe, 2 = DIN, 4 = Light, 5 = CO2
                        lim_cocco=np.column_stack((Fecocco[m,:],Cocco_recom_limiter[m,:],Cocco_Light_limiter[m,:],Cocco_CO2_limiter[m,:]))
                        #limcocco1=lim.argmin(axis=1)
                        months_limcocco[m, :] = np.choose(lim_cocco.argmin(axis=1), [1, 2, 4, 5])
                        
                    if Tfunction == 'New_functions':
                    # Most limiting factors found for Phaeo
                    # 1 = Fe, 2 = DIN, 4 = Light, 5 = CO2
                        lim_phaeo=np.column_stack((Fephaeo[m,:],Phaeo_recom_limiter[m,:],Phaeo_Light_limiter[m,:],Phaeo_CO2_limiter[m,:]))
                        #limphaeo1=lim.argmin(axis=1)
                        months_limphaeo[m, :] = np.choose(lim_phaeo.argmin(axis=1), [1, 2, 4, 5])
                    
                    
                if self.plotting:
                    pfts_to_plot = [
                        ('Diatoms', months_limdia),
                        ('Small Phytoplankton', months_limphy)
                    ]
                    if cocco_path.is_file():
                        pfts_to_plot.append(('Coccolithophore', months_limcocco))
                    if Tfunction == 'New_functions':
                        pfts_to_plot.append(('Polar Phaeocystis', months_limphaeo))

                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    levels = np.arange(0.5, 6.5, 1) 

                    for pft_name, pft_data in pfts_to_plot:
                        fig, axes = plt.subplots(3, 4, figsize=(16, 12), 
                                                 subplot_kw=dict(projection=self.mapproj),
                                                 constrained_layout=True)

                        axes_flat = axes.flatten()
                        last_f = None

                        for m in range(12):
                            ax = axes_flat[m]
                            last_f = pf.subplot(mesh, fig, ax, pft_data[m, :], 
                                                levels=levels,
                                                units='',  
                                                mapproj=self.mapproj, 
                                                cmap=self.cmap,
                                                cmap_extension='neither',
                                                titles=month_names[m],
                                                box=box)

                        fig.suptitle(f'{pft_name} Limiting Factors', fontsize=22, weight='bold')

                        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
                        cbar = fig.colorbar(last_f, cax=cbar_ax, orientation='horizontal', ticks=[1, 2, 3, 4, 5])
                        cbar.ax.tick_params(labelsize=14)
                        cbar.ax.set_xticklabels(['Iron (Fe)', 'DIN', 'DSi (Silicate)', 'Light', 'CO2'])

                        fig.subplots_adjust(bottom=0.12)
                        
                    
                        # fig export  -------------------------------------------------------------------------------------
                        if(self.savefig==True):                
                            plt.savefig(self.savepath+self.runname+'_'+'LIMFact_monthly'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                                    dpi = 300, bbox_inches='tight')
                            plt.savefig(self.savepath+self.runname+'_'+'LIMFact_monthly'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                                    bbox_inches='tight')
                        plt.show(block=False)
                    
        elif frequency == 'yearly':
                    print('yearly frequency selected')
                    Phy_Light_limiter = np.zeros(shape=(len(mesh.x2)))
                    Dia_Light_limiter = np.zeros(shape=(len(mesh.x2)))
                    if cocco_path.is_file():
                        Cocco_Light_limiter = np.zeros(shape=(len(mesh.x2)))
                    if Tfunction == 'New_functions':
                        Phaeo_Light_limiter = np.zeros(shape=(len(mesh.x2)))

                    DIN = pf.get_data(self.resultpath, "DIN", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DSi = pf.get_data(self.resultpath, "DSi", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DFe = pf.get_data(self.resultpath, "DFe", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    HCO3 = pf.get_data(self.resultpath, "HCO3", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    CO2 = pf.get_data(self.resultpath, "CO2", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    pH = pf.get_data(self.resultpath, "pH", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    PhyC = pf.get_data(self.resultpath, "PhyC", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    PhyN = pf.get_data(self.resultpath, "PhyN", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    PhyChl = pf.get_data(self.resultpath, "PhyChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DiaSi = pf.get_data(self.resultpath, "DiaSi", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DiaC = pf.get_data(self.resultpath, "DiaC", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DiaN = pf.get_data(self.resultpath, "DiaN", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    DiaChl = pf.get_data(self.resultpath, "DiaChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    if cocco_path.is_file():
                    
                        CoccoC = pf.get_data(self.resultpath, "CoccoC", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                        CoccoN = pf.get_data(self.resultpath, "CoccoN", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                        CoccoChl = pf.get_data(self.resultpath, "CoccoChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    temp = pf.get_data(self.resultpath, "temp", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                    PAR = pf.get_data(self.resultpath, "PAR", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)
                    
                    if Tfunction == 'New_functions':

                        PhaeoC = pf.get_data(self.resultpath, "PhaeoC", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                        PhaeoN = pf.get_data(self.resultpath, "PhaeoN", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)

                        PhaeoChl = pf.get_data(self.resultpath, "PhaeoChl", years, mesh, 
                                       how="mean", compute=True, runid=self.runname, silent=True)


                    DIN2D  = DIN[:,0]
                    DSi2D  = DSi[:,0]
                    DFe2D  = DFe[:,0]
                    HCO32D = HCO3[:,0]
                    CO22D  = CO2[:,0]
                    pH2D   = pH[:,0]
                    phyc2D = PhyC[:,0]
                    phyn2D = PhyN[:,0]
                    phychl2D = PhyChl[:,0]
                    diasi2D = DiaSi[:,0]
                    diac2D = DiaC[:,0]
                    dian2D = DiaN[:,0]
                    diachl2D = DiaChl[:,0]
                    if cocco_path.is_file():
                        coccoc2D = CoccoC[:,0]
                        coccon2D = CoccoN[:,0]
                        coccochl2D = CoccoChl[:,0]
                    T2d = temp[:,0]
                    par2d = PAR[:,0]
                    if Tfunction == 'New_functions':
                        phaeoc2D = PhaeoC[:,0]
                        phaeon2D = PhaeoN[:,0]
                        phaeochl2D = PhaeoChl[:,0]


                    # Quotas
                    phychl2c = phychl2D/phyc2D 
                    phyn2c   = phyn2D/phyc2D
                    diachl2c = diachl2D/diac2D
                    dian2c   = dian2D/diac2D
                    diasi2c  = diasi2D/diac2D
                    
                    if cocco_path.is_file():
                        coccochl2c = coccochl2D/coccoc2D
                        coccon2c   = coccon2D/coccoc2D

                    if Tfunction == 'New_functions':
                        phaeochl2c = phaeochl2D/phaeoc2D
                        phaeon2c   = phaeon2D/phaeoc2D

                    # Temperature dependence
                    if Tfunction == 'Arrhenius':
                        T2d_loc = 1./(T2d + C2K)
                        T_phy   = np.exp( -Ae * (T2d_loc - rTref))
                        T_dia   = np.exp( -Ae * (T2d_loc - rTref))
                        
                        if cocco_path.is_file():
                            T_cocco = max(0.1419 * T2d**0.8151,1e-15)
                            
                    if Tfunction == 'New_functions':
                        T_phy   = np.exp(base_phy + exp_phy * T2d)
                        T_dia   = np.exp(base_dia + exp_dia * T2d)
                        T_cocco = np.exp(base_cocco + exp_cocco * T2d)
                        T_phaeo = np.exp(base_phaeo + exp_phaeo * T2d)

                    # Nutrient growth limitation
                    Phy_dq              = NCmin - phyn2c
                    Phy_recom_limiter   = 1.0 - np.exp( -NMinSlope*( abs(Phy_dq)-Phy_dq )**2)

                    Dia_dq              = NCmin - dian2c
                    Dia_recom_limiterN  = 1.0 - np.exp( -NMinSlope*( abs(Dia_dq)-Dia_dq )**2)
                    Dia_dq              = SiCmin - diasi2c;
                    Dia_recom_limiterSi = 1.0 - np.exp( -SiMinSlope*( abs(Dia_dq)-Dia_dq )**2)
                    
                    if cocco_path.is_file():
                        Cocco_dq              = NCmin - coccon2c
                        Cocco_recom_limiter   = 1.0 - np.exp( -NMinSlope*( abs(Cocco_dq)-Cocco_dq )**2)
                        
                    if Tfunction == 'New_functions':
                        Phaeo_dq              = NCmin - phaeon2c
                        Phaeo_recom_limiter   = 1.0 - np.exp( -NMinSlope*( abs(Phaeo_dq)-Phaeo_dq )**2)
                
                    
                    # Most limiting nutrient 
                    Fephy       = DFe2D/(DFe2D+k_fe_phy)
                    Phy_qlimFac = np.column_stack((Phy_recom_limiter,Fephy))
                    Phy_qlimFac = Phy_qlimFac.min(axis=1)

                    Fedia       = DFe2D/(DFe2D+k_fe_dia)
                    Dia_qlimFac = np.column_stack((Dia_recom_limiterN,Dia_recom_limiterSi,Fedia))
                    Dia_qlimFac = Dia_qlimFac.min(axis=1)
                    
                    if cocco_path.is_file():
                        Fecocco       = DFe2D/(DFe2D+k_fe_cocco)
                        Cocco_qlimFac = np.column_stack((Cocco_recom_limiter,Fecocco))
                        Cocco_qlimFac = Cocco_qlimFac.min(axis=1)
                        
                    if Tfunction == 'New_functions':
                        Fephaeo       = DFe2D/(DFe2D+k_fe_phaeo)
                        Phaeo_qlimFac = np.column_stack((Phaeo_recom_limiter,Fephaeo))
                        Phaeo_qlimFac = Phaeo_qlimFac.min(axis=1)
                        
                    # pmax
                    if Tfunction == 'Arrhenius':
                        Pmax_phy = Pcm * T_phy * Phy_qlimFac 
                        Pmax_dia = Pcm_d * T_dia * Dia_qlimFac
                        
                        if cocco_path.is_file():
                            Pmax_cocco = Pcm_c * T_cocco * Cocco_qlimFac
                            
                    elif Tfunction == 'New_functions':
                        Pmax_phy   = T_phy * Phy_qlimFac 
                        Pmax_dia   = T_dia * Dia_qlimFac
                        Pmax_cocco = T_cocco * Cocco_qlimFac
                        Pmax_phaeo = T_phaeo * Phaeo_qlimFac

                    # Light limitation
                    Phy_Light_limiter = 1 - np.exp((-alfa_phy * phychl2c * par2d)/Pmax_phy)
                    Dia_Light_limiter = 1 - np.exp((-alfa_dia * diachl2c * par2d)/Pmax_dia)
                    if cocco_path.is_file():
                        Cocco_Light_limiter = 1 - np.exp((-alfa_cocco * coccochl2c * par2d)/Pmax_cocco)
                    if Tfunction == 'New_functions':
                        Phaeo_Light_limiter = 1 - np.exp((-alfa_phaeo * phaeochl2c * par2d)/Pmax_phaeo)
                    
                    
                    # Carbonate system limitation
                    Phy_CO2_limiter = a_co2_phy * HCO32D * Cunits / (b_co2_phy + HCO32D * Cunits) - np.exp(-c_co2_phy * CO22D * Cunits) - d_co2_phy * 10.**(-pH2D)
                    Phy_CO2_limiter = np.clip(Phy_CO2_limiter, 0.0, 3.0)                                                            
                    
                    Dia_CO2_limiter = a_co2_dia * HCO32D * Cunits / (b_co2_dia + HCO32D * Cunits) - np.exp(-c_co2_dia * CO22D * Cunits) - d_co2_dia * 10.**(-pH2D)
                    Dia_CO2_limiter = np.clip(Dia_CO2_limiter, 0.0, 3.0)                                                                
                    
                    if cocco_path.is_file():
                        Cocco_CO2_limiter = a_co2_cocco * HCO32D * Cunits / (b_co2_cocco + HCO32D * Cunits) - np.exp(-c_co2_cocco * CO22D * Cunits) - d_co2_cocco * 10.**(-pH2D)
                        Cocco_CO2_limiter = np.clip(Cocco_CO2_limiter, 0.0, 3.0)                                                              
                        
                    if Tfunction == 'New_functions':
                        Phaeo_CO2_limiter = a_co2_phaeo * HCO32D * Cunits / (b_co2_phaeo + HCO32D * Cunits) - np.exp(-c_co2_phaeo * CO22D * Cunits) - d_co2_phaeo * 10.**(-pH2D)
                        Phaeo_CO2_limiter = np.clip(Phaeo_CO2_limiter, 0.0, 3.0)                                                             
  
                    
                    # Most limiting factors found for nano
                    # 0 = Fe, 1 = DIN, 2 = Light
                    lim=np.column_stack((Fephy,Phy_recom_limiter,Phy_Light_limiter,Phy_CO2_limiter))
                    limphy1=lim.argmin(axis=1)

                    # Most limiting factors found for dia
                    # 0 = Fe, 1 = DIN, 2 = DSi, 3 = Light
                    lim=np.column_stack((Fedia,Dia_recom_limiterN,Dia_recom_limiterSi,Dia_Light_limiter,Dia_CO2_limiter))
                    limdia1=lim.argmin(axis=1)
                    
                    if cocco_path.is_file():
                    # Most limiting factors found for cocco
                    # 0 = Fe, 1 = DIN, 2 = Light
                        lim=np.column_stack((Fecocco,Cocco_recom_limiter,Cocco_Light_limiter,Cocco_CO2_limiter))
                        limcocco1=lim.argmin(axis=1)
                        
                    if Tfunction == 'New_functions':
                    # Most limiting factors found for Phaeo
                    # 0 = Fe, 1 = DIN, 2 = Light
                        lim=np.column_stack((Fephaeo,Phaeo_recom_limiter,Phaeo_Light_limiter,Phaeo_CO2_limiter))
                        limphaeo1=lim.argmin(axis=1)
                    
                    
                    limphy1 = (np.array(limphy1, dtype = float) + 1) # .astype('Float32')
                    limdia1 = (np.array(limdia1, dtype = float) + 1) # .astype('Float32')
                    if cocco_path.is_file():
                        limcocco1 = (np.array(limcocco1, dtype = float) + 1) # .astype('Float32')
                    if Tfunction == 'New_functions':
                        limphaeo1 = (np.array(limphaeo1, dtype = float) + 1) # .astype('Float32')
        
                    
                    # Find where none is limiting for hatching (>0.5)
                    nolimphy = np.random.rand(*np.shape(limphy1))
                    nolimdia = np.random.rand(*np.shape(limdia1))
                    
                    nolimphy[np.where((Fephy > 0.5) & (Phy_recom_limiter > 0.5) & (Phy_Light_limiter > 0.5) & (Phy_CO2_limiter > 0.5))] = -1
                    nolimdia[np.where((Fedia > 0.5) & (Dia_recom_limiterN > 0.5) & (Dia_recom_limiterSi > 0.5) & (Dia_Light_limiter > 0.5) & (Dia_CO2_limiter > 0.5))] = -1
                    
                    if cocco_path.is_file():
                        nolimcocco = np.random.rand(*np.shape(limcocco1))
                        nolimcocco[np.where((Fecocco > 0.5) & (Cocco_recom_limiter > 0.5) & (Cocco_Light_limiter > 0.5) & (Cocco_CO2_limiter > 0.5))] = -1
                        
                    if Tfunction == 'New_functions':
                        nolimphaeo = np.random.rand(*np.shape(limphaeo1))
                        nolimphaeo[np.where((Fephaeo > 0.5) & (Phaeo_recom_limiter > 0.5) & (Phaeo_Light_limiter > 0.5) & (Phaeo_CO2_limiter > 0.5))] = -1
                    
                    # Plotting
                    if True:
                        self.limphy = limphy1
                        self.limdia   = limdia1
                        self.nolimdia = nolimdia
                        self.Phy_Light_limiter = Phy_Light_limiter
                        if cocco_path.is_file():
                            self.limcocco   = limcocco1
                        if Tfunction == 'New_functions':
                            self.limphaeo   = limphaeo1
                        
        
                    if self.plotting:    
                        plots = ["A", "B"]
                        if cocco_path.is_file():
                            plots.append("C")
                        if Tfunction == 'New_functions':
                            plots.append("D")
                        mosaic_layout = "\n".join(plots)
                    
                        fig = plt.figure(figsize=(8, 5 * len(plots)), constrained_layout=True)
                        axes = fig.subplot_mosaic(
                                    mosaic_layout,
                                    gridspec_kw={'hspace': 0.01, 'wspace': 0.01}, 
                                    subplot_kw=dict(projection=self.mapproj))
                        fig.get_layout_engine().set(w_pad=0.2)
                        
                        
                        m1 = axes['A']
                        levels = np.arange(0,25,1)
                        f1 = pf.subplot(mesh, fig, m1, self.limdia,
                                    levels = np.arange(0.5,6.5,1),
                                    units='', 
                                    mapproj=self.mapproj, # robinson projection takes more time!
                                    cmap = self.cmap,
                                    cmap_extension='neither',
                                    titles='Diatoms limiting factor',
                                    box = box,
                                   )
                        
                        f1a = pf.subhatch(mesh, fig, m1, nolimdia)
                        
                        #cbar0_ax = fig.add_axes([0.25, 0.54, 0.6, 0.02])
                        cbar0 = fig.colorbar(f1,
                                        #cax = cbar0_ax, 
                                        ax=m1,
                                        orientation = 'horizontal',
                                        fraction=0.046, pad=0.04, ticks=[1,2,3,4,5]) 
                        #cbar0.set_label('Limiting Factor', fontsize=18)
                        cbar0.ax.tick_params(labelsize=18)
                        cbar0.ax.set_xticklabels(['Fe', 'DIN', 'DSi', 'Light', 'CO2']) 
                        
                        m2 = axes['B']
                        f2 = pf.subplot(mesh, fig, m2, self.limphy, 
                                    levels = np.arange(0.5,5.5,1),
                                    units='', 
                                    mapproj=self.mapproj, # robinson projection takes more time!
                                    cmap = self.cmap,
                                    cmap_extension='neither',
                                    titles='Small phytoplankton limiting factor',
                                    box = box,
                                   )
                        
                        f2a = pf.subhatch(mesh, fig, m2, nolimphy)
                        
                        #cbar1_ax = fig.add_axes([0.25, 0.04, 0.6, 0.02])
                        cbar1 = fig.colorbar(f2,
                                        #cax = cbar1_ax, 
                                        ax = m2, 
                                        orientation = 'horizontal',
                                        fraction=0.046, pad=0.04, ticks=[1,2,3,4]) 
                        #cbar1.set_label('Limiting Factor', fontsize=18)
                        cbar1.ax.tick_params(labelsize=18)
                        cbar1.ax.set_xticklabels(['Fe', 'DIN', 'Light', 'CO2'])  
                        
                        if cocco_path.is_file():
                            m3 = axes['C']
                            f3 = pf.subplot(mesh, fig, m3, self.limcocco, 
                                        levels = np.arange(0.5,5.5,1),
                                        units='', 
                                        mapproj=self.mapproj, # robinson projection takes more time!
                                        cmap = self.cmap,
                                        cmap_extension='neither',
                                        titles='Coccolithophores limiting factor',
                                        box = box,
                                       )

                            f3a = pf.subhatch(mesh, fig, m3, nolimcocco)
                            
                            #cbar3_ax = fig.add_axes([0.25, -0.46, 0.6, 0.02])
                            cbar3 = fig.colorbar(f3,
                                            #cax = cbar3_ax, 
                                            ax = m3, 
                                            orientation = 'horizontal',
                                            fraction=0.046, pad=0.04, ticks=[1,2,3,4]) 
                            #cbar1.set_label('Limiting Factor', fontsize=18)
                            cbar3.ax.tick_params(labelsize=18)
                            cbar3.ax.set_xticklabels(['Fe', 'DIN', 'Light', 'CO2'])  
                        
                        if Tfunction == 'New_functions':
                            m4 = axes['D']
                            f4 = pf.subplot(mesh, fig, m4, self.limphaeo, 
                                        levels = np.arange(0.5,5.5,1),
                                        units='', 
                                        mapproj=self.mapproj, # robinson projection takes more time!
                                        cmap = self.cmap,
                                        cmap_extension='neither',
                                        titles='Polar Phaeocystis limiting factor',
                                        box = box,
                                       )

                            f4a = pf.subhatch(mesh, fig, m4, nolimphaeo)
                            
                            #cbar4_ax = fig.add_axes([0.25, -0.96, 0.6, 0.02])
                            cbar4 = fig.colorbar(f4,
                                            #cax = cbar4_ax, 
                                            ax = m4, 
                                            orientation = 'horizontal',
                                            fraction=0.046, pad=0.04, ticks=[1,2,3,4]) 
                            #cbar1.set_label('Limiting Factor', fontsize=18)
                            cbar4.ax.tick_params(labelsize=18)
                            cbar4.ax.set_xticklabels(['Fe', 'DIN', 'Light', 'CO2'])  
                        
                        
                        m1.text(-0.1, 1.05, 'A', transform=m1.transAxes,
                                    size=30, weight='bold')
                        m2.text(-0.1, 1.05, 'B', transform=m2.transAxes,
                                    size=30, weight='bold')
                        if cocco_path.is_file():
                            m3.text(-0.1, 1.05, 'C', transform=m3.transAxes,
                                    size=30, weight='bold')
                        if Tfunction == 'New_functions':
                            m4.text(-0.1, 1.05, 'D', transform=m4.transAxes,
                                    size=30, weight='bold')
            
                       # last_f = f2
                       # if cocco_path.is_file():
                       #     last_f = f3
                       # if Tfunction == 'New_functions':
                       #     last_f = f4
                       # 
                       # fig.subplots_adjust(bottom=0.2)
                        
                        
                            
                        
                        # fig export  -------------------------------------------------------------------------------------
                        if(self.savefig==True):                
                            plt.savefig(self.savepath+self.runname+'_'+'LIMFact'+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                                    dpi = 300, bbox_inches='tight')
                            plt.savefig(self.savepath+self.runname+'_'+'LIMFact'+'_'+str(years[0])+'to'+str(years[-1])+'.pdf', 
                                    bbox_inches='tight')
                        plt.show(block=False)
                    
        else:
                print('no frequency selected')