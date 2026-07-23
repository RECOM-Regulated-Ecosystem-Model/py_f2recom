"""
Visualization tools for bar plots in REcoM model output.
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

class plot_bars_maredat_sina:
    '''
    class Bio_Maredat_comp(runname,resultpath,savepath,meshpath,ncfileMaredat,FT,first_year,last_year,
                 mapproj='pc',savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
    '''
    
    def __init__(self,resultpath,savepath,mesh,meshpath,ncfileMaredat,FT,first_year,last_year,
                 mapproj='pc',runid='fesom',
                 savefig=False,output=False,plotting=True,verbose=False,Taylor=True):

        self.runid = runid
        self.resultpath = resultpath
        self.savepath = savepath
        self.mesh = mesh
        self.meshpath = meshpath
        self.fyear = first_year
        self.lyear = last_year
        self.mapproj = mapproj
        self.savefig = savefig
        self.ncfileMaredat = ncfileMaredat
        self.FT = FT
        self.verbose = verbose
        self.output = output
        self.plotting = plotting
        self.Taylor = Taylor
        
        import warnings
        warnings.filterwarnings('ignore')
        
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

            
# load Maredat -------------------------------------------------------------------------------
    # for Phaeocystis Maredat: Extract Arctic and Antarctic Data
        if ncfileMaredat == '/albedo/home/simuel001/master_thesis/py_f2recom_4phy/eval_maredat_phaeo/PANGAEA/MarEDat20120424Phaeocystis.nc':
            

            ncfileMaredat = xr.open_dataset(self.ncfileMaredat, decode_times=False)
            
            mask = ((ncfileMaredat.LATITUDE < 45) & (ncfileMaredat.LATITUDE > -45))

            filtered_data = ncfileMaredat.where(~mask, drop=False)

            # saving maredat data with extracted latitude data:
            
            path = '/albedo/home/simuel001/master_thesis/py_f2recom_4phy/eval_maredat_phaeo/PANGAEA/MarEDat20120424Phaeocystis_filtered.nc'
            filtered_data.to_netcdf(path=path)

            # continue with extracted data: 
            lat_maredat, lon_maredat, maredat_layered_sum, maredat_layered_mean = load_maredat(path)

        else:
            lat_maredat, lon_maredat, maredat_layered_sum, maredat_layered_mean = load_maredat(self.ncfileMaredat)

            
            
# load FESOM mesh -------------------------------------------------------------------------------------
        years = np.arange(self.fyear, self.lyear+1,1)
        
        lon_fesom = mesh.x2
        lat_fesom = mesh.y2       
            
            
# load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag=resultpath+'/'+runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag
        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag,runid=runid)
        mesh_depths = diag['nz'].values

        if self.verbose:
            check_plot_mesh(mesh, plot_globe = False)
                        
            
# load FESOM Diatom C data -------------------------------------------------------------------------------------
        fesom_layered_sum, fesom_layered_mean = fesom_to_maredat_levels(self.resultpath, self.runid, years, mesh, lon_maredat, lat_maredat, self.FT)
    
            
# mask FESOM data for gridpoints without Maredat data ----------------------------------------------------------
        fesom_layered_sum_masked  = mask_model_with_maredat(fesom_layered_sum,maredat_layered_sum)
        fesom_layered_mean_masked = mask_model_with_maredat(fesom_layered_mean,maredat_layered_mean)
                   
            
# Taylor statistics ---------------------------------------------------------------------------------       
        fig_1, sdev_1, crmsd_1, ccoef_1, fig_2, sdev_2, crmsd_2, ccoef_3, fig_3, sdev_3, crmsd_3, ccoef_3, fig_4, sdev_4, crmsd_4, ccoef_4 = plot_Taylor(maredat_layered_sum,fesom_layered_sum_masked,self.FT,self.savefig)
        

# Barplots ----------------------------------------------------------------------------------------  
        barplots(fesom_layered_sum_masked, maredat_layered_sum, lon_maredat, lat_maredat, self.FT, self.savefig)
    
    
# plot Maredat and (unfiltered) FESOM next to each other ---------------------------------------------------
#        figlabel = '' # label used for plotting
#        savelabel = '' # extra label used for saving output
#        mapproj='pc' # other projection than for the remaining Master script

#        plot_2cols(fesom_layered_sum,figlabel,savelabel, maredat_layered_sum, lon_maredat, lat_maredat, mapproj, years, self.savefig, self.savepath)


# Scatter plot ----------------------------------------------------------------------------------------        
#        plot_scatter(fesom_layered_sum_masked, figlabel, savelabel, maredat_layered_sum, years, self.savefig, self.savepath, ccoef_1)

def plot_bars_maredat(fesom_layered_sum_masked, maredat_layered_sum, lon_maredat, lat_maredat, FT, savefig):
    
    '''
    Plot barcharts with total global biomass in each depth layer for Maredat and fesom
    
    Input:
    fesom_layered_sum_masked
    maredat_layered_sum
    lon_maredat
    lat_maredat
    FT (functional group, e.g., DiaC, CoccoC, ...)
    savefig: save figures or not
    '''

    # create an area array to be able to integrate biomass spatially
    radius_earth = 6371000 # in m
    dx = [2*np.pi*radius_earth*np.cos(math.radians(i))/(360) for i in lat_maredat]
    dy = 2*np.pi*radius_earth/(360)
    dy = np.repeat(dy,len(lon_maredat))
    dx_all, dy_all = np.meshgrid(dx,dy)
    area_reg = np.transpose(dx_all*dy_all) # shape: 180, 360
    #area = np.repeat(area_reg[np.newaxis, :, :], 33, axis=0) # shape: 33, 180, 360
        
    # area correction
    fesom_layered_sum_masked_corr = np.zeros((4,180,360))
    fesom_layered_sum_masked_corr[0] = fesom_layered_sum_masked[0] * area_reg
    fesom_layered_sum_masked_corr[1] = fesom_layered_sum_masked[1] * area_reg
    fesom_layered_sum_masked_corr[2] = fesom_layered_sum_masked[2] * area_reg
    fesom_layered_sum_masked_corr[3] = fesom_layered_sum_masked[3] * area_reg
    
    maredat_layered_sum_corr = np.zeros((4,180,360))
    maredat_layered_sum_corr[0] = maredat_layered_sum[0] * area_reg
    maredat_layered_sum_corr[1] = maredat_layered_sum[1] * area_reg
    maredat_layered_sum_corr[2] = maredat_layered_sum[2] * area_reg
    maredat_layered_sum_corr[3] = maredat_layered_sum[3] * area_reg
    
    # compute sum
    fesom_latsum = np.zeros((4,360))
    fesom_totalsum = np.zeros((4))
    for d in range(0,4):
        fesom_latsum[d]   = np.nansum(fesom_layered_sum_masked_corr[d], axis=0)
        fesom_totalsum[d] = np.nansum(fesom_latsum[d], axis=0)
        fesom_totalsum[d] = fesom_totalsum[d]/1e15 # Tg
        
    maredat_latsum = np.zeros((4,360))
    maredat_totalsum = np.zeros((4))
    for d in range(0,4):
        maredat_latsum[d]   = np.nansum(maredat_layered_sum_corr[d], axis=0)
        maredat_totalsum[d] = np.nansum(maredat_latsum[d], axis=0)
        maredat_totalsum[d] = maredat_totalsum[d]/1e15 # Tg
        
        
    #print('fesom totalsum depth 1: ', fesom_totalsum[0])
    #print('fesom totalsum depth 2: ', fesom_totalsum[1])
    #print('fesom totalsum depth 3: ', fesom_totalsum[2])
    #print('fesom totalsum depth 4: ', fesom_totalsum[3])
    
    #print('maredat totalsum depth 1: ', maredat_totalsum[0])
    #print('maredat totalsum depth 2: ', maredat_totalsum[1])
    #print('maredat totalsum depth 3: ', maredat_totalsum[2])
    #print('maredat totalsum depth 4: ', maredat_totalsum[3])
    
    
    # plot preparation
    barWidth = 0.25
    ra3 = np.arange(len(maredat_totalsum))
    rb3 = [x + barWidth for x in ra3]
    rc3 = [x + barWidth for x in rb3]
    rd3 = [x + barWidth for x in rc3]
    re3 = [x + barWidth for x in rd3]
    rf3 = [x + barWidth for x in re3]
    rg3 = [x + barWidth for x in rf3]
    rh3 = [x + barWidth for x in rg3]
    ri3 = [x + barWidth for x in rh3]

    # plot
    fig = plt.figure(num=250,figsize=(10,5),dpi=300, facecolor='w', edgecolor='k')
    plt.bar(ra3, maredat_totalsum, color='blue', width=barWidth, edgecolor='dimgrey', label='Maredat')
    plt.bar(rb3, fesom_totalsum, color='orange', width=barWidth, edgecolor='dimgrey', label='Model')
    #plt.xticks([])
    plt.xticks([f * 0.33 for f in range(0,10)], ['0-5m','','','5-25m','','','25-100m','','','100m to bottom'], fontsize=9, rotation=90)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    #plt.ylim(ymax = 18e-3, ymin = 0)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.ylabel('Annual mean of globally integrated '+FT+' [Tg C]', fontsize=9)
    plt.legend(fontsize=8, loc='upper left', frameon=False)
    
    if(savefig == True):
        fig.savefig(savepath+'Barplot_MaredatFesom_'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
        
    plt.show()

def plot_scatter_maredat(model_masked, figlabel, savelabel, maredat_layered_sum, 
                years, savefig=False, savepath='',
                corr = False):
    '''
    Plot MarEDAT ~ FESOM scattered in two columns (overview & focus), 
    seperated by depth ranges
    
    Input:
    model_masked: FESOM data array with 4 data layers
    figlabel: used for xlabel
    savelabel: used to save the produced graph ('MarEDAT_scatter_'+savelabel...)
    maredat_layered_sum: maredat data as array[4] layered into 0-5, 5-25, 25-100, 100m-bottom; sum in each depth layer
    savefig=False
    savepath
    corr: array(4,2) containing correlation coefficients (out of plot_Taylor)
    '''
        
    if(savepath == ''):
        if(savefig == True):
            raise ValueError('Input for saving graph insufficient')
            
    if(corr != False):
        corr0 = corr[0][1]
        corr1 = corr[1][1]
        corr2 = corr[2][1]
        corr3 = corr[3][1]
    
    print(figlabel)
    
    fig, axes = plt.subplots(4,2, #gridspec_kw={'hspace': 0.001, 'wspace': 0.1}, 
                             figsize=(10,10))

    m1 = axes[0,0]
    m1.plot(np.log10(model_masked[0]),np.log10(maredat_layered_sum[0]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(-3,2.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    if(corr != False):
        m1.text(-2.5,4, "Depth range: 0-5 m, ccoef = {0:5.4f}".format(corr0),color='darkred')
    else: m1.text(-2.5,4, "Depth range: 0-5 m",color='darkred')
    

    m1 = axes[0,1]
    m1.plot(np.log10(model_masked[0]),np.log10(maredat_layered_sum[0]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(1,2)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    #m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-3}$]')
    m1.text(1.1,4, "Depth range: 0-5 m",color='darkred')

    # ---------------------------------------------------------------------------------------------------

    m1 = axes[1,0]
    m1.plot(np.log10(model_masked[1]),np.log10(maredat_layered_sum[1]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(-3,2.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    if(corr != False):
        m1.text(-2.5,4, "Depth range: 5-25 m, ccoef = {0:5.4f}".format(corr1),color='darkred')
    else: m1.text(-2.5,4, "Depth range: 5-25 m",color='darkred')

    m1 = axes[1,1]
    m1.plot(np.log10(model_masked[1]),np.log10(maredat_layered_sum[1]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(1,2)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    #m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-3}$]')
    m1.text(1.1,4, "Depth range: 5-25 m",color='darkred')

    # ---------------------------------------------------------------------------------------------------

    m1 = axes[2,0]
    m1.plot(np.log10(model_masked[2]),np.log10(maredat_layered_sum[2]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(-3,2.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    if(corr != False):
        m1.text(-2.5,4, "Depth range: 25-100 m, ccoef = {0:5.4f}".format(corr2),color='darkred')
    else: m1.text(-2.5,4, "Depth range: 25-100 m",color='darkred')

    m1 = axes[2,1]
    m1.plot(np.log10(model_masked[2]),np.log10(maredat_layered_sum[2]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(0,1.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    #m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-3}$]')
    m1.text(0.15,4, "Depth range: 25-100 m",color='darkred')

    # ---------------------------------------------------------------------------------------------------

    m1 = axes[3,0]
    m1.plot(np.log10(model_masked[3]),np.log10(maredat_layered_sum[3]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(-3,2.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    m1.set_xlabel('Model "'+figlabel+'" Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    if(corr != False):
        m1.text(-2.5,4, "Depth range: 100 m - bot, ccoef = {0:5.4f}".format(corr3),color='darkred')
    else: m1.text(-2.5,4, "Depth range: 100 m - bot",color='darkred')

    m1 = axes[3,1]
    m1.plot(np.log10(model_masked[3]),np.log10(maredat_layered_sum[3]),'.',color='black')
    m1.plot([-4,4], [-4,4])
    m1.set_xlim(-2,0.5)
    m1.set_ylim(-5,5)
    #plt.yticks([-5,-4,-3,-2,-1,0,1,2,3])
    #m1.set_ylabel('Maredat Log$_{10}$ biomass \n [mg C m$^{-3}$]')
    m1.set_xlabel('Model '+figlabel+' Log$_{10}$ biomass \n [mg C m$^{-2}$]')
    m1.text(-1.7,4, "Depth range: 100 m - bottom",color='darkred')
    
    if(savefig == True):
        fig.savefig(savepath+'MarEDAT_scatter_'+savelabel+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
    plt.show()

class plot_bars_chl_npp_latitudes:
    '''class Chl_NPP_lat_comp(resultpath,savepath,mesh,
                            matfileChlsurf,ncfileJohnson2013,matfileNPPcpbm,matfileNPPvgpn,
                            first_year,last_year,savefig=False, verbose=False, output=False, 
                            plotting=True, Taylor=True)
                            
        plot latitudinal Chl and NPP distributions
    '''
    def __init__(self,resultpath,savepath,mesh,
                 matfileChlsurf,ncfileJohnson2013,matfileNPPcpbm,matfileNPPvgpn,
                 first_year,last_year,savefig=False, verbose=False, output=False,
                 plotting=True,runid='fesom'):
        
        self.resultpath = resultpath
        self.fyear = first_year
        self.lyear = last_year
        
        import numpy as np
        import matplotlib.pyplot as plt

        lat = np.arange(-89.5,90,1)
        lat_SO = np.arange(-89.5,-29.5,1.)
        
        years = np.arange(self.fyear, self.lyear+1,1)

        from NPPGlobal import NPPGlobal
        NPPcpbm = NPPGlobal(resultpath,savepath,mesh,matfileNPPcpbm,
                              first_year,last_year,
                              savefig=savefig,plotting=False,output=True,Taylor=False)
        NPPvgpn = NPPGlobal(resultpath,savepath,mesh,matfileNPPvgpn,
                              first_year,last_year,
                              savefig=savefig,plotting=False,output=True,Taylor=False)

        from ChlSouthern import ChlSouthern
        CHLsouth = ChlSouthern(resultpath,savepath,mesh,ncfileJohnson2013,
                                  first_year,last_year,
                                  savefig=savefig,plotting=False,output=True,Taylor=False)

        from ChlGlobal import ChlGlobal
        CHLoccci = ChlGlobal(resultpath,savepath,mesh,matfileChlsurf,
                                  first_year,last_year,
                                  savefig=savefig,plotting=False,output=True,Taylor=False)
        
        from pathlib import Path
        cocco_path = Path(self.resultpath + '/NPPc.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation
        phaeo_path = Path(self.resultpath + '/NPPp.fesom.'+str(years[0])+'.nc') # assuming that coccos were used for the entire simulation if they were used in the first year of simulation

        CHLfesom_lat = np.nanmean(CHLoccci.chl_fesom, axis = 0)
        CHLdfesom_lat = np.nanmean(CHLoccci.chld_fesom, axis = 0)
        CHLnfesom_lat = np.nanmean(CHLoccci.chln_fesom, axis = 0)
        if cocco_path.is_file():
            CHLcfesom_lat = np.nanmean(CHLoccci.chlc_fesom, axis = 0)
        if phaeo_path.is_file():
            CHLpfesom_lat = np.nanmean(CHLoccci.chlp_fesom, axis = 0)
        CHLoccci_lat = np.nanmean(CHLoccci.chl_oc, axis = 0)
        CHLsouth_lat = np.nanmean(CHLsouth.Chl_johnson, axis = 0)
        NPPfesom_lat = np.nanmean(NPPcpbm.NPPt_interp, axis = 0)
        NPPdfesom_lat = np.nanmean(NPPcpbm.NPPd_interp, axis = 0)
        NPPnfesom_lat = np.nanmean(NPPcpbm.NPPn_interp, axis = 0)
        if cocco_path.is_file():
            NPPcfesom_lat = np.nanmean(NPPcpbm.NPPc_interp, axis = 0)
        if phaeo_path.is_file():
            NPPpfesom_lat = np.nanmean(NPPcpbm.NPPp_interp, axis = 0)
        NPPcpbm_lat = np.nanmean(NPPcpbm.NPPt_OC, axis = 0)
        NPPvgpn_lat = np.nanmean(NPPvgpn.NPPt_OC, axis = 0)
        
        if plotting:
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
            
            fig, axes = plt.subplots(2,1,gridspec_kw={'hspace': 0.05, 'wspace': 0.5},
                                             figsize=(15,10), sharex=True)

            axes[0].plot(lat,CHLfesom_lat,label='FESOM-REcoM (Total)',color='C0',lw=3)
            axes[0].plot(lat,CHLdfesom_lat,label='FESOM-REcoM (diatoms)',color='C0',lw=2, linestyle= ':')
            axes[0].plot(lat,CHLnfesom_lat,label='FESOM-REcoM (small phytoplankton)',color='C0',lw=2, linestyle= '-.')
            if cocco_path.is_file():
                axes[0].plot(lat,CHLcfesom_lat,label='FESOM-REcoM (coccolithophores)',color='C0',lw=2, linestyle= '--')
            if phaeo_path.is_file():
                axes[0].plot(lat,CHLpfesom_lat,label='FESOM-REcoM (phaeocystis)',color='C0',lw=1)#, linestyle= '-', marker='s')
            axes[0].plot(lat,CHLoccci_lat,label='OC-CCI',color='C1',lw=3)
            axes[0].plot(lat_SO,CHLsouth_lat,label='Johnson et al.',color='C2',lw=3)
            axes[0].set_ylabel(CHLoccci.unit,fontsize=14)
            axes[0].tick_params(labelsize=14)
            axes[0].grid()
            axes[0].legend(loc='upper center', borderaxespad=0., fontsize=14)

            axes[1].plot(lat,NPPfesom_lat,label='FESOM-REcoM (Total)',color='C0',lw=3)
            axes[1].plot(lat,NPPdfesom_lat,label='FESOM-REcoM (diatoms)',color='C0',lw=2, linestyle= ':')
            axes[1].plot(lat,NPPnfesom_lat,label='FESOM-REcoM (small phytoplankton)',color='C0',lw=2, linestyle= '-.')
            if cocco_path.is_file():
                axes[1].plot(lat,NPPcfesom_lat,label='FESOM-REcoM (coccolithophores)',color='C0',lw=2, linestyle= '--')
            if phaeo_path.is_file():
                axes[1].plot(lat,NPPpfesom_lat,label='FESOM-REcoM (phaeocystis)',color='C0',lw=1)#, linestyle= '', marker='o')
            axes[1].plot(lat,NPPcpbm_lat,label='CbPM',color='C3',lw=3)
            axes[1].plot(lat,NPPvgpn_lat,label='VGPM',color='C4',lw=3)
            axes[1].set_ylabel(NPPcpbm.unit,fontsize=14)
            axes[1].set_xlabel('Latitude ($^\circ$N)',fontsize=14)
            axes[1].tick_params(labelsize=14)
            axes[1].grid()
            axes[1].legend(loc='upper center', borderaxespad=0., fontsize=14)
            
            axes[0].text(-0.12, 1.05, 'A', transform=axes[0].transAxes,
                        size=30, weight='bold')
            axes[1].text(-0.12, 1.05, 'B', transform=axes[1].transAxes,
                        size=30, weight='bold')
        
        if output:
            self.lat = lat
            self.CHLfesom_lat   = CHLfesom_lat
            self.CHLsouth_lat = CHLsouth_lat
            self.CHLoccciunit = CHLoccci.unit
            self.NPPfesom_lat = NPPfesom_lat
            self.NPPcpbm_lat = NPPcpbm_lat
            self.NPPvgpn_lat = NPPvgpn_lat
            self.NPPcpbmunit = NPPcpbm.unit
        
        # fig export  -------------------------------------------------------------------------------------
        if savefig:                
            plt.savefig(savepath+runid+'_'+'Chla_NPP_latitudinal'+'_'+str(first_year)+'to'+str(last_year)+'.png', 
                        dpi = 300, bbox_inches='tight')
            plt.savefig(savepath+runid+'_'+'Chla_NPP_latitudinal'+'_'+str(first_year)+'to'+str(last_year)+'.pdf', 
                        bbox_inches='tight')
        plt.show(block=False)