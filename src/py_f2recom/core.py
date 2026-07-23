"""
Core functionality for FESOM2-REcoM2 analysis.
"""
import xarray as xr
import contextlib
import io
import matplotlib.cm as cm
import cmocean as cmo
import pyfesom2 as pf
import numpy as np
import skill_metrics as sm
import matplotlib.pyplot as plt

# plot mesh nodal area overview 
def plot_mesh_area(mesh, plot_globe = True, plot_poles=False, plot_zoom=False, levels=np.arange(0,10500,500)):
    '''check mesh setup nodal and area
    input: mesh object
    '''
    meshdiag = pf.get_meshdiag(mesh)
    
    if plot_globe:    
        # whole globe
        pf.plot(mesh,np.array(meshdiag.nod_area[0,:])/1e6,units='nodal area (km$^{2}$)', mapproj='rob',levels=levels, 
        cmap_extension='max', cmap = cm.viridis_r)
        
    if plot_poles:
        # ... and both poles
        pf.plot(mesh,np.array(meshdiag.nod_area[0,:])/1e6,units='nodal area (km$^{2}$)', mapproj='np', box=[-180, 180, 60, 90], cmap = cm.viridis_r,levels=levels)
        pf.plot(mesh,np.array(meshdiag.nod_area[0,:])/1e6,units='nodal area (km$^{2}$)', mapproj='sp', box=[-180, 180, -90, -60], cmap = cm.viridis_r,levels=levels)

    if plot_zoom: 
        # plot nodal area and mesh structure together
        pf.tplot(mesh, np.array(nod_area[0,:])/1e6, ptype='tri', box=[-30, 30, 60, 82], mapproj='np',lw=0.5,units='nodal area')
        pf.tplot(mesh, np.array(nod_area[0,:])/1e6, ptype='tri', box=[-30, 30, -30, 30], mapproj='merc',lw=0.5,units='nodal area')

# plot mesh resolution overview 
def plot_mesh_resolution(mesh, plot_globe = True, plot_poles=False, plot_zoom=False, levels=np.arange(0,125,5)):
    '''check mesh resolution
    input: mesh objects
    '''
    meshdiag = pf.get_meshdiag(mesh)
    resolution = np.array(2*np.sqrt(meshdiag.nod_area[0,:]/np.pi)/1000)
    
    if plot_globe:    
        # whole globe
        pf.plot(mesh,resolution,units='mesh resolution (km)', mapproj='rob',levels=levels, 
        cmap_extension='max', cmap = cm.viridis_r)
        
    if plot_poles:     
        # ... and both poles
        pf.plot(mesh,resolution,units='mesh resolution (km)', mapproj='np', box=[-180, 180, 60, 90], cmap = cm.viridis_r,levels=levels)
        pf.plot(mesh,resolution,units='mesh resolution (km)', mapproj='sp', box=[-180, 180, -90, -60], cmap = cm.viridis_r,levels=levels)

def plot_taylor_norm(data_ref,data_pred,
                mask=True,
                title='Normalized Taylor Diagram',
                label=['Observation', 'Model'],
                loc='right',
                tickRMSangle = 135.0,
                verbose = True,
                plot = True):
    '''
    Reference: observation
    Predicted: model data
    
    careful: here used different order as in input in skill_metrics: sm.taylor_statistics(pred,ref)
    
    Use this function to get Taylor statistics and plot them as normalized to std(reference) = 1
    
    Root mean sqare error are centered, see sm.centered_rms_dev? for details
    
    Applied normalization: 
        std_normalized(ref) = std(ref) / std(ref)
        std_normalized(pred) = std(pred) / std(ref)
        ... similar for CRMSD
        
    Input:
    data_ref: array with references/observation data
    data_pred: array with model data
    mask: bolean if observation data should be masked with np.where(data_pred !=0)
    title: string for plot title
    label: list with 2 entries for label for 'Observation', 'Model'
    loc: string, positional argument for legend
    tickRMSangle: used for ticks of RMSE in Taylor plot
    verbose: print calculated Taylor statistics
    plot: plot Taylor diagram
    
    Output:
    fig, sdev, crmsd, ccoef
    
    Changed style of marker in skill_metrics/plot_pattern_diagram_markers.py
    '''
                    
    if mask == True:
        # get statistics only from ocean gridpoints (where model data != 0)
        ind_stat = np.where(np.isfinite(data_pred != 0) & np.isfinite(data_ref != 0) & (data_pred != 0) & (data_ref != 0))
        taylor_stats1 = sm.taylor_statistics(data_pred[ind_stat],data_ref[ind_stat])
    else:
        taylor_stats1 = sm.taylor_statistics(data_pred,data_ref)


    # original
    sdev = np.array([taylor_stats1['sdev'][0], 
                     taylor_stats1['sdev'][1]])
    
    crmsd = np.array([taylor_stats1['crmsd'][0],
                      taylor_stats1['crmsd'][1]])
    
    ccoef = np.array([taylor_stats1['ccoef'][0], 
                      taylor_stats1['ccoef'][1]])

    if verbose:
        print('\nOriginal Taylor stats:\nSDEV pred: {0:6.5f}, ref: {1:6.5f}\nCRMS pred: {2:6.5f}, ref: {3:6.5f}\nCORRCOEF: {4:6.5f}'.format(
            sdev[1],sdev[0],crmsd[1],crmsd[0],ccoef[1], ccoef[0]))
    
    # normalized    
    sdev = np.array([taylor_stats1['sdev'][0]/taylor_stats1['sdev'][0], 
                     taylor_stats1['sdev'][1]/taylor_stats1['sdev'][0]])
    
    crmsd = np.array([taylor_stats1['crmsd'][0]/taylor_stats1['sdev'][0],
                      taylor_stats1['crmsd'][1]/taylor_stats1['sdev'][0]])
    
    ccoef = np.array([taylor_stats1['ccoef'][0], 
                      taylor_stats1['ccoef'][1]])

    if verbose:
        print('\nNormalized Taylor stats:\nSDEV pred: {0:6.5f}, ref: {1:6.5f}\nCRMS pred: {2:6.5f}, ref: {3:6.5f}\nCORRCOEF: {4:6.5f}'.format(
            sdev[1],sdev[0],crmsd[1],crmsd[0],ccoef[1], ccoef[0]))

    if plot:
        if(sdev[1] > 1):
            axismax = np.round(sdev[1],0) +0.5
            print("Adjust axis max to {0}".format(axismax))
            tickSTD = np.arange(0,axismax,0.25)
        else: 
            axismax = 1.25
            tickSTD = [0,0.25,0.5,0.75,1.25]
    
        fig = plt.figure(figsize=(10,10), facecolor='w', edgecolor='k')
        sm.taylor_diagram(sdev,crmsd,ccoef, styleOBS = '-', colOBS = 'r', markerobs = 'o',
                              titleOBS = 'observation', markerLabel = label,
                              markerLabelColor = 'c',
                              markerColor = 'c', markerLegend = 'on',
                              tickRMS = [0,0.25,0.5,0.75,1.25],
                              tickRMSangle = tickRMSangle,
                              colRMS = 'm', styleRMS = ':', widthRMS = 2.0,
                              titleRMS = 'off', tickSTD = tickSTD,# tickSTD = [0,0.25,0.5,0.75,1.25],
                              axismax = axismax, #axismax = 1.25, 
                              colSTD = 'b', styleSTD = '-.',
                              widthSTD = 1.0, titleSTD = 'on',
                              colCOR = 'k', styleCOR = '--', widthCOR = 1.0,
                              titleCOR = 'on')

        plt.title(title, loc=loc)   
    
    else:
        fig = 'none'

    return fig, sdev, crmsd, ccoef

def plot_taylor_comp(data_ref,data_pred,mesh,
                depth_array = (0,50,200,1000,2000,4000),
                mask=True,
                title='Normalized Taylor Diagram',
                label=['Observation'],
                loc='left',
                tickRMSangle = 135.0,
                verbose = True,
                plot = True):
    '''
    Reference: observation
    Predicted: model data
    
    careful: here used different order as in input in skill_metrics: sm.taylor_statistics(pred,ref)
    
    Use this function to get Taylor statistics and plot them as normalized to std(reference) = 1
    
    Root mean sqare error are centered, see sm.centered_rms_dev? for details
    
    Applied normalization: 
        std_normalized(ref) = std(ref) / std(ref)
        std_normalized(pred) = std(pred) / std(ref)
        ... similar for CRMSD
        
    Input:
    data_ref: array with references/observation data
    data_pred: array with model data
    mask: bolean if observation data should be masked with np.where(data_pred !=0)
    title: string for plot title
    label: list with 2 entries for label for 'Observation', 'Model'
    loc: string, positional argument for legend
    tickRMSangle: used for ticks of RMSE in Taylor plot
    verbose: print calculated Taylor statistics
    plot: plot Taylor diagram
    
    Output:
    fig, sdev, crmsd, ccoef
    
    Changed style of marker in skill_metrics/plot_pattern_diagram_markers.py
    '''
    
    SDEV = []
    CRMSD = []
    CCOEF = []
    
    for d in depth_array:
        # get mesh index closest to desired depth
        i = pf.ind_for_depth(d,mesh) 
        # get midlevel depth
        plot_depth = str((mesh.zlev[i]+mesh.zlev[i+1])/2)
        label.append('{0} m'.format(plot_depth))
        data_ref_temp = data_ref[:,i]
        data_pred_temp = data_pred[:,i]
        if mask == True:
            # get statistics only from ocean gridpoints (where model data != 0)
            ind_stat = np.where(np.isfinite(data_pred_temp != 0) & np.isfinite(data_ref_temp != 0) & (data_pred_temp != 0) & (data_ref_temp != 0))
            taylor_stats1 = sm.taylor_statistics(data_pred_temp[ind_stat],data_ref_temp[ind_stat])
        else:
            taylor_stats1 = sm.taylor_statistics(data_pred_temp,data_ref_temp)
        
        
        # normalized    
        sdev = np.array([taylor_stats1['sdev'][0]/taylor_stats1['sdev'][0], 
                         taylor_stats1['sdev'][1]/taylor_stats1['sdev'][0]])

        crmsd = np.array([taylor_stats1['crmsd'][0]/taylor_stats1['sdev'][0],
                          taylor_stats1['crmsd'][1]/taylor_stats1['sdev'][0]])

        ccoef = np.array([taylor_stats1['ccoef'][0], 
                          taylor_stats1['ccoef'][1]])

        if verbose:
            print('\nNormalized Taylor stats:\nSDEV pred: {0:6.5f}, ref: {1:6.5f}\nCRMS pred: {2:6.5f}, ref: {3:6.5f}\nCORRCOEF: {4:6.5f}'.format(
                sdev[1],sdev[0],crmsd[1],crmsd[0],ccoef[1], ccoef[0]))
            
        if d == depth_array[0]:
            SDEV = np.append(SDEV,sdev)
            CRMSD = np.append(CRMSD,crmsd)
            CCOEF = np.append(CCOEF,ccoef)
        else:
            SDEV = np.append(SDEV,sdev[1])
            CRMSD = np.append(CRMSD,crmsd[1])
            CCOEF = np.append(CCOEF,ccoef[1])
        
    if plot:
        if max(SDEV) > 1:
            axismax = np.round(max(SDEV),0) +0.5
            print("Adjust axis max to {0}".format(axismax))
            tickSTD = np.arange(0,axismax,0.25)
        else: 
            axismax = 1.25
            tickSTD = [0,0.25,0.5,0.75,1.25]
        
        fig = plt.figure(figsize=(10,10), facecolor='w', edgecolor='k')
        sm.taylor_diagram(SDEV,CRMSD,CCOEF, styleOBS = '-', colOBS = 'r', markerobs = 'o',
                                  titleOBS = 'observation', markerLabel = label,
                                  markerLegend = 'on',
                                  markerColor = 'c',
                                  tickRMS = [0,0.25,0.5,0.75,1.25],
                                  tickRMSangle = tickRMSangle,
                                  colRMS = 'm', styleRMS = ':', widthRMS = 2.0,
                                  titleRMS = 'off', tickSTD = tickSTD,# tickSTD = [0,0.25,0.5,0.75,1.25],
                                  axismax = axismax, #axismax = 1.25, 
                                  colSTD = 'b', styleSTD = '-.',
                                  widthSTD = 1.0, titleSTD = 'on',
                                  colCOR = 'k', styleCOR = '--', widthCOR = 1.0,
                                  titleCOR = 'on')

        #plt.title(title, loc=loc) 
        print('displaying '+title)

    else:
        fig = 'none'

    return fig, sdev, crmsd, ccoef


def plot_taylor(ref,pred,FT,savefig):
    
    '''
    Plot Taylor diagrams for all depth layers
    
    Input:
    ref: reference/observation data -> Maredat
    pred: predicted/model data -> Model
    FT -> functional group (e.g., DiaC, CoccoC, ...)
    savefig: save figures or not
    '''    
    
    pred[np.isnan(pred)] = 0
    
    title_1 = FT+': 0-5 m'
    title_2 = FT+': 5-25 m'
    title_3 = FT+': 25-100 m'
    title_4 = FT+': 100m to bottom'
    
    fig_1, sdev_1, crmsd_1, ccoef_1 = plt_Taylor_norm(ref[0],pred[0],mask=True,title=title_1)
    fig_2, sdev_2, crmsd_2, ccoef_2 = plt_Taylor_norm(ref[1],pred[1],mask=True,title=title_2)
    fig_3, sdev_3, crmsd_3, ccoef_3 = plt_Taylor_norm(ref[2],pred[2],mask=True,title=title_3)
    fig_4, sdev_4, crmsd_4, ccoef_4 = plt_Taylor_norm(ref[3],pred[3],mask=True,title=title_4)
    
    if(savefig == True):
        fig_1.savefig(savepath+'Taylor_0-5m,'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
        fig_2.savefig(savepath+'Taylor_5-25m,'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
        fig_3.savefig(savepath+'Taylor_25-100m,'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
        fig_4.savefig(savepath+'Taylor_100-bottom,'+FT+'_'+str(years[0])+'to'+str(years[-1])+'.png', 
                        dpi = 300, bbox_inches='tight')
    
    return fig_1, sdev_1, crmsd_1, ccoef_1, fig_2, sdev_2, crmsd_2, ccoef_2, fig_3, sdev_3, crmsd_3, ccoef_3, fig_4, sdev_4, crmsd_4, ccoef_4

