"""
loading tools for FESOM2-REcoM2 model output.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    from scipy.io import loadmat, matlab
    import numpy as np
    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def load_maredat(ncfileMaredat):
    '''
    Input: 
    ncfileMaredat
    
    Output:
    lon_maredat
    lat_maredat
    maredat_layered_sum: biomass in [mg C/m2], summed over depth ranges
    maredat_layered_mean: biomass in [mg C/m3], averaged over depth ranges
    '''
    
    import numpy as np
    from netCDF4 import Dataset
    
    f               = Dataset(ncfileMaredat, 'r')
    biomass_maredat = f['BIOMASS'][:] # mug C/L = mg C /m3
    lon_maredat     = f['LONGITUDE'][:]
    lat_maredat     = f['LATITUDE'][:]
    depth_maredat   = f['DEPTH'][:]

    biomass_maredat[:][biomass_maredat[:] == 1e+35] = np.nan # remove fill value
    #print(np.shape(biomass_maredat))

    # mean over year
    biomass_maredat_annual = np.nanmean(biomass_maredat[:],axis=0)
    #print(np.shape(biomass_maredat_annual))
    
    # create depth vector with original Maredat depths
    depths_vector = np.zeros(33) # shape: 33
    for k in np.arange(0,33):
        if k == 0:
            depths_vector[k] = (depth_maredat[k+1]-depth_maredat[k])/2
        elif k == 32:
            depths_vector[k] = (depth_maredat[k]-depth_maredat[k-1])/2
        else:
            depths_vector[k] = ((depth_maredat[k+1]-depth_maredat[k])/2) + ((depth_maredat[k]-depth_maredat[k-1])/2)

    depth = np.repeat(depths_vector[:, np.newaxis], 180, axis=1)
    depth = np.repeat(depth[:, :, np.newaxis], 360, axis=2) # shape: 33, 180, 360
    
    #print('depth_maredat: ', depth_maredat)
    #print('depths_vector: ', depths_vector)


    # depth indices
    dep_ind1  = np.nonzero((0 <= depth_maredat[:] )&(depth_maredat[:] <= 5))
    dep_ind2  = np.nonzero((5 < depth_maredat[:] )&(depth_maredat[:] <= 25))
    dep_ind3  = np.nonzero((25 < depth_maredat[:] )&(depth_maredat[:] <= 100))
    dep_ind4  = np.nonzero((100 < depth_maredat[:]))
    
    # multiply Maredat data with depth vector
    biomass_maredat_annual_aux = biomass_maredat_annual * depth
    
    #print('original Maredat data: ', biomass_maredat_annual[:,12,144])
    #print('Maredat data multiplied with depth: ', biomass_maredat_annual_aux[:,12,144])
    
    # sum over depth ranges
    biomass_maredat_0_5_sum      = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind1], axis=0))
    biomass_maredat_5_25_sum     = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind2], axis=0))
    biomass_maredat_25_100_sum   = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind3], axis=0))
    biomass_maredat_100_bot_sum  = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind4], axis=0))
    
    #print('Indices of second depth range: ', dep_ind2)
    #print('Maredat data summed over second depth range: ', biomass_maredat_5_25_sum[12,144])
    
    maredat_layered_sum = np.zeros((4,180,360))
    maredat_layered_sum[0,:,:] = biomass_maredat_0_5_sum
    maredat_layered_sum[1,:,:] = biomass_maredat_5_25_sum
    maredat_layered_sum[2,:,:] = biomass_maredat_25_100_sum
    maredat_layered_sum[3,:,:] = biomass_maredat_100_bot_sum
    
    # mean over depth ranges
    biomass_maredat_0_5_mean     = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind1],axis=0)/np.nansum(depth[dep_ind1],axis=0))
    biomass_maredat_5_25_mean    = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind2],axis=0)/np.nansum(depth[dep_ind2],axis=0))
    biomass_maredat_25_100_mean  = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind3],axis=0)/np.nansum(depth[dep_ind3],axis=0))
    biomass_maredat_100_bot_mean = np.squeeze(np.nansum(biomass_maredat_annual_aux[dep_ind4],axis=0)/np.nansum(depth[dep_ind4],axis=0))
    
    #print('Maredat data averaged over second depth range: ', biomass_maredat_5_25_mean[12,144])
    
    maredat_layered_mean = np.zeros((4,180,360))
    maredat_layered_mean[0,:,:] = biomass_maredat_0_5_mean
    maredat_layered_mean[1,:,:] = biomass_maredat_5_25_mean
    maredat_layered_mean[2,:,:] = biomass_maredat_25_100_mean
    maredat_layered_mean[3,:,:] = biomass_maredat_100_bot_mean
    
    
    return lat_maredat, lon_maredat, maredat_layered_sum, maredat_layered_mean


def fesom_to_maredat_levels(resultpath, runid, years, mesh, lon_maredat, lat_maredat, FT):
    '''
    Process FESOM data analogue to Maredat layers 0-5, 5-25, 25-100, 100-bottom
    as means and sums
    and interpolate to 1x1 grid
    
    Input:
    resultpath as list
    runid as list
    years 
    mesh: mesh object as loaded with pyfesom2
    lon_maredat
    lat_maredat
    FT (functional type, i.e. Dia, Cocco, ...) 
    
    Output:
    fesom_layered_sum: array with FESOM on 1x1 grid, layered sums
    fesom_layered_mean: array with FESOM on 1x1 grid, layered means
    '''
    
    import pyfesom2 as pf
    import numpy as np
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs
    
    # get interpolation input -> data of all levels
    # annual mean of fesom data already caluclated when reading model results
    #print('\nProcessing {0} for years {1}-{2}'.format(resultpath,years[0],years[-1]))
    
    biomass_fesom = pf.get_data(resultpath, FT, years, mesh, runid=runid, how="mean")
    biomass_fesom = biomass_fesom * 12.01
    #print('DiaC_fesom shape: ',np.shape(biomass_fesom))
    # shape DiaC_fesom: 126858, 47

    # interp preperation
    lons, lats = np.meshgrid(lon_maredat, lat_maredat)

    # load FESOM mesh diag 
    meshdiag=resultpath+'/'+runid+'.mesh.diag.nc'
    #!ncdump -h $meshdiag
    diag = pf.get_meshdiag(mesh,meshdiag=meshdiag,runid=runid)
    mesh_depths = -diag['nz'].values

    # because initially, depths are negative in fesom2, but we want the depth difference positive
    #print('mesh_depths: ',mesh_depths)
    #mesh_depths:  [-0.00e+00  5.00e+00  1.00e+01  2.00e+01  3.00e+01  4.00e+01  5.00e+01
      #6.00e+01  7.00e+01  8.00e+01  9.00e+01  1.00e+02  1.15e+02  1.35e+02
      #1.60e+02  1.90e+02  2.30e+02  2.80e+02  3.40e+02  4.10e+02  4.90e+02
      #5.80e+02  6.80e+02  7.90e+02  9.10e+02  1.04e+03  1.18e+03  1.33e+03
      #1.50e+03  1.70e+03  1.92e+03  2.15e+03  2.40e+03  2.65e+03  2.90e+03
      #3.15e+03  3.40e+03  3.65e+03  3.90e+03  4.15e+03  4.40e+03  4.65e+03
      #4.90e+03  5.15e+03  5.40e+03  5.65e+03  6.00e+03  6.25e+03]

    
    # interpolate fesom data to 1x1 grid
    biomass_fesom_interp = np.zeros((48, 180, 360))
    for d in range(0,len(mesh_depths)-1):
        biomass_fesom_interp[d,:] = pf.fesom2regular(
                data = biomass_fesom[:,d],
                mesh = mesh,
                lons = lons, 
                lats = lats)
       
    
    # create depth vector with original fesom depths
    depths_vector_fesom = np.zeros(48) # shape: 48
    for k in np.arange(0,48):
        if k == 0:
            depths_vector_fesom[k] = (mesh_depths[k+1]-mesh_depths[k])/2
        elif k == 47:
            depths_vector_fesom[k] = (mesh_depths[k]-mesh_depths[k-1])/2
        else:
            depths_vector_fesom[k] = ((mesh_depths[k+1]-mesh_depths[k])/2) + ((mesh_depths[k]-mesh_depths[k-1])/2)
    
    #print('depths_vector_fesom: ',depths_vector_fesom)
    #depths_vector_fesom:  [  2.5   5.    7.5  10.   10.   10.   10.   10.   10.   10.   10.   12.5
      #17.5  22.5  27.5  35.   45.   55.   65.   75.   85.   95.  105.  115.
     #125.  135.  145.  160.  185.  210.  225.  240.  125.  250.  250.  250.
     #250.  250.  250.  250.  250.  250.  250.  250.  250.  300.  300. ]
    
    depth_fesom = np.repeat(depths_vector_fesom[:, np.newaxis], 180, axis=1)
    depth_fesom = np.repeat(depth_fesom[:, :, np.newaxis], 360, axis=2) # shape: 48, 180, 360
    
    #print('mesh_depths: ', mesh_depths)
    #print('depths_vector_fesom: ', depths_vector_fesom)
    
    
    # depth indices similar to MarEDAT
    depth_range = [0,5,25,100,1e5]
    dep_ind1_fesom  = np.nonzero((0 <= mesh_depths[:] )&(mesh_depths[:] <= 5))
    dep_ind2_fesom  = np.nonzero((5 < mesh_depths[:] )&(mesh_depths[:] <= 25))
    dep_ind3_fesom  = np.nonzero((25 < mesh_depths[:] )&(mesh_depths[:] <= 100))
    dep_ind4_fesom  = np.nonzero((100 < mesh_depths[:]))


    # multiply fesom data with depth vector
    biomass_fesom_interp_aux = biomass_fesom_interp * depth_fesom
    
    #print('original fesom data: ', biomass_fesom_interp[:,12,144])
    #print('fesom data multiplied with depth: ', biomass_fesom_interp_aux[:,12,144])
    
    # sum over depth ranges
    biomass_fesom_0_5_sum      = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind1_fesom], axis=0))
    biomass_fesom_5_25_sum     = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind2_fesom], axis=0))
    biomass_fesom_25_100_sum   = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind3_fesom], axis=0))
    biomass_fesom_100_bot_sum  = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind4_fesom], axis=0))
    
    fesom_layered_sum = np.zeros((4,180,360))
    fesom_layered_sum[0,:,:] = biomass_fesom_0_5_sum
    fesom_layered_sum[1,:,:] = biomass_fesom_5_25_sum
    fesom_layered_sum[2,:,:] = biomass_fesom_25_100_sum
    fesom_layered_sum[3,:,:] = biomass_fesom_100_bot_sum
    
    #print('fesom_layered_sum: ', fesom_layered_sum[:,12,144])

    # mean over depth ranges
    biomass_fesom_0_5_mean     = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind1_fesom],axis=0)/np.nansum(depth_fesom[dep_ind1_fesom],axis=0))
    biomass_fesom_5_25_mean    = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind2_fesom],axis=0)/np.nansum(depth_fesom[dep_ind2_fesom],axis=0))
    biomass_fesom_25_100_mean  = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind3_fesom],axis=0)/np.nansum(depth_fesom[dep_ind3_fesom],axis=0))
    biomass_fesom_100_bot_mean = np.squeeze(np.nansum(biomass_fesom_interp_aux[dep_ind4_fesom],axis=0)/np.nansum(depth_fesom[dep_ind4_fesom],axis=0))
    
    #print('fesom data averaged over second depth range: ', biomass_fesom_5_25_mean[12,144])
    
    fesom_layered_mean = np.zeros((4,180,360))
    fesom_layered_mean[0,:,:] = biomass_fesom_0_5_mean
    fesom_layered_mean[1,:,:] = biomass_fesom_5_25_mean
    fesom_layered_mean[2,:,:] = biomass_fesom_25_100_mean
    fesom_layered_mean[3,:,:] = biomass_fesom_100_bot_mean
        
    
    return fesom_layered_sum, fesom_layered_mean
    

def mask_model_with_maredat(fesom_data,maredat_data):
    '''
    Mask model data where no MarEDAT available, then fill masked_array for better processing
    
    Input:
    model: FESOM data array with 4 data layers, i.e. fesom_layered_sum as output of fesom_to_maredat_levels()
    maredat_layered_sum: maredat data as array[4] layered into 0-5, 5-25, 25-100, 100m-bottom
    
    '''
    
    import pyfesom2 as pf
    import numpy as np
    import matplotlib.pylab as plt
    import cartopy.crs as ccrs
    
    model_ma = fesom_data
    model_ma[0] = np.where(maredat_data[0]==0, np.nan, model_ma[0])
    model_ma[1] = np.where(maredat_data[1]==0, np.nan, model_ma[1])
    model_ma[2] = np.where(maredat_data[2]==0, np.nan, model_ma[2])
    model_ma[3] = np.where(maredat_data[3]==0, np.nan, model_ma[3])
    
    return model_ma
    

class load_chau_data:
    '''
    Load SOCAT file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class SOCATdata with self.glodap_int containing interpolated SOCAT ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,first_year,last_year,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview
        self.fyear = first_year
        self.lyear = last_year

        import matplotlib.pyplot as plt
        import numpy as np
        import netCDF4
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        from numba import njit
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle
        import pyfesom2 as pf
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading Chau et al. file: {0}\n***'.format(self.ncfile))
        
        f = Dataset(ncfile, 'r')
        if self.ncvariable == 'fgco2':
            VARraw  = f.variables['fgco2'][:,:,:]      # fCO2 observations
        elif self.ncvariable == 'spco2':
            VARraw  = f.variables['spco2'][:,:,:]
        VARraw  = np.ma.filled(VARraw, np.nan)  # From masked array to numpy array
        VARraw = np.transpose(VARraw, (1, 2, 0)) # now ordered lat, lon, time

        if self.ncvariable == 'fgco2':
            VARraw  = - VARraw * 3600 * 24 * 365 * 1000 / 12.01 # converting kg/m2/s to molC/m2/year
        elif self.ncvariable == 'spco2':
            VARraw *= 9.8692326671 # conversion from Pascal to uatm
        lon_socat  = f.variables['longitude'][:]           
        lon_socat[lon_socat>180.] = lon_socat[lon_socat>180.]-360. # -180 to 180, okay.
        # print min(lon_socat), max(lon_socat)
        lat_socat  = f.variables['latitude'][:]
        hours_socat = f.variables['time'][:]          # hours since 1950-01-01 00:00:00
        time_convert = netCDF4.num2date(hours_socat[:], "hours since 1950-01-01 00:00:00", "Gregorian")

        year_socat = hours_socat/(365*24)+1950              # fractional years
        
        if (self.fyear < 1985):
            print("\n***\n no Chau et al. data to be plotted before 1985\n***")  
            self.fyear = 1985
        if (self.lyear > 2020):
            print("\n***\n no Chau et al. data to be plotted after 2020\n***")  
            self.lyear = 2020
        if ((self.lyear < 1985) or (self.fyear>2020)):
            print("\n***\n Chau et al. data only available for the 1985-2020 period\n*** no matchup possible with FESOM period {0}-{1}\n***\n*** comparing to SOCAT climatology\n***\n***\n***".format(self.fyear,self.lyear))  
            self.lyear = 2020
            self.fyear = 1985
            
        # Define regular grid
        X360, Y180 = np.meshgrid(lon_socat,lat_socat) 
        
        # Select time period
        VARsoc = VARraw[:,:,(year_socat>=self.fyear)&(year_socat<(self.lyear+1))]
        time_convert_select = time_convert[(year_socat>=self.fyear)&(year_socat<(self.lyear+1))]
        
        print("\n***\n years extracted in Chau et al. {0}-{1} \n***".format(self.fyear,self.lyear))
        
        # Loop over months
        #@njit
        def derive_matchups(VARsoc,time_convert_select,mesh,X360,Y180):
            socat_int = np.empty((len(mesh.x2),len(time_convert_select)))
            for i in range(len(time_convert_select)):
                ind   = ~np.isnan(VARsoc[:,:,i])
                aux = VARsoc[:,:,i]
                data1 = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), (mesh.x2,mesh.y2), method='nearest')
                data1 = np.ma.filled(data1, np.nan)
                socat_int[:,i] = data1
            return socat_int
        
        socat_int = derive_matchups(VARsoc,time_convert_select,mesh,X360,Y180)
        
        # calculate mean over that time-period:
        VARsoc_ave = np.nanmean(VARsoc,axis=2)#,dtype=ndarray)
        ind   = ~np.isnan(VARsoc_ave)

        # Interpolation to fesom's grid
        aux = VARsoc_ave
        
        #data1 = griddata((X360[ind], Y180[ind]), aux[ind], (mesh.x2,mesh.y2), method='nearest')
        data1 = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), (mesh.x2,mesh.y2), method='nearest')
        socat_int_mean = np.ma.filled(data1, np.nan)
                
        #socat_int = np.swapaxes(socat_int,0,1) # adjust axes layout to FESOM output
        self.socat_int_mean = socat_int_mean
        self.socat_int = socat_int
        self.time = time_convert_select

class load_glodap_data:
    '''
    Load GLODAPv2 file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class GLODAPdata with self.glodap_int containing interpolated GLODAP ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview

        import matplotlib.pyplot as plt
        import numpy as np
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle

        import pyfesom2 as pf

        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading GLODAP file: {0}\n***'.format(self.ncfile))
        f          = Dataset(self.ncfile, 'r')
        DepthRaw   = -f.variables['Depth'][:]                                
        lonwoa     =  f.variables['lon'][:]
        latwoa     =  f.variables['lat'][:]

        VARraw     =  f.variables[ncvariable][:]
        VARraw     = np.squeeze(VARraw)
        VARraw     = np.ma.filled(VARraw, np.nan)
        
        VARglo     = np.zeros(shape=(np.shape(VARraw)))                   # Change longitude from 0:360 to -180:180 
        for i in range(0,len(DepthRaw)):
          VARglo[i,:,:] = np.hstack((VARraw[i,:,161:360],VARraw[i,:,0:161]))
        
        x360       = np.arange(-179.5,180.,1.)
        y180       = np.arange(-89.5,89.6,1.)
        X360, Y180 = np.meshgrid(x360, y180)

        if(self.get_overview==True):
            #!ncdump -h $self.ncfile
            
            fig = plt.figure(figsize= (7,7))
            ax = plt.subplot()
            im = ax.pcolor(X360, Y180, VARraw[0,:,:])
            cbar = fig.colorbar(im, orientation = 'horizontal')
            cbar.set_label(ncvariable) 
            plt.title('GLODAP var "{0} before interpolation"'.format(self.ncvariable))

        # load FESOM mesh ------------------------------------------------------------------------------------
        #mesh = pf.load_mesh(self.meshpath)
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag
        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runid)
        
        if 'nl' in diag.dims.mapping.keys():
            #nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
            #nod_area.load()
            mesh_depths = diag['Z'].values
        elif 'nz1' in diag.dims.mapping.keys():
            mesh_depths = diag['nz1'].values

        
        #print(np.shape(nod_area[:,:]), type(nod_area))
        
        # check maximum depth in WOA compared to FESOM
        dmin_glodap = np.min(DepthRaw)
        dmin_fesom = np.min(mesh_depths)#mesh.zlev)

        if(dmin_glodap <= dmin_fesom):
            print('***\nDepth greater in GLODAP ({0}) than in FESOM ({1})'.format(dmin_glodap, dmin_fesom))
            ilev = len(mesh_depths)
            max_zlev = mesh_depths[ilev-1]
        else:
            print('***\nDepth greater in FESOM ({1}) than in GLODAP ({0})'.format(dmin_glodap, dmin_fesom))
            ilev = np.where(mesh_depths >= dmin_glodap)
            ilev = ilev[0][-1]
            max_zlev = mesh_depths[ilev]

        print('Please consider choosing max depth level {0} with max depth at {1}!\n***'.format(ilev,max_zlev))
    

        # storage container
        glodap_int = np.zeros((len(mesh.zlev)-1,len(mesh.x2)))
        #print(np.shape(din_int))

        for k in range(0,len(mesh_depths)): # depth are layer depth --> between levels, c.f. FESOM documentation
            lev = mesh_depths[k] # current FESOM depth
            ind1 = np.where(DepthRaw >= lev)
            ind1 = ind1[0][-1]
            ind2 = np.where(DepthRaw < lev)[0]

            if ind2.size > 0:                            # If we have not yet reached the bottom
                ind2 = ind2[0]                           # The index of the depth level below the current fesom level
                c    = DepthRaw[ind1]-DepthRaw[ind2]     # Difference in depth between the data value above and below the fesom depth
                c1   = DepthRaw[ind1]-lev                # Difference between fesom depth and data depth above
                c2   = -(DepthRaw[ind2]-lev)             # Difference between fesom depth and data depth below
                c1   = (c-c1)/c                          # Scaling coefficient for the depth above
                c2   = (c-c2)/c                          # Scaling coefficient for the depth below
            else:                                        # We have reached the bottom
                c1   = 1.
                c2   = 0.
                ind2 = ind1

            indZ  = np.where(mesh_depths == lev)                               
            # original code:
            # indZ  = np.where(-mesh.z3 == lev)          # Find the mesh index of the current fesom depth
            indZ = indZ[0] 
            if False: #(self.get_overview == True):
                print('\nFESOM depth = {0}, GLODAP depths = {1}, {2} \nDepth indices: {3} {4},  FESOM index: {5} \nScaling c1 = {6}, c2 = {7}'.format(lev,DepthRaw[ind1],DepthRaw[ind2],ind1, ind2,indZ,c1,c2))


            aux1  = VARglo[ind1,:,:]                     # Find the data above the current fesom depth
            aux2  = VARglo[ind2,:,:]                     # Find the data below the current fesom depth
            aux   = np.squeeze(c1*aux1+c2*aux2)          # Scaling the data according to vertical distribution as found above
            ind   = np.squeeze(~np.isnan(aux)) 
            #print(np.shape(aux), np.shape(ind))

            # first interpolation to original grid to close empty parts
            aux = griddata((X360[ind], Y180[ind]), aux[ind], (X360, Y180), method='nearest')                             
            # 2D field without nans                           

            # second interpolation to FESOM grid
            glodap_int[indZ,:] = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')  
            # Final interpolated field

            if np.isnan(np.min(glodap_int)): print('WARNING: The interpolated field contains NaNs at depth',lev)                 # Testing if results contain NaNs. If yes, the routine needs adjustments

            if(self.get_overview ==True):
                print('Depth: {0} min = {1} max = {2} mean = {3}'.format(lev,np.min(glodap_int), np.max(glodap_int), np.mean(glodap_int)))

        glodap_int = np.swapaxes(glodap_int,0,1) # adjust axes layout to FESOM output
        #print(np.shape(woa_int))    
        
        self.layer_depths = mesh_depths
        self.glodap_int = glodap_int

class load_mld_data:
    '''
    Load MLD Atlas Matlab(c) file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: matfileMLD,matvariable,meshpath
    
    Out: class MLDdata with self.mld_int containing interpolated MLD variable
    '''
    
    def __init__(self,runname,resultpath,mesh,matfileMLD,verbose=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.matfileMLD = matfileMLD
        self.verbose = verbose

        import matplotlib.pyplot as plt
        import numpy as np
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle

        import pyfesom2 as pf

        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading MLD file: {0}\n***'.format(self.matfileMLD))
        
        matMLD = load_mat(self.matfileMLD)
        MLDobs = matMLD['MLClimato']['ML_depth']
        lon = matMLD['MLClimato']['lon']
        lon[lon>180]=lon[lon>180]-360
        lat = matMLD['MLClimato']['lat']
        londic, latdic = np.meshgrid(lon, lat)
        MLDsept = np.squeeze(MLDobs[8,:,:])
        MLDmarc = np.squeeze(MLDobs[2,:,:])
        
#         MLDsept = np.ma.filled(MLDsept, np.nan)
#         MLDmarc = np.ma.filled(MLDmarc, np.nan)
#         MLDsept = np.ma.filled(MLDsept, 0)
#         MLDmarc = np.ma.filled(MLDmarc, 0)
        MLDmarc[MLDmarc ==0] = np.nan
        MLDsept[MLDsept ==0] = np.nan
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runid)
        if 'nl' in diag.dims.mapping.keys():
            #nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
            #nod_area.load()
            mesh_depths = diag['Z'].values
        elif 'nz1' in diag.dims.mapping.keys():
            mesh_depths = diag['nz1'].values

        # storage container
        mld_marc_int = np.zeros(len(mesh.x2))
        mld_sept_int = np.zeros(len(mesh.x2))
        
        aux1 = np.zeros((MLDmarc.size))*np.nan
        aux2 = np.zeros((MLDmarc.size))*np.nan

        ind1   = np.squeeze(~np.isnan(MLDmarc)) 
        ind2   = np.squeeze(~np.isnan(MLDsept)) 

        # first interpolation to original grid to close empty parts
        aux1 = griddata((londic[ind1], latdic[ind1]), MLDmarc[ind1], (londic, latdic), method='nearest')                             
        aux2 = griddata((londic[ind2], latdic[ind2]), MLDsept[ind2], (londic, latdic), method='nearest')                             
        # 2D field without nans                           
        
        aux1[~ind1] = np.nan
        aux2[~ind2] = np.nan
        
        # second interpolation to FESOM grid
        mld_marc_int[:] = griddata((londic.ravel(), latdic.ravel()), aux1.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')
        mld_sept_int[:] = griddata((londic.ravel(), latdic.ravel()), aux2.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')  
        # Final interpolated field

        if np.isnan(np.min(mld_marc_int)): print('WARNING: The interpolated field contains NaNs')  # Testing if results contain NaNs. If yes, the routine needs adjustments

        if(self.verbose):
            print('Winter MLD: min = {0} max = {1} mean = {2}'.format(np.nanmin(mld_marc_int), np.nanmax(mld_marc_int), np.nanmean(mld_marc_int)))
            print('Summer MLD: min = {0} max = {1} mean = {2}'.format(np.nanmin(mld_sept_int), np.nanmax(mld_sept_int), np.nanmean(mld_sept_int)))

        #mld_sept_int = np.swapaxes(mld_sept_int,0,1) # adjust axes layout to FESOM output
        #mld_marc_int = np.swapaxes(mld_marc_int,0,1)
        #print(np.shape(woa_int))    
        
        self.layer_depths = mesh_depths
        self.mld_sept_int = mld_sept_int
        self.mld_marc_int = mld_marc_int
        self.lon = lon
        self.lat = lat

class load_phc_data:
    '''
    Load PHV3 Atlas NetCDF4 file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class PHCdata with self.phc_int containing interpolated PHC ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview

        import matplotlib.pyplot as plt
        import numpy as np
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle

        import pyfesom2 as pf

        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading PHC file: {0}\n***'.format(self.ncfile))
        f          = Dataset(self.ncfile, 'r')
        DepthRaw   = -f.variables['depth'][:]                                
        lonphc     =  f.variables['lon'][:]
        lonphc[lonphc>180]=lonphc[lonphc>180]-360
        latphc     =  f.variables['lat'][:]
        VARphc     =  f.variables[ncvariable][:]
        VARphc     = np.squeeze(VARphc)
        VARphc     = np.ma.filled(VARphc, np.nan)

        X360, Y180 = np.meshgrid(lonphc, latphc)

        if(self.get_overview==True):
            #!ncdump -h $self.ncfile
            
            fig = plt.figure(figsize= (7,7))
            ax = plt.subplot()
            im = ax.pcolor(X360, Y180, VARphc[0,:,:])
            cbar = fig.colorbar(im, orientation = 'horizontal')
            cbar.set_label(ncvariable) 
            plt.title('PHC var "{0} before interpolation"'.format(self.ncvariable))

        # load FESOM mesh ------------------------------------------------------------------------------------
        #mesh = pf.load_mesh(self.meshpath)
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runid)
        if 'nl' in diag.dims.mapping.keys():
            #nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
            #nod_area.load()
            mesh_depths = diag['Z'].values
        elif 'nz1' in diag.dims.mapping.keys():
            mesh_depths = diag['nz1'].values
        
        # check maximum depth in PHC compared to FESOM
        dmin_phc = np.min(DepthRaw)
        dmin_fesom = np.min(mesh_depths)#mesh.zlev)

        if(dmin_phc <= dmin_fesom):
            print('***\nDepth greater in PHC ({0}) than in FESOM ({1})'.format(dmin_phc, dmin_fesom))
            ilev = len(mesh_depths)
            max_zlev = mesh_depths[ilev-1]
        else:
            print('***\nDepth greater in FESOM ({1}) than in PHC ({0})'.format(dmin_phc, dmin_fesom))
            ilev = np.where(mesh_depths >= dmin_phc)
            ilev = ilev[0][-1]
            max_zlev = mesh_depths[ilev]
        
        # storage container
        phc_int = np.zeros((len(mesh.zlev)-1,len(mesh.x2)))
        #print(np.shape(din_int))

        for k in range(0,len(mesh_depths)): # depth are layer depth --> between levels, c.f. FESOM documentation
            lev = mesh_depths[k] # current FESOM depth
            ind1 = np.where(DepthRaw >= lev)
            ind1 = ind1[0][-1]
            ind2 = np.where(DepthRaw < lev)[0]

            if ind2.size > 0:                            # If we have not yet reached the bottom
                ind2 = ind2[0]                           # The index of the depth level below the current fesom level
                c    = DepthRaw[ind1]-DepthRaw[ind2]     # Difference in depth between the data value above and below the fesom depth
                c1   = DepthRaw[ind1]-lev                # Difference between fesom depth and data depth above
                c2   = -(DepthRaw[ind2]-lev)             # Difference between fesom depth and data depth below
                c1   = (c-c1)/c                          # Scaling coefficient for the depth above
                c2   = (c-c2)/c                          # Scaling coefficient for the depth below
            else:                                        # We have reached the bottom
                c1   = 1.
                c2   = 0.
                ind2 = ind1

            indZ  = np.where(mesh_depths == lev)                               
            # original code:
            # indZ  = np.where(-mesh.z3 == lev)          # Find the mesh index of the current fesom depth
            indZ = indZ[0] 
            if False: #(self.get_overview == True):
                print('\nFESOM depth = {0}, PHC depths = {1}, {2} \nDepth indices: {3} {4},  FESOM index: {5} \nScaling c1 = {6}, c2 = {7}'.format(lev,DepthRaw[ind1],DepthRaw[ind2],ind1, ind2,indZ,c1,c2))


            aux1  = VARphc[ind1,:,:]                     # Find the data above the current fesom depth
            aux2  = VARphc[ind2,:,:]                     # Find the data below the current fesom depth
            aux   = np.squeeze(c1*aux1+c2*aux2)          # Scaling the data according to vertical distribution as found above
            ind   = np.squeeze(~np.isnan(aux)) 
            #print(np.shape(aux), np.shape(ind))

            # first interpolation to original grid to close empty parts
            aux = griddata((X360[ind], Y180[ind]), aux[ind], (X360, Y180), method='nearest')                             
            # 2D field without nans                           

            # second interpolation to FESOM grid
            phc_int[indZ,:] = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')  
            # Final interpolated field

            if np.isnan(np.min(phc_int)): print('WARNING: The interpolated field contains NaNs at depth',lev)                 # Testing if results contain NaNs. If yes, the routine needs adjustments

            if(self.get_overview ==True):
                print('Depth: {0} min = {1} max = {2} mean = {3}'.format(lev,np.min(phc_int), np.max(phc_int), np.mean(phc_int)))

        phc_int = np.swapaxes(phc_int,0,1) # adjust axes layout to FESOM output
        #print(np.shape(phc_int))    
        
        self.layer_depths = mesh_depths
        self.phc_int = phc_int

class load_pisces_data:
    '''
    Load PISCES Atlas NetCDF4 file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class PISCESdata with self.pisces_int containing interpolated PISCES ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview

        import matplotlib.pyplot as plt
        import numpy as np
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle

        import pyfesom2 as pf
        
        # load FESOM mesh ------------------------------------------------------------------------------------
        #mesh = pf.load_mesh(self.meshpath)
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runid)
        if 'nl' in diag.dims.mapping.keys():
            #nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
            #nod_area.load()
            mesh_depths = diag['Z'].values
        elif 'nz1' in diag.dims.mapping.keys():
            mesh_depths = diag['nz1'].values
        
        # load raw PISCES data -------------------------------------------------------------------------------------

        f          = Dataset(self.ncfile, 'r')
        DepthRaw   = -f.variables['DEPTH'][:]                                # Depth is negative
        DFe        =  f.variables[self.ncvariable][:]                         # Unit [mol/L]
        DFe        = 1.e9 * DFe                                              # [mol/L] => [umol/m3] 
        DFe        = np.ma.filled(DFe, np.nan)                               # From masked array to numpy array

        DFeRaw     = np.zeros(shape=(np.shape(DFe)))                         # Change longitude from 0:360 to -180:180 
        for i in range(0,len(DepthRaw)):
          DFeRaw[i,:,:] = np.hstack((DFe[i,:,181:360],DFe[i,:,0:181]))

        x360       = np.arange(-179.5,180.,1.)
        y180       = np.arange(-89.5,89.6,1.)
        X360, Y180 = np.meshgrid(x360, y180)
        
        # interpolate PISCES data -------------------------------------------------------------------------------------

        # check maximum depth in PISCES compared to FESOM
        dmin_woa = np.min(DepthRaw)
        dmin_fesom = np.min(mesh_depths)

        if(dmin_woa <= dmin_fesom):
            print('***\nDepth greater in PISCES ({0}) than in FESOM ({1})'.format(dmin_woa, dmin_fesom))
            ilev = len(mesh_depths)
            max_zlev = mesh_depths[ilev-1]
        else:
            print('***\nDepth greater in FESOM ({1}) than in PISCES ({0})'.format(dmin_woa, dmin_fesom))
            ilev = np.where(mesh_depths >= dmin_woa)
            ilev = ilev[0][-1]
            max_zlev = mesh_depths[ilev]

        # storage container
        pisces_int = np.zeros((len(mesh_depths),len(mesh.x2)))
        #print(np.shape(din_int))

        for k in range(0,len(mesh_depths)): # layer depth as in meshi diag.Z
            lev = mesh_depths[k] # current FESOM depth
            ind1 = np.where(DepthRaw >= lev)
            ind1 = ind1[0][-1]
            ind2 = np.where(DepthRaw < lev)[0]

            if ind2.size > 0:                            # If we have not yet reached the bottom
                ind2 = ind2[0]                           # The index of the depth level below the current fesom level
                c    = DepthRaw[ind1]-DepthRaw[ind2]     # Difference in depth between the data value above and below the fesom depth
                c1   = DepthRaw[ind1]-lev                # Difference between fesom depth and data depth above
                c2   = -(DepthRaw[ind2]-lev)             # Difference between fesom depth and data depth below
                c1   = (c-c1)/c                          # Scaling coefficient for the depth above
                c2   = (c-c2)/c                          # Scaling coefficient for the depth below
            else:                                        # We have reached the bottom
                c1   = 1.
                c2   = 0.
                ind2 = ind1

            indZ  = np.where(mesh_depths == lev)                               
            # original code:
            # indZ  = np.where(-mesh.z3 == lev)          # Find the mesh index of the current fesom depth
            indZ = indZ[0] 
            if(False):
                print('\nFESOM depth = {0}, WOA depths = {1}, {2} \nDepth indices: {3} {4},  FESOM index: {5} \nScaling c1 = {6}, c2 = {7}'.format(lev,DepthRaw[ind1],DepthRaw[ind2],ind1, ind2,indZ,c1,c2))


            aux1  = DFeRaw[ind1,:,:]                     # Find the data above the current fesom depth
            aux2  = DFeRaw[ind2,:,:]                     # Find the data below the current fesom depth
            aux   = np.squeeze(c1*aux1+c2*aux2)          # Scaling the data according to vertical distribution as found above
            ind   = np.squeeze(~np.isnan(aux)) 
            #print(np.shape(aux), np.shape(ind))

            # first interpolation to original grid to close empty parts
            aux = griddata((X360[ind], Y180[ind]), aux[ind], (X360, Y180), method='nearest')                             
            # 2D field without nans                           

            # second interpolation to FESOM grid
            pisces_int[indZ,:] = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')  
            # Final interpolated field

            if np.isnan(np.min(pisces_int)): print('WARNING: The interpolated field contains NaNs at depth',lev)                 # Testing if results contain NaNs. If yes, the routine needs adjustments

            if(False):
                print('Depth: {0} min = {1} max = {2} mean = {3}'.format(lev,np.min(pisces_int), np.max(pisces_int), np.mean(pisces_int)))

        pisces_int = np.swapaxes(pisces_int,0,1) # adjust axes layout to FESOM output   
        
        self.layer_depths = mesh_depths
        self.pisces_int = pisces_int

class load_socat_data:
    '''
    Load SOCAT file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class SOCATdata with self.glodap_int containing interpolated SOCAT ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,first_year,last_year,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview
        self.fyear = first_year
        self.lyear = last_year

        import matplotlib.pyplot as plt
        import numpy as np
        import netCDF4
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        from numba import njit
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle
        import pyfesom2 as pf
        
        # load FESOM mesh -------------------------------------------------------------------------------------
        #mesh       = pf.load_mesh(meshpath)
        years = np.arange(self.fyear, self.lyear+1,1)
        
        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading SOCAT file: {0}\n***'.format(self.ncfile))
        
        f = Dataset(ncfile, 'r')
        VARraw  = f.variables['fco2_ave_unwtd'][:,:,:]      # fCO2 observations
        VARraw  = np.ma.filled(VARraw, np.nan)  # From masked array to numpy array
        VARraw = np.transpose(VARraw, (1, 2, 0)) # now ordered lat, lon, time
        lon_socat  = f.variables['xlon'][:]           # -180 to 180, okay.
        # print min(lon_socat), max(lon_socat)
        lat_socat  = f.variables['ylat'][:]
        days_socat = f.variables['tmnth'][:]          # days since 1970-01-01 00:00:00
        time_convert = netCDF4.num2date(days_socat[:], "days since 1970-01-01 00:00:00", "Gregorian")

        year_socat = days_socat/365+1970              # fractional years
        
        if (self.fyear < 1970):
            print("\n***\n no SOCAT data to be plotted before 1970\n***")  
            self.fyear = 1970
        if (self.lyear > 2020):
            print("\n***\n no SOCAT data to be plotted after 2020\n***")  
            self.lyear = 2020
        if ((self.lyear < 1970) or (self.fyear>2020)):
            print("\n***\n SOCAT data only available for the 1970-2020 period\n*** no matchup possible with FESOM period {0}-{1}\n***\n*** comparing to SOCAT climatology\n***\n***\n***".format(self.fyear,self.lyear))  
            self.lyear = 2020
            self.fyear = 1970
            
        # Define regular grid
        X360, Y180 = np.meshgrid(lon_socat,lat_socat) 
        
        # Select time period
        VARsoc = VARraw[:,:,(year_socat>=self.fyear)&(year_socat<(self.lyear+1))]
        time_convert_select = time_convert[(year_socat>=self.fyear)&(year_socat<(self.lyear+1))]
        
        print("\n***\n years extracted in SOCCAT {0}-{1} \n***".format(self.fyear,self.lyear))
        
        # Loop over months
        #@njit
        def derive_matchups(VARsoc,time_convert_select,mesh,X360,Y180):
            socat_int = np.empty((len(mesh.x2),len(time_convert_select)))
            for i in range(len(time_convert_select)):
                ind   = ~np.isnan(VARsoc[:,:,i])
                aux = VARsoc[:,:,i]
                data1 = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), (mesh.x2,mesh.y2), method='nearest')
                data1 = np.ma.filled(data1, np.nan)
                socat_int[:,i] = data1
            return socat_int
        
        socat_int = derive_matchups(VARsoc,time_convert_select,mesh,X360,Y180)
        
        # calculate mean over that time-period:
        VARsoc_ave = np.nanmean(VARsoc,axis=2)#,dtype=ndarray)
        ind   = ~np.isnan(VARsoc_ave)

        # Interpolation to fesom's grid
        aux = VARsoc_ave
        
        #data1 = griddata((X360[ind], Y180[ind]), aux[ind], (mesh.x2,mesh.y2), method='nearest')
        data1 = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), (mesh.x2,mesh.y2), method='nearest')
        socat_int_mean = np.ma.filled(data1, np.nan)
                
        #socat_int = np.swapaxes(socat_int,0,1) # adjust axes layout to FESOM output
        self.socat_int_mean = socat_int_mean
        self.socat_int = socat_int
        self.time = time_convert_select

class load_woa_data:
    '''
    Load World Ocean Atlas NetCDF4 file and interpolate to FESOM mesh.
    
    Careful: Output still needs to be masked with an ocean bathymetry mask, e.g. with FESOM output!
    
    In: ncfile,ncvariable,meshpath
    
    Out: class WOAdata with self.woa_int containing interpolated WOA ncvariable
    '''
    
    def __init__(self,runname,resultpath,mesh,ncfile,ncvariable,get_overview=False):
    
        self.runid = runname
        self.resultpath = resultpath
        self.mesh = mesh
        self.ncfile = ncfile
        self.ncvariable = ncvariable
        self.get_overview = get_overview

        import matplotlib.pyplot as plt
        import numpy as np
        from netCDF4 import Dataset
        from scipy.interpolate import griddata
        #import skill_metrics as sm
        #import cartopy.crs as ccrs
        #import pickle

        import pyfesom2 as pf

        # load NetCDF ------------------------------------------------------------------------------------
        print('***\nLoading WOA file: {0}\n***'.format(self.ncfile))
        f          = Dataset(self.ncfile, 'r')
        DepthRaw   = -f.variables['depth'][:]                                
        lonwoa     =  f.variables['lon'][:]
        lonwoa[lonwoa>180]=lonwoa[lonwoa>180]-360
        latwoa     =  f.variables['lat'][:]
        Timewoa     =  f.variables['time'][:]
        VARwoa_temp     =  f.variables[ncvariable][:]
        VARwoa_temp     = np.squeeze(VARwoa_temp)
        VARwoa_temp     = np.ma.filled(VARwoa_temp, np.nan)
        
        VARwoa     = np.zeros(shape=(np.shape(VARwoa_temp)))                         # Change longitude from 0:360 to -180:180 
        for i in range(0,len(DepthRaw)):
          VARwoa[i,:,:] = np.hstack((VARwoa_temp[i,:,181:360],VARwoa_temp[i,:,0:181]))
        
        x360       = np.arange(-179.5,180.,1.)
        y180       = np.arange(-89.5,89.6,1.)
        X360, Y180 = np.meshgrid(x360, y180)
        #X360, Y180 = np.meshgrid(lonwoa, latwoa)

        if(self.get_overview==True):
            #!ncdump -h $self.ncfile
            
            fig = plt.figure(figsize= (7,7))
            ax = plt.subplot()
            im = ax.pcolor(X360, Y180, VARwoa[0,:,:])
            cbar = fig.colorbar(im, orientation = 'horizontal')
            cbar.set_label(ncvariable) 
            plt.title('WOA var "{0} before interpolation"'.format(self.ncvariable))

        # load FESOM mesh ------------------------------------------------------------------------------------
        #mesh = pf.load_mesh(self.meshpath)
        
        # load FESOM mesh diag -------------------------------------------------------------------------------
        meshdiag= self.mesh.path+'/'+self.runid+'.mesh.diag.nc'
        #!ncdump -h $meshdiag

        diag = pf.get_meshdiag(mesh,meshdiag=meshdiag, runid=self.runid)
        if 'nl' in diag.dims.mapping.keys():
            #nod_area = diag.rename_dims({"nl": "nz1", "nod_n": "nod2"}).nod_area
            #nod_area.load()
            mesh_depths = diag['Z'].values
        elif 'nz1' in diag.dims.mapping.keys():
            mesh_depths = diag['nz1'].values
        
        # check maximum depth in WOA compared to FESOM
        dmin_woa = np.min(DepthRaw)
        dmin_fesom = np.min(mesh_depths)#mesh.zlev)

        if(dmin_woa <= dmin_fesom):
            print('***\nDepth greater in WOA ({0}) than in FESOM ({1})'.format(dmin_woa, dmin_fesom))
            ilev = len(mesh_depths)
            max_zlev = mesh_depths[ilev-1]
        else:
            print('***\nDepth greater in FESOM ({1}) than in WOA ({0})'.format(dmin_woa, dmin_fesom))
            ilev = np.where(mesh_depths >= dmin_woa)
            ilev = ilev[0][-1]
            max_zlev = mesh_depths[ilev]

        # storage container
        woa_int = np.zeros((len(mesh.zlev)-1,len(mesh.x2)))
        #print(np.shape(din_int))

        for k in range(0,len(mesh_depths)): # depth are layer depth --> between levels, c.f. FESOM documentation
            lev = mesh_depths[k] # current FESOM depth
            ind1 = np.where(DepthRaw >= lev)
            ind1 = ind1[0][-1]
            ind2 = np.where(DepthRaw < lev)[0]

            if ind2.size > 0:                            # If we have not yet reached the bottom
                ind2 = ind2[0]                           # The index of the depth level below the current fesom level
                c    = DepthRaw[ind1]-DepthRaw[ind2]     # Difference in depth between the data value above and below the fesom depth
                c1   = DepthRaw[ind1]-lev                # Difference between fesom depth and data depth above
                c2   = -(DepthRaw[ind2]-lev)             # Difference between fesom depth and data depth below
                c1   = (c-c1)/c                          # Scaling coefficient for the depth above
                c2   = (c-c2)/c                          # Scaling coefficient for the depth below
            else:                                        # We have reached the bottom
                c1   = 1.
                c2   = 0.
                ind2 = ind1

            indZ  = np.where(mesh_depths == lev)                               
            # original code:
            # indZ  = np.where(-mesh.z3 == lev)          # Find the mesh index of the current fesom depth
            indZ = indZ[0] 
            if False: #(self.get_overview == True):
                print('\nFESOM depth = {0}, WOA depths = {1}, {2} \nDepth indices: {3} {4},  FESOM index: {5} \nScaling c1 = {6}, c2 = {7}'.format(lev,DepthRaw[ind1],DepthRaw[ind2],ind1, ind2,indZ,c1,c2))


            aux1  = VARwoa[ind1,:,:]                     # Find the data above the current fesom depth
            aux2  = VARwoa[ind2,:,:]                     # Find the data below the current fesom depth
            aux   = np.squeeze(c1*aux1+c2*aux2)          # Scaling the data according to vertical distribution as found above
            ind   = np.squeeze(~np.isnan(aux)) 
            #print(np.shape(aux), np.shape(ind))

            # first interpolation to original grid to close empty parts
            aux = griddata((X360[ind], Y180[ind]), aux[ind], (X360, Y180), method='nearest')                             
            # 2D field without nans                           

            # second interpolation to FESOM grid
            woa_int[indZ,:] = griddata((X360.ravel(), Y180.ravel()), aux.ravel(), 
                                   (mesh.x2, mesh.y2), method='nearest')  
            # Final interpolated field

            if np.isnan(np.min(woa_int)): print('WARNING: The interpolated field contains NaNs at depth',lev)                 # Testing if results contain NaNs. If yes, the routine needs adjustments

            if(self.get_overview ==True):
                print('Depth: {0} min = {1} max = {2} mean = {3}'.format(lev,np.min(woa_int), np.max(woa_int), np.mean(woa_int)))

        woa_int = np.swapaxes(woa_int,0,1) # adjust axes layout to FESOM output
        #print(np.shape(woa_int))    
        
        self.layer_depths = mesh_depths
        self.woa_int = woa_int