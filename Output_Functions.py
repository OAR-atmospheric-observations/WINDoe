import os
import numpy as np
import scipy.io
import glob
import sys
from datetime import datetime, timezone
from netCDF4 import Dataset
from collections import Counter

###############################################################################
# This file contains the following functions:
# write_output()
# find_last_time()
###############################################################################

def write_output(vip, globatt, xret, dindices, prior, fsample, exectime, nfilename, shour, verbose):
    
    success = 0
    
    # If fsample is zero, then we will create the netCDF file
    if fsample == 0:
        dt = datetime.utcfromtimestamp(xret['secs'])
        hh = datetime.utcfromtimestamp(xret['secs']).hour
        nn = datetime.utcfromtimestamp(xret['secs']).minute
        ss = datetime.utcfromtimestamp(xret['secs']).second
        hms = hh*10000 + nn*100 + ss
        
        nfilename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + dt.strftime('%Y%m%d.%H%M%S') + '.nc'

        if ((os.path.exists(nfilename)) and (vip['output_clobber'] == 0)):
            print('Error: output file exists -- aborting (' + nfilename + ')')
            return success, nfilename
        
        elif os.path.exists(nfilename):
            print('Warning: clobbering existing output file (' + nfilename + ')')
        
        fid = Dataset(nfilename, 'w')
        tdim = fid.createDimension('time', None)
        nht = len(xret['z'])
        hdim = fid.createDimension('height',nht)
        ddim = fid.createDimension('dfs', len(xret['dfs']))
        
        
        base_time = fid.createVariable('base_time', 'i4')
        base_time.long_name = 'Epoch time'
        base_time.units = 's since 1970/01/01 00:00:00 UTC'
        
        time_offset = fid.createVariable('time_offset', 'f8', ('time',))
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 's'
        
        hour = fid.createVariable('hour', 'f8', ('time',))
        hour.long_name = 'Time'
        hour.units = 'Hours from 00:00 UTC'
        
        qc_flag = fid.createVariable('qc_flags', 'i2', ('time',))
        qc_flag.long_name = 'Manual QC flag'
        qc_flag.units = 'unitless'
        qc_flag.comment = 'value of 0 implies quality is ok; non-zero values indicate that the sample has suspect quality'
        qc_flag.value_2 = 'Implies retrieval did not converge'
        qc_flag.value_3 = 'Implies retrieval converged but RMS between the observed and computed spectrum is too large'
        qc_flag.RMS_threshold_used_for_QC = str(vip['qc_rms_value'])
        
        height = fid.createVariable('height', 'f4', ('height',))
        height.long_name = 'height'
        height.units = 'km AGL'
        
        u_wind = fid.createVariable('u_wind', 'f4', ('time','height',))
        u_wind.long_name = 'U-component of wind'
        u_wind.units = 'm/s'
        
        v_wind = fid.createVariable('v_wind', 'f4', ('time','height',))
        v_wind.long_name = 'V-component of wind'
        v_wind.units = 'm/s'
        
        w_wind = fid.createVariable('w_wind', 'f4', ('time','height',))
        w_wind.long_name = 'W-component of wind'
        w_wind.units = 'm/s'
        
        sigmaU = fid.createVariable('sigma_u', 'f4', ('time','height',))
        sigmaU.long_name = '1-sigma uncertainty in U-component of wind'
        sigmaU.units = 'm/s'
        
        sigmaV = fid.createVariable('sigma_v', 'f4', ('time','height',))
        sigmaV.long_name = '1-sigma uncertainty in V-component of wind'
        sigmaV.units = 'm/s'
        
        sigmaW = fid.createVariable('sigma_w', 'f4', ('time','height',))
        sigmaW.long_name = '1-sigma uncertainty in W-component of wind'
        sigmaW.units = 'm/s'
        
        converged_flag = fid.createVariable('converged_flag', 'i2', ('time',))
        converged_flag.long_name = 'convergence flag'
        converged_flag.units = 'unitless'
        converged_flag.value_0 = '0 indicates no convergence'
        converged_flag.value_1 = '1 indicates convergence in Rodgers sense (i.e., di2n << nX)'
        converged_flag.value_2 = '2 indicates convergence (best rms after rms increased drastically'
        converged_flag.value_3 = '3 indicates convergence (best rms after max_iter)'
        converged_flag.value_4 = '4 indicates linear forward model so no iterations. Ignore rmsa, rmsp, chi2.'
        
        gamma = fid.createVariable('gamma', 'f4', ('time',))
        gamma.long_name = 'gamma parameter'
        gamma.units = 'unitless'
        
        n_iter = fid.createVariable('n_iter', 'i2', ('time',))
        n_iter.long_name = 'number of iterations performed'
        n_iter.units = 'unitless'
        
        rmsa = fid.createVariable('rmsa', 'f4', ('time',))
        rmsa.long_name = 'root mean square error between AERI and MWR obs in the observation vector and forward calculation'
        rmsa.units = 'unitless'
        rmsa.comment1 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        rmsa.comment2 = 'Entire observation vector used in this calculation'
        
        rmsp = fid.createVariable('rmsp', 'f4', ('time',))
        rmsp.long_name = 'root mean square error between prior T/q profile and the retrieved T/q profile'
        rmsp.units = 'unitless'
        rmsp.comment1 = 'Computed as sqrt( mean[ ((Xa - Xn) / sigma_Xa)^2 ] )'
        
        chi2 = fid.createVariable('chi2', 'f4', ('time',))
        chi2.long_name = 'Chi-square statistic of Y vs. F(Xn)'
        chi2.units = 'unitless'
        chi2.comment = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'

        convergence_criteria = fid.createVariable('convergence_criteria', 'f4', ('time',))
        convergence_criteria.long_name = 'convergence criteria di^2'
        convergence_criteria.units = 'unitless'
        
        dfs = fid.createVariable('dfs', 'f4', ('time','dfs',))
        dfs.long_name = 'degrees of freedom of signal'
        dfs.units = 'unitless'
        dfs.comment = 'total DFS, then DFS for U and V'

        sic = fid.createVariable('sic', 'f4', ('time',))
        sic.long_name = 'Shannon information content'
        sic.units = 'unitless'

        vres_u = fid.createVariable('vres_U', 'f4', ('time','height',))
        vres_u.long_name = 'Vertical resolution of the U-wind profile'
        vres_u.units = 'km'
        
        vres_v = fid.createVariable('vres_V', 'f4', ('time','height',))
        vres_v.long_name = 'Vertical resolution of the V-wind profile'
        vres_v.units = 'km'
        
        vres_w = fid.createVariable('vres_W', 'f4', ('time','height',))
        vres_w.long_name = 'Vertical resolution of the W-wind profile'
        vres_w.units = 'km'
        
        cdfs_u = fid.createVariable('cdfs_U', 'f4', ('time','height',))
        cdfs_u.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for U-wind'
        cdfs_u.units = 'unitless'
        
        cdfs_v = fid.createVariable('cdfs_V', 'f4', ('time','height',))
        cdfs_v.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for V-wind'
        cdfs_v.units = 'unitless'
        
        cdfs_w = fid.createVariable('cdfs_W', 'f4', ('time','height',))
        cdfs_w.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for W-wind'
        cdfs_w.units = 'unitless'
        
        srh1 = fid.createVariable('srh_1km', 'f4', ('time',))
        srh1.long_name = '0-1 km storm relative helicity'
        srh1.units = 'm^2/s^2'
        
        srh3 = fid.createVariable('srh_3km', 'f4', ('time',))
        srh3.long_name = '0-3 km storm relative helicity'
        srh3.units = 'm^2/s^2'
        
        sigma_srh1 = fid.createVariable('sigma_srh_1km', 'f4', ('time',))
        sigma_srh1.long_name = '1-sigma uncertainty for 0-1 km storm relative helicity'
        sigma_srh1.units = 'm^2/s^2'
        
        sigma_srh3 = fid.createVariable('sigma_srh_3km', 'f4', ('time',))
        sigma_srh3.long_name = '1-sigma uncertainty for 0-3 km storm relative helicity'
        sigma_srh3.units = 'm^2/s^2'
        
        obs_max = fid.createVariable('obs_hgt_max', 'f4', ('time',))
        obs_max.long_name = 'Height of highest observation'
        obs_max.units = 'km'
        
        obs_min = fid.createVariable('obs_hgt_min', 'f4', ('time',))
        obs_min.long_name = 'Height of lowest observation'
        obs_min.units = 'km'

        for i in xret['max_heights']:
            if xret['max_heights'][i] != -999:
                max_height = fid.createVariable(f'{i}_hgt_max', 'f4', ('time',))
                max_height.long_name = f'Maximum height of {i} data'
                max_height.units = 'km'
        
        if vip['raw_lidar_number'] > 0:
            rdim = fid.createDimension('raw_dim', vip['raw_lidar_number'])
            raw_timedelta = fid.createVariable('raw_lidar_timedelta', 'f4', ('raw_dim',))
            raw_timedelta.long_name = 'Time window for raw lidar data to be included in retrieval for each data source'
            raw_timedelta.units = 'min'
            
            raw_type = fid.createVariable('raw_lidar_type', 'i2', ('raw_dim',))
            raw_type.long_name = 'Type of raw lidar data used'
            raw_type.comment1 = '1 - CLAMPS Halo data'
            raw_type.comment2 = '2 - Windcube 200s lidar'
        
        if vip['proc_lidar_number'] > 0:
            pdim = fid.createDimension('proc_dim', vip['proc_lidar_number'])
            proc_timedelta = fid.createVariable('proc_lidar_timedelta', 'f4', ('proc_dim',))
            proc_timedelta.long_name = 'Time window for processed lidar data to be included in retrieval for each data source'
            proc_timedelta.units = 'min'
            
            proc_type = fid.createVariable('proc_lidar_type', 'i2', ('proc_dim',))
            proc_type.long_name = 'Type of processed lidar data used'
            proc_type.comment1 = '1 - CLAMPS VAD'
            proc_type.comment2 = '2 - ARM/NCAR VAD'

        if vip['cons_profiler_number'] > 0:
            pdim = fid.createDimension('cons_dim', vip['cons_profiler_number'])
            profiler_timedelta = fid.createVariable('wind_profiler_timedelta', 'f4',('cons_dim',))
            profiler_timedelta.long_name = 'Time window for wind profiler data to be included in the retrieval'
            profiler_timedelta.units = 'min'
        
            prof_type = fid.createVariable('wind_profiler_type', 'i2', ('cons_dim',))
            prof_type.long_name = 'Type wind profiler data used'
            prof_type.comment1 = '1 - NCAR 449 Mhz Profiler'
            prof_type.comment2 = '2 - NOAA 915 Mhz Profiler high-res'
            prof_type.comment2 = '3 - NOAA 915 Mhz Profiler low-res'
        
        # B. Adler: count how many observations per type are used at each time stamp
        obsflag_dim = fid.createDimension('obsflag_dim',None)
        obscount_flag = fid.createVariable('obscount_flag','i2',('time','obsflag_dim',))
        obscount_flag.long_name = 'Number of different observation types per profile'
        obscount_flag.comment1 = 'Counts how many observations of a certain platform are used per profile'
        obsunique_flag = fid.createVariable('obsunique_flag','i2',('time','obsflag_dim',))
        obsunique_flag.long_name = 'Unique observations per profile'
        obsunique_flag.comment1 = 'Flags of unique observations per profile'

        if vip['keep_file_small'] == 0:
            nht2 = fid.createDimension('nht2', (nht * 2))
            cov = fid.createVariable('cov', 'f4', ('time', 'nht2', 'nht2'))
            cov.long_name = 'Covariance matrix'
           
            # B. Adler: in addition to covariance matrix, save observations from individual inputs
            # save observation vector, etc, if it is not lidar raw data
            # obsvecidx = np.where(xret['flagY']>1)[0]
            # save observation vector for all data, can get very large if not averaged
            obsvecidx = np.where(xret['flagY']>0)[0]
            if len(obsvecidx)>0:
                obs_dim = fid.createDimension('obs_dim',None)
                #obs_dim = fid.createDimension('obs_dim',len(obsvecidx))
                obs_vector = fid.createVariable('obs_vector','f4',('time','obs_dim'))
                obs_vector.long_name = 'Observation vector Y'
                obs_vector.comment1 = 'mixed units -- see obs_flag field above'

                obs_flag = fid.createVariable('obs_flag', 'i2', ('time','obs_dim',))
                obs_flag.long_name = 'Flag indicating type of observation for each vector element'
                obs_flag.comment1 = 'unitless'

                obs_dimension = fid.createVariable('obs_dimension', 'f8', ('time','obs_dim',))
                obs_dimension.long_name = 'Dimension of the observation vector'
                obs_dimension.comment1 = 'mixed units -- see obs_flag field above'

                obs_vector_uncertainty = fid.createVariable('obs_vector_uncertainty', 'f4', ('time','obs_dim',))
                obs_vector_uncertainty.long_name = '1-sigma uncertainty in the observation vector (sigY)'
                obs_vector_uncertainty.comment1 = 'mixed units -- see obs_flag field above'

                forward_calc = fid.createVariable('forward_calc', 'f4', ('time','obs_dim',))
                forward_calc.long_name = 'Forward calculation from state vector (i.e., F(Xn))'
                forward_calc.comment1 = 'mixed units -- see obs_flag field above'
                        
        # for prior informatoin
        arb_dim1 = fid.createDimension('arb_dim1',None)
        Xa = fid.createVariable('Xa', 'f8', ('arb_dim1',))
        Xa.long_name = 'Prior mean state'
        Xa.units = 'm/s'

        Sa = fid.createVariable('Sa', 'f8', ('arb_dim1','arb_dim1',))
        Sa.long_name = 'Prior covariance'
        Sa.units = 'm2/s2'


        # These should be the last three variables in the file
        lat = fid.createVariable('lat', 'f4')
        lat.long_name = 'latitude'
        lat.units = 'degrees north'

        lon = fid.createVariable('lon', 'f4')
        lon.long_name = 'longitude'
        lon.units = 'degrees east'

        alt = fid.createVariable('alt', 'f4')
        alt.long_name = 'altitude'
        alt.units = 'm above MSL'
        
        # Add some global attributes
        for i in range(len(list(globatt.keys()))):
            fid.setncattr(list(globatt.keys())[i], globatt[list(globatt.keys())[i]])
        fid.Prior_dataset_comment = prior['comment']
        fid.Prior_dataset_filename = prior['filename']
        fid.Prior_dataset_number_profiles = prior['nsonde']
        fid.Number_raw_lidar_sources = vip['raw_lidar_number']
        fid.Number_proc_lidar_sources = vip['proc_lidar_number']
        fid.shour = shour
        if vip['cons_profiler_number'] > 0:
            fid.Number_wind_profiler_sources = vip['cons_profiler_number']
        else:
            fid.Number_wind_profiler_sources = 0
        fid.Total_clock_execution_time_in_s = exectime
        
        
        # Add some of the static (non-time-dependent) data
        base_time[:] = xret['secs']
        height[:] = xret['z']
        if vip['raw_lidar_number'] > 0:
            raw_timedelta = np.array(vip['raw_lidar_timedelta'])
            raw_type = np.array(vip['raw_lidar_type'])
        if vip['proc_lidar_number'] > 0:
            proc_timedelta = np.array(vip['proc_lidar_timedelta'])
            proc_type = np.array(vip['proc_lidar_type'])
        if vip['cons_profiler_number'] > 0:
            profiler_timedelta = np.array(vip['cons_profiler_timedelta'])
            prof_type = np.array(vip['cons_profiler_type'])
        
        Xa[:] = prior['Xa']
        Sa[:,:] = prior['Sa']

        lat[:] = vip['station_lat']
        lon[:] = vip['station_lon']
        alt[:] = vip['station_alt']

                
        fid.close()
    
    # Now append the sample from xret into the file
    nht = len(xret['z'])
    sig = np.sqrt(np.diag(xret['Sop']))
    
    if verbose >= 3:
        print('Appending data to ' + nfilename)
    
    fid = Dataset(nfilename, 'a')
    fid.Total_clock_execution_time_in_s = str(exectime)
    
    time_offset = fid.variables['time_offset']
    hour = fid.variables['hour']
    qc_flag = fid.variables['qc_flags']
    
    u_wind = fid.variables['u_wind']
    v_wind = fid.variables['v_wind']
    w_wind = fid.variables['w_wind']
    
    sigmaU = fid.variables['sigma_u']
    sigmaV = fid.variables['sigma_v']
    sigmaW = fid.variables['sigma_w']
    
    converged_flag = fid.variables['converged_flag']
    gamma = fid.variables['gamma']
    n_iter = fid.variables['n_iter']
    rmsa = fid.variables['rmsa']
    rmsp = fid.variables['rmsp']
    chi2 = fid.variables['chi2']
    convergence_criteria = fid.variables['convergence_criteria']
    dfs = fid.variables['dfs']
    sic = fid.variables['sic']
    vres_u = fid.variables['vres_U']
    vres_v = fid.variables['vres_V']
    vres_w = fid.variables['vres_W']
    cdfs_u = fid.variables['cdfs_U']
    cdfs_v = fid.variables['cdfs_V']
    cdfs_w = fid.variables['cdfs_W']
    srh1 = fid.variables['srh_1km']
    srh3 = fid.variables['srh_3km']
    sigma_srh1 = fid.variables['sigma_srh_1km']
    sigma_srh3 = fid.variables['sigma_srh_3km']
    obs_max = fid.variables['obs_hgt_max']
    obs_min = fid.variables['obs_hgt_min']
    
    basetime = fid.variables['base_time'][:]
    
    time_offset[fsample] = xret['secs'] - basetime
    hour[fsample] = xret['hour']
    qc_flag[fsample] = xret['qcflag']

    did = np.where(np.array(list(fid.dimensions.keys())) == 'height')[0]
    if len(did) == 0:
        print('Whoaa -- this should not happen -- aborting')
        return success, nfilename

    if fid.dimensions['height'].size != len(xret['z']):
        print('Whoaa -- this should not happen size -- aborting')
        return success, nfilename
    
    u_wind[fsample, :] = xret['Xn'][0:nht]
    v_wind[fsample, :] = xret['Xn'][nht:2*nht]
    w_wind[fsample, :] = xret['Xn'][2*nht:3*nht]
    
    sigmaU[fsample, :] = sig[0:nht]
    sigmaV[fsample, :] = sig[nht:2*nht]
    sigmaW[fsample, :] = sig[2*nht:3*nht]
    
    converged_flag[fsample] = xret['converged']
    gamma[fsample] = xret['gamma']
    n_iter[fsample] = xret['niter']
    rmsa[fsample] = xret['rmsa']
    rmsp[fsample] = xret['rmsp']
    chi2[fsample] = xret['chi2']
    convergence_criteria[fsample] = xret['di2n']
    dfs[fsample] = xret['dfs']
    sic[fsample] = xret['sic']
    vres_u[fsample,:] = xret['vres'][0,:]
    vres_v[fsample,:] = xret['vres'][1,:]
    vres_w[fsample,:] = xret['vres'][2,:]
    cdfs_u[fsample,:] = xret['cdfs'][0,:]
    cdfs_v[fsample,:] = xret['cdfs'][1,:]
    cdfs_w[fsample,:] = xret['cdfs'][2,:]
    srh1[fsample] = dindices['indices'][0]
    srh3[fsample] = dindices['indices'][1]
    sigma_srh1[fsample] = dindices['sigma_indices'][0]
    sigma_srh3[fsample] = dindices['sigma_indices'][1]
    foo = np.where((((xret['flagY'] < 8) | (xret['flagY'] > 11)) & (xret['sigY'] < 50)))[0]
    obs_max[fsample] = np.nanmax(xret['dimY'][foo])
    obs_min[fsample] = np.nanmin(xret['dimY'][foo])
    
    for i in xret['max_heights']:
        if xret['max_heights'][i] != -999:
            max_hgt = fid.variables[f'{i}_hgt_max']
            max_hgt[fsample] = xret['max_heights'][i]
    
    # B. Adler: save how many obs per type were used 
    c = Counter(xret['flagY'])
    obsflagunique = np.unique(xret['flagY'])
    obsflagcount = np.full((len(obsflagunique)),np.nan)
    for i in range(len(obsflagunique)): 
        obsflagcount[i]=c[obsflagunique[i]]
    obscount_flag = fid.variables['obscount_flag']
    obscount_flag[fsample,:] = obsflagcount
    obsunique_flag = fid.variables['obsunique_flag']
    obsunique_flag[fsample,:] = obsflagunique


    # only add the covariance matrix if specified
    if vip['keep_file_small'] == 0:
        cov = fid.variables['cov']
        cov[fsample,:,:] = xret['Sop'][:2*nht,:2*nht]
        
        # savee observations from each input
        if all(vip['raw_lidar_average_rv']) == 1:
            # for all obs
            obsvecidx = np.where(xret['flagY']>0)[0]
        else:
            # save observation vector and unertainty, for everything but raw lidar data
            # obsvecidx = np.where(xret['flagY']>1)[0]
            # save observation vector for all data, can get very large if not averaged
            obsvecidx = np.where(xret['flagY']>0)[0]

        if len(obsvecidx)>0:
            obs_vector = fid.variables['obs_vector']
            obs_vector[fsample,:] = xret['Y'][obsvecidx]
            obs_vector_uncertainty = fid.variables['obs_vector_uncertainty']
            obs_vector_uncertainty[fsample,:] = xret['sigY'][obsvecidx]
            obs_flag = fid.variables['obs_flag']
            obs_flag[fsample,:] = xret['flagY'][obsvecidx]
            obs_dimension = fid.variables['obs_dimension']
            obs_dimension[fsample,:] = xret['dimY'][obsvecidx]
            forward_calc = fid.variables['forward_calc']
            forward_calc[fsample,:] = xret['FXn'][obsvecidx]
            

    
    fid.close()
    
    success = 1
    
    return success, nfilename

def find_last_time(date, vip, z, shour):
    
    # Find all of the output files with this date
    
    files = []
    filename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + str(date) + '.*.nc'
    files = files + (glob.glob(filename))
    
    # If none are found, then just run the code as normal
    if len(files) == 0:
        print('The flag output_clober was set to 2 for append, but no prior file was found')
        print('     so code will run as normal')
        nfilename = ' '
        return 0, -999., nfilename
    
    # Otherwise, let's initialize from the last file
    nfilename = files[-1]
    fid = Dataset(nfilename,'r+')
    
    # if the shour is not the same, create a new file
    if fid.shour != shour:
        # close the previously opened file
        fid.close()
        print('The flag output_clober was set to 2 for append, but no prior file was found')
        print('     so code will run as normal')
        nfilename = ' '
        return 0, -999., nfilename
    
    # otherwise, continue initializing from the last time
    bt = fid.variables['base_time'][:]
    to = fid.variables['time_offset'][:]
    xz = fid.variables['height'][:]
    fid.close()
    
    secs = bt+to
    
    # A couple very basic checks to makes sure some of the variables in the output
    # file are the same as the current ones
    diff = np.abs(z-xz)
    foo = np.where(diff > 0.001)[0]
    if ((len(xz) != len(z)) or (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in heights')
        return -999, -999, nfilename
    
    # if the user wants the covariance matrix, but is appending to a file that didn't previously have the variable, exit
    if 'cov' not in fid.variables.keys() and vip['keep_file_small'] == 0:
        print('Error: output clobber is set to 2 (append), but there is no covariance matrix data for the previous timestamps')
        return -999, -999, nfilename
    
    # Return the fsample, last of the secs array, and nfilename
    fsample = len(secs)
    return fsample, secs[-1], nfilename
    
    
    
