import os
import shutil
import numpy as np
import scipy.io

###############################################################################
# This file contains the following functions:
# read_vip_file()
# check_vip()
# abort()
###############################################################################

def read_vip_file(filename,globatt,verbose):
    """
    This routine reads in the controlling parameters from the VIP file

    Parameters
    ----------
    filename : str
        The path to the vip file
    globatt : dict
        Dictionary of global attributes
    verbose : int
        Controls the verbosity of function
    
    Returns
    -------
    vip : dict
        Dictionary that stores all the namelist options for the retrieval

    """

    # This is the dictionary with all of the input that we need.
    # The code below ill be searching the VIP input file for the same keys
    # that are listed in this structure. If a line is found with the same
    # name, then the code will determine the type for the value
    # (e.g., is the value for that key a float, string, integer, etc) and then
    # will cast the value provided in the VIP as that.
    
    # The code will output how many of the keys in the structure were found.
    # Note that not all of them have to be in the VIP file; if a key in this
    # structure is not found in the VIP file then it maintains its default value
    
    vip = ({'success':0,
            'tres':5.0,                    # Temporal resolution [min]
            'first_guess':1,             # First guess for the solution. 1-Use prior, 2-Use previous retrieval
            'max_iterations':10,         # Maximum number of iterations in the retrieval
            'diagonal_covariance':1,      # 0-don't force observation covariance matrix to be diagonal, 1-force it to be diagonal
            'w_mean':2.0,                # mean values for vertical velocity to use in the prior
            'w_lengthscale':1.0,        # length scale in km for the squared covariance function for vertical velocity
            'run_fast':1,               # If 1 the retrieval will not iterate if only linear forward models are used
            
            'station_lat':-999.,         # Station latitude [degN]
            'station_lon':-999.,         # Station longitude [degE]
            'station_alt':-999.,         # Station altitude [m MSL]
            # B. Adler: define grid to use in the retrieval, this can be different from the prior, but the maximum height has to be less than the prior height grid maximum height
            'zgrid': [0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,       0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,       0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,       0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,       0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,       0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,       1.1 , 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2 ,       1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3 , 1.31,       1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4 , 1.41, 1.42,       1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5 , 1.51, 1.52, 1.53,       1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6 , 1.61, 1.62, 1.63, 1.64,       1.65, 1.66, 1.67, 1.68, 1.69, 1.7 , 1.71, 1.72, 1.73, 1.74, 1.75,       1.76, 1.77, 1.78, 1.79, 1.8 , 1.81, 1.82, 1.83, 1.84, 1.85, 1.86,       1.87, 1.88, 1.89, 1.9 , 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97,       1.98, 1.99, 2.   , 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08,       2.09, 2.1 , 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19,       2.2 , 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.3 ,       2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39, 2.4 , 2.41,       2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5 , 2.51, 2.52,       2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59, 2.6 , 2.61, 2.62, 2.63,       2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7 , 2.71, 2.72, 2.73, 2.74,       2.75, 2.76, 2.77, 2.78, 2.79, 2.8 , 2.81, 2.82, 2.83, 2.84, 2.85,       2.86, 2.87, 2.88, 2.89, 2.9 , 2.91, 2.92, 2.93, 2.94, 2.95, 2.96,       2.97, 2.98, 2.99, 3.  , 3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07,       3.08, 3.09, 3.1 , 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18,       3.19, 3.2 , 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 3.28, 3.29,       3.3 , 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39, 3.4 ,       3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49, 3.5 , 3.51,       3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59, 3.6 , 3.61, 3.62,       3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69, 3.7 , 3.71, 3.72, 3.73,       3.74], #Comma delimited list of heights for the z-grid to use in the retrieval [km]
            
            'raw_lidar_number':0,            # Number of lidar data sources used in the retrieval
            'raw_lidar_type':[0],            # List of lidar types. 0-None, 1-CLAMPS Halo, 2-Windcube non-dbs
            'raw_lidar_paths':[None],        # List of paths for the lidar data. Length of list should be same as raw_lidar_number
            'raw_lidar_minrng':[0],          # Minimum range [km] to use lidar data. Length of list should be same as raw_lidar_number
            'raw_lidar_maxrng':[2],          # Maximum range [km] to use lidar data. Length of list should be same as raw_lidar_number
            'raw_lidar_maxsnr':[-5],
            'raw_lidar_minsnr':[-23],
            'raw_lidar_altitude':[0],        # Altitude of the lidar [m msl]
            'raw_lidar_timedelta':[5],        # Length of window [min] for lidar data to be include in each retrieval time (e.g. 5 means all data within 5 minutes of retrieval time will be used)
            'raw_lidar_fix_csm_azimuths':[0],          # Fix the azimuths of the lidar scans
            'raw_lidar_fix_heading':[0],     # Use the heading in the lidar file to add to the azimuths
            'raw_lidar_average_rv':[0],     # Average all radial velocities at full azimuth angles, this allows to reduce the sample size, 0: do not average, 1: average
            'raw_lidar_eff_N':-1,            # The effective number samples to use when calculating the lidar error. -1 means use actual N, -10 means use actual N / 10
            'raw_lidar_sig_thresh': 10,       # sigma value to filter noise from windoe estimate
            
            'proc_lidar_number':0,           # Number of lidar data sources used in the retrieval
            'proc_lidar_type':[0],           # List of lidar types. 0-None, 1-CLAMPS VAD, 2-ARM/NCAR VAD
            'proc_lidar_paths':[None],       # List of paths for the lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_minalt':[0],         # Minimum altitude [km] to use lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_maxalt':[2],         # Maximum altitude [km] to use lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_altitude':[0],       # Altitude of the lidar [m msl]
            'proc_lidar_timedelta':[5],      # Length of window [min] for lidar data to be include in each retrieval time (e.g. 5 means all data within 5 minutes of retrieval time will be used)
            'proc_lidar_average_uv':[0],    # Average u,v compontens over time, this allows to reduce the sample size, 0: do not average, 1: average

            
            'cons_profiler_number':0,        # Number of profiler data sources used in the retrieval
            'cons_profiler_type':[0],       # Type of wind profiler. 0-None, 1-NCAR 449Mhz profiler, 2- NOAA PSL format high-resolution wind profiler, 3- NOAA PSL format low-resolution wind profiler
            'cons_profiler_paths': [None],   # Path to wind profiler data
            'cons_profiler_minalt':[0],     # Minimum range [km] to use the lidar data
            'cons_profiler_maxalt':[5],     # Maximum range [km] to use the wind profiler data
            'cons_profiler_alitude':[0],    # Altitude of the wind profiler [m msl]
            'cons_profiler_timedelta':[5],  # Length of window [min] for lidar data to be included in each retreival time
            
            'raw_profiler_number':0,      # Number of profiler data sources used in the retrieval
            'raw_profiler_type':[0],        # Type of wind profiler. 0-None, 1-ARM 915 MHz wind profiler
            'raw_profiler_paths': [None],  # Path to raw wind profiler data
            'raw_profiler_minalt':[0],     # Minimum range [km] to use the profiler data
            'raw_profiler_maxalt':[7],     # Maximum range [km] to use the profiler data
            'raw_profiler_altitude':[0],   # Altitude of the wind profiler [m msl]
            'raw_profiler_timedelta':[15],  # Length of window [min] for raw profiler data to be included in each retreival time
            'consensus_cutoff':[7],        # Cutoff in m/s for the consensus averaging of the data
            'consensus_min_pct':[50],      # Minimum percentage of data that needs to be included in the consensus average
            
            'insitu_number':0,            # Number of insitu data sources used in the retrieval
            'insitu_type':[0],            # List of insitu data types. 0-None, 1-NCAR Tower data
            'insitu_paths':[None],        # List of paths for the lidar data. Length of list should be same as insitu_number.
            'insitu_minalt':[0],          # Minimum altitude [km] to use insitu data.
            'insitu_maxalt':[2],          # Maximum altitude [km] to use insitu data
            'insitu_timedelta':[5],       # Length of window [min] for insitu data to be included in retrieval
            'insitu_npts':[1],            # Number of insitu points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation
            'insitu_station_height':[10], # Height of in situ measurements on  tower [m agl], default is 10 m

            
            'use_model':0,             # 0-No model constraint, 1-use a model constraint
            'model_path':'None',              # Path to model data
            'model_timedelta':30,           # Length of window [min] for model data to be included in retrieval
            'model_err_inflation':1.0,     # Inflation for model error
            
            'use_copter':0,               # 0-No copter data, 1-OU CopterSonde data
            'copter_path':'None',          # Path to copter data
            'copter_timedelta':30,        # Length of window [min] for copter data to be included in retrieval
            'copter_filter_time':5.0,      # Time in seconds to filter the CopterSonde data. We want to do this to only get independent yaw measurments 
            'copter_pitch_err':1.,         # Error value to be used for the pitch measurement (deg)
            'copter_roll_err':1.,           # Error value to be used for the yaw measurement (deg)
            'copter_ascent_rate':3.,       # Ascent rate of the copter
            'copter_constants':[39.4,-5.71],  # Constant for converting pitch to wind speed for the CopterSonde. Must be two values
            'copter_constants_unc':[0.0,0.0],     # Uncertainty in the parameters for converting pitch to wind speed for CopterSonde. Must be two values
            'copter_fit_unc': 0.0,         # Uncertainty in the fit of the retrieval
            'copter_thin_factor':1,         # Factor used to thin coptersonde data (i.e. 2 means cut the data in half)
            
            'use_windoe':0,               # 0-No ensemble constraint, 1-use a ensemble constraint
            'windoe_path':'None',         # Path to ensemble data
            'windoe_timedelta':45,        # Length of window [min] for insitu data to be included in retrieval
            
            'output_rootname':'None',     # String with rootname of the output file
            'output_path':'None',         # Path where the output file will be placed
            'output_clobber':0,           # 0 - do not clobber preexisting output files, 1 - clobber them, 2 - append to the last file of this day
            'qc_rms_value':3,             # Maximum RMS for retrieval to be good
            'keep_file_small':1,          # 0 - return the covariance matrix, 1 - do not return the covariance matrix
            'vip_filename':'None'}        # Just for tracability
           )
    
    # Read in the file all at once
    
    if os.path.exists(filename):
        
        if verbose >= 1:
            print('Reading the VIP file: ' + filename)
        
        try:
            inputt = np.genfromtxt(filename, dtype=str, comments='#', delimiter='=', autostrip=True)
        except Exception as e:
            print('There was an problem reading the VIP file')
        
    else:
        print('The VIP file ' + filename + ' does not exist')
        return vip
    
    if len(inputt) == 0:
        print('There were no valid lines found in the VIP file')
        return vip
    
    # Look for these tags
    
    nfound = 1
    for key in vip.keys():
        if key != 'success':
            nfound += 1
            if key == 'vip_filename':
                vip['vip_filename'] = filename
            else:
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('Error: There were multple lines with the same key in VIP file: ' + key)
                    return vip
                
                elif len(foo) == 1:
                    if verbose == 3:
                        print('Loading the key ' + key)
                
                    if ((key == 'raw_lidar_type') or
                        (key == 'raw_lidar_paths') or
                        (key == 'raw_lidar_minrng') or
                        (key == 'raw_lidar_maxrng') or
                        (key == 'raw_lidar_maxsnr') or
                        (key == 'raw_lidar_minsnr') or
                        (key == 'raw_lidar_altitude') or
                        (key == 'raw_lidar_timedelta') or
                        (key == 'raw_lidar_fix_csm_azimuths') or
                        (key == 'raw_lidar_fix_heading') or
                        (key == 'raw_lidar_average_rv') or
                        #(key == 'raw_lidar_sig_thresh') or
                        (key == 'proc_lidar_type') or
                        (key == 'proc_lidar_paths') or
                        (key == 'proc_lidar_minalt') or
                        (key == 'proc_lidar_maxalt') or
                        (key == 'proc_lidar_altitude') or
                        (key == 'proc_lidar_timedelta') or
                        (key == 'proc_lidar_average_uv') or
                        (key == 'cons_profiler_type') or
                        (key == 'cons_profiler_paths') or
                        (key == 'cons_profiler_minalt') or
                        (key == 'cons_profiler_maxalt') or
                        (key == 'cons_profiler_altitude') or
                        (key == 'cons_profiler_timedelta') or
                        (key == 'raw_profiler_type') or
                        (key == 'raw_profiler_paths') or
                        (key == 'raw_profiler_minalt') or
                        (key == 'raw_profiler_maxalt') or
                        (key == 'raw_profiler_altitude') or
                        (key == 'raw_profiler_timedelta') or
                        (key == 'consensus_cutoff') or
                        (key == 'consensus_min_pct') or
                        (key == 'insitu_type') or
                        (key == 'insitu_paths') or
                        (key == 'insitu_minalt') or
                        (key == 'insitu_maxalt') or
                        (key == 'insitu_timedelta') or
                        (key == 'insitu_npts') or
                        (key == 'insitu_station_height') or
                        (key == 'copter_constants') or
                        (key == 'copter_constants_unc') or
                        (key == 'zgrid')):
                    
                        feh = inputt[foo,1][0].split(',')
                        if key[0:3] == 'raw':
                            if key[0:5] == 'raw_l':
                                if len(feh) != vip['raw_lidar_number']:
                                    print('Error: The key ' + key + ' in VIP file must be the same length as '
                                          + ' lidar_number ( ' + str(vip['raw_lidar_number']) + ')')
                                    return vip
                            if key[0:5] == 'raw_p':
                                if len(feh) != vip['raw_profiler_number']:
                                    print('Error: The key ' + key + ' in VIP file must be the same length as '
                                          + ' profiler_number ( ' + str(vip['raw_profiler_number']) + ')')
                                    return vip
                        elif key[0:3] == 'pro':
                              if len(feh) != vip['proc_lidar_number']:
                                print('Error: The key ' + key + ' in VIP file must be the same length as '
                                        + ' lidar_number ( ' + str(vip['proc_lidar_number']) + ')')
                                return vip
                        elif key[0:3] == 'ins':
                            if len(feh) != vip['insitu_number']:
                                print('Error: The key ' + key + ' in VIP file must be the same length as '
                                    + 'insitu_number ( ' + str(vip['insitu_number']) + ')')
                                return vip
                        elif key[0:3] == 'cop':
                            if len(feh) != 2:
                                print('Error: The key ' + key + ' in VIP file must be length 2')
                                return vip
                        elif key[0:5] == 'conse':
                            if len(feh) != vip['raw_profiler_number']:
                                print('Error: The key ' + key + ' in VIP file must be the same length as '
                                      + ' profiler_number ( ' + str(vip['raw_profiler_number']) + ')')
                                return vip
                        vip[key] = []
                        for x in feh:
                            if (key == 'raw_lidar_paths') or (key == 'proc_lidar_paths') or (key == 'insitu_paths') or (key == 'raw_profiler_paths') or (key == 'cons_profiler_paths'):
                                vip[key].append(x.strip())
                            elif ((key == 'raw_lidar_type') or (key == 'raw_lidar_type') or (key == 'insitu_type') or (key == 'raw_profiler_type') or (key == 'cons_profiler_type') or (key == 'raw_lidar_fix_csm_azimuths') or (key == 'raw_lidar_fix_heading')):
                                vip[key].append(int(x))
                            else:
                                vip[key].append(float(x))
                
                    else:
                        vip[key] = type(vip[key])(inputt[foo,1][0])
                else:
                    if verbose == 3:
                        print('UNABLE to find the key ' + key)
                    nfound -= 1
                
    if verbose == 3:
        print(vip)
    if verbose >= 2:
        print('There were ' + str(nfound) + ' entries found out of ' + str(len(list(vip.keys()))))
    
    # Now look for any global attributes that might have been entered in the file
    
    matching = [s for s in inputt[:,0] if "globatt" in s]
    
    if verbose >= 2:
        print('There were ' + str(len(matching)) + ' glabal attributes found')
        
    for i in range(len(matching)):
        foo = np.where(matching[i] == inputt[:,0])[0]
        globatt[matching[i][8:]] = inputt[foo,1][0]
    
    vip['success'] = 1
    
    return vip



def check_vip(vip):
    """
    This function performs some QC on the entries in the VIP to ensure that they
    are within a valid range. Not every entry is checked...

    Parameters
    ----------
    vip : dict
        Dictionary that stores all the namelist options for the retrieval

    Returns
    -------
    flag : TYPE
        Flag for aborting retrieval

    """
    
    flag = 0             # default is everything is ok
    
    if ((vip['output_clobber'] < 0) or (vip['output_clobber'] > 2)):
        print('Error: The output_clobber flag can only be set to 0, 1, or 2')
        flag = 1
        
    if ((vip['keep_file_small'] < 0) or (vip['keep_file_small'] > 1)):
        print('Error: The keep_file_small flag can only be set to 0 or 1')
        flag = 1
    
    foo = np.where((np.array(vip['raw_lidar_type']) < 0) | (np.array(vip['raw_lidar_type']) > 5))[0]
    if len(foo) > 0:
        print('Error: lidar_type can only be set to 0, 1, 2, 3, or 4')
        flag = 1
    
    foo = np.where((np.array(vip['proc_lidar_type']) < 0) | (np.array(vip['proc_lidar_type']) > 5))[0]
    if len(foo) > 0:
        print('Error: lidar_type can only be set to 0, 1, 2, 3, 4, 5')
        flag = 1
    
    foo = np.where((np.array(vip['cons_profiler_type']) < 0) | (np.array(vip['cons_profiler_type']) > 4))[0]
    if len(foo) > 0:
        print('Error: The wind_profiler_type can only be set to 0, 1, 2, or 3')
        flag = 1
        
    if vip['tres'] <= 0:
        print('Error: tres must be greater than 0')
        flag = 1
    
    if ((vip['raw_lidar_number'] == 0) and (vip['proc_lidar_number'] == 0) and (vip['raw_profiler_number'] == 0)
         and (vip['cons_profiler_number'] == 0) and (vip['use_copter'] == 0)):
        print('Error: No lidar data, wind profiler, or copter data was selected as input')
        flag = 1
    
    if (vip['station_alt'] <= 0):
        print('Error: the station altitude must be > 0 [m MSL]')
        flag = 1
    
    if ((vip['first_guess'] <= 0) or (vip['first_guess'] > 2)):
        print('Error: The first guess value must be 1 or 2.')
        flag = 1
        
    return flag


def abort(date):
    """
    This routine is called when WINDoe aborts

    Parameters
    ----------
    date : int
        Retrieval date

    Returns
    -------
    None.

    """
    
    print('>>> WINDoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<')
    print('------------------------------------------------------------------')
    print(' ')


        
