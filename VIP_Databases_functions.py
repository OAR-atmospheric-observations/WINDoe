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
            'raw_lidar_eff_N':-1,            # The effective number samples to use when calculating the lidar error. -1 means use actual N
            'raw_lidar_sig_thresh': 10,       # sigma value to filter noise from windoe estimate
            
            'proc_lidar_number':0,           # Number of lidar data sources used in the retrieval
            'proc_lidar_type':[0],           # List of lidar types. 0-None, 1-CLAMPS VAD, 2-ARM/NCAR VAD
            'proc_lidar_paths':[None],       # List of paths for the lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_minalt':[0],         # Minimum altitude [km] to use lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_maxalt':[2],         # Maximum altitude [km] to use lidar data. Length of list should be same as proc_lidar_number
            'proc_lidar_altitude':[0],       # Altitude of the lidar [m msl]
            'proc_lidar_timedelta':[5],      # Length of window [min] for lidar data to be include in each retrieval time (e.g. 5 means all data within 5 minutes of retrieval time will be used)
            
            'cons_profiler_type':0,       # Type of wind profiler. 0-None, 1-NCAR 449Mhz profiler
            'cons_profiler_path': 'None',   # Path to wind profiler data
            'cons_profiler_minalt':0.,     # Minimum range [km] to use the lidar data
            'cons_profiler_maxalt':0.,     # Maximum range [km] to use the wind profiler data
            'cons_profiler_alitude':0.,    # Altitude of the wind profiler [m msl]
            'cons_profiler_timedelta':5.,  # Length of window [min] for lidar data to be included in each retreival time
            
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
                        (key == 'raw_lidar_sig_thresh') or
                        (key == 'proc_lidar_type') or
                        (key == 'proc_lidar_paths') or
                        (key == 'proc_lidar_minalt') or
                        (key == 'proc_lidar_maxalt') or
                        (key == 'proc_lidar_altitude') or
                        (key == 'proc_lidar_timedelta') or
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
                        (key == 'copter_constants') or
                        (key == 'copter_constants_unc')):
                    
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
                        elif key[0:3] == 'con':
                            if len(feh) != vip['raw_profiler_number']:
                                print('Error: The key ' + key + ' in VIP file must be the same length as '
                                      + ' profiler_number ( ' + str(vip['raw_profiler_number']) + ')')
                                return vip
                        vip[key] = []
                        for x in feh:
                            if (key == 'raw_lidar_paths') or (key == 'proc_lidar_paths') or (key == 'insitu_paths') or (key == 'raw_profiler_paths'):
                                vip[key].append(x.strip())
                            elif ((key == 'raw_lidar_type') or (key == 'raw_lidar_type') or (key == 'insitu_type') or (key == 'raw_profiler_type') or
                                 (key == 'raw_lidar_fix_csm_azimuths') or (key == 'raw_lidar_fix_heading')):
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
    
    foo = np.where((np.array(vip['raw_lidar_type']) < 0) | (np.array(vip['raw_lidar_type']) > 6))[0]
    if len(foo) > 0:
        print('Error: lidar_type can only be set to 0, 1, 2, 3, 4, 5, or 6')
        flag = 1
    
    foo = np.where((np.array(vip['proc_lidar_type']) < 0) | (np.array(vip['proc_lidar_type']) > 3))[0]
    if len(foo) > 0:
        print('Error: lidar_type can only be set to 0, 1, 2, 3')
        flag = 1
    
    if ((vip['cons_profiler_type'] < 0) or (vip['cons_profiler_type'] > 2)):
        print('Error: The wind_profiler_type can only be set to 0, 1, or 2')
        flag = 1
        
    if vip['tres'] <= 0:
        print('Error: tres must be greater than 0')
        flag = 1
    
    if ((vip['raw_lidar_number'] == 0) and (vip['proc_lidar_number'] == 0) and (vip['raw_profiler_number'] == 0)
         and (vip['cons_profiler_type'] == 0) and (vip['use_copter'] == 0)):
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


        