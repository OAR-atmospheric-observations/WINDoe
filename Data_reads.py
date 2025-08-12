import numpy as np
import glob
from scipy import interpolate
from netCDF4 import Dataset
import calendar
from datetime import datetime, timedelta, timezone


import VIP_Databases_functions
import Other_functions

###############################################################################
# This file contains the following functions:
# read_all_data()
# read_lidar()
# read_wind_profiler()
# grid_lidar()
# grid_wind_profiler()
###############################################################################


def read_raw_lidar(date, retz, rtime, vip, verbose):
    """


    Parameters
    ----------
    date : int
        Date retrieval is run for
    vip : dict
        Contains all namelist options
    verbose : int
        Controls the verbosity of function

    Returns
    -------
    raw_lidar : dict
        Dictionary that contains key variables from the raw lidar files

    """

    lsecs = []
    rng = []
    az = []
    el = []
    vr = []
    vr_var = []

    available = np.zeros(vip['raw_lidar_number'])

    cdf = ['nc', 'cdf']

    # Loop over all user specified lidar types to be used
    for k in range(vip['raw_lidar_number']):
        no_data = True

        # Read in Halo lidar data
        if vip['raw_lidar_type'][k] == 1:
            if verbose >= 1:
                print('Reading in unprocessed CLAMPS Halo lidar file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(
                            glob.glob(vip['raw_lidar_paths'][k] + '/' + '*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No CLAMPS Halo lidar files found in this directory for this date')
                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                first_scan = True
                for i in range(len(files)):
                    if verbose >= 3:
                        print(f"    Reading in dl file {files[i]}")

                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]
                    rngx = fid.variables['range'][:]
                    #azx =(fid.variables['azimuth'][:] +
                    #       fid.variables['heading'][:]) % 360

                    if vip['raw_lidar_fix_heading'][k] == 1:
                        azx = (fid.variables['azimuth'][:] +
                               fid.variables['heading'][:]) % 360
                        hd = fid.variables['heading'][:]
                    else:
                        azx = fid.variables['azimuth'][:]

                    elx = fid.variables['elevation'][:]
                    vrx = fid.variables['velocity'][:, :]
                    snrx = 10*np.log10(fid.variables['intensity'][:, :] - 1)

                    snum = fid.variables['snum'][:]
                    u_snum, i_snum = np.unique(snum, return_index=True)
                    u_snum = snum[i_snum]

                    to_scan = []
                    vr_scan = []
                    az_scan = []
                    el_scan = []
                    snr_scan = []

                    len_scan = 0
                    for m in range(len(u_snum)):
                        foo = np.where(u_snum[m] == snum)[0]
                        
                        temp_to = np.nanmean(to[foo])
                        if ((bt+temp_to >= rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) &
                            (bt+temp_to < rtime+((vip['raw_lidar_timedelta'][k]/2.)*60))):
                        
                        
                            if first_scan:
                                len_scan = len(foo)

                            elif len_scan != len(foo):
                                print('Error: Raw lidar data source ' + str(k+1) +
                                      ' changed during period retrieval period')
                                continue

                            to_scan.append(np.nanmean(to[foo]))
                            az_scan.append(azx[foo])
                            el_scan.append(elx[foo])
                            vr_scan.append(vrx[foo, :])
                            snr_scan.append(snrx[foo, :])
                    
                    to_scan = np.array(to_scan)
                    az_scan = np.array(az_scan)
                    el_scan = np.array(el_scan)
                    vr_scan = np.array(vr_scan)
                    snr_scan = np.array(snr_scan)

                    # There are no times we want here so just move on
                    if len(to_scan) == 0:
                        if verbose >= 3:
                            print(f"No scans found in {files[i]} for this time ...")
                        
                        fid.close()
                        continue

                    fid.close()

                    if no_data:
                        lsecsx = bt+to_scan
                        rngxx = np.array([rngx]*len(to_scan))
                        azxx = np.copy(az_scan)
                        elxx = np.copy(el_scan)
                        vrxx = np.copy(vr_scan)
                        snrxx = np.copy(snr_scan)
                        no_data = False

                    else:

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if (len(rngx) != len(rngxx[0])):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue

                        lsecsx = np.vstack((lsecsx, bt+to_scan))
                        rngxx = np.vstack((rngxx, np.array(
                            [rngx]*len(to_scan))))
                        azxx = np.vstack((azxx, az_scan))
                        elxx = np.vstack((elxx, el_scan))
                        vrxx = np.vstack((vrxx, vr_scan))
                        snrxx = np.vstack((snrxx, snr_scan))

                if not no_data:

                    # First check that all scans are the same
                    azxx[azxx == 360] = 0

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan
                    snrxx[foo] = np.nan

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))

                    vrxx[foo[0], :, foo[1]] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(vrxx), elxx.shape[1], len(retz)))*-999
                    vr_varzz = np.ones(
                        (len(vrxx), elxx.shape[1], len(retz)))*-999

                    for ii in range(vrxx.shape[0]):
                        for jj in range(vrxx.shape[1]):
                            hgt = rngxx[ii]*np.sin(np.deg2rad(elxx[ii, jj]))
                            vrzz[ii, jj, :] = np.interp(
                                retz, hgt, vrxx[ii, jj, :], left=-999, right=-999)

                        temp_sig, thresh_sig = Other_functions.wind_estimate(
                            vrzz[ii], elxx[ii], azxx[ii], retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                        vr_varzz[ii, :, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999

                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vrxx)] = -999

                    vrxx = vrxx.reshape(
                        (vrxx.shape[0]*vrxx.shape[1], vrxx.shape[2]))
                    vr_varxx = vr_varxx.reshape(
                        (vr_varxx.shape[0]*vr_varxx.shape[1], vr_varxx.shape[2]))

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where(vrxx != -999)[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid lidar data found')

                    rngxx = rngxx[0]
                else:
                    print('No raw lidar data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None

        # Read in Windcube 200s data
        elif vip['raw_lidar_type'][k] == 2:
            if verbose >= 1:
                print('Reading in unprocessed Windcube 200s lidar files')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y-%m-%d'),
                     (datetime.strptime(str(date), '%Y%m%d')).strftime('%Y-%m-%d'),
                     (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y-%m-%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(glob.glob(
                            vip['raw_lidar_paths'][k] + '/' + 'WLS200s*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No Windcube 200s lidar files found in this directory for this date')

                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')

                    keys = list(fid.groups.keys())

                    bt = fid.groups[keys[1]].variables['time_reference'][:]
                    bt = datetime.fromisoformat(bt.replace("Z", "+00:00"))
                    bt = (bt - datetime(1970, 1, 1, tzinfo=timezone.utc)
                          ).total_seconds()
                    to = np.nanmean(fid.groups[keys[1]].variables['time'][:])

                    if ((to < rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) |
                            (to >= rtime+((vip['raw_lidar_timedelta'][k]/2.)*60))):

                        # There are no times we want here so just move on
                        fid.close()
                        continue

                    rngx = fid.groups[keys[1]].variables['range'][:]/1000.
                    azx = fid.groups[keys[1]].variables['azimuth'][:]
                    elx = fid.groups[keys[1]].variables['elevation'][:]
                    vrx = fid.groups[keys[1]].variables['radial_wind_speed'][:]
                    snrx = fid.groups[keys[1]].variables['cnr'][:]

                    fid.close()

                    if no_data:
                        lsecsx = bt+to
                        rngxx = np.array([rngx])
                        azxx = np.array([azx])
                        elxx = np.array([elx])
                        vrxx = np.array([vrx])
                        snrxx = np.array([snrx])
                        no_data = False

                    else:
                        lsecsx = np.append(lsecsx, bt+to)

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if ((len(rngx) != len(rngxx[0])) or
                           (len(azx) != len(azxx[0])) or
                           (len(elx) != len(elxx[0]))):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue

                        rngxx = np.append(rngxx, np.array([rngx]), axis=0)
                        azxx = np.append(azxx, np.array([azx]), axis=0)
                        elxx = np.append(elxx, np.array([elx]), axis=0)
                        vrxx = np.append(vrxx, np.array([vrx]), axis=0)
                        snrxx = np.append(snrxx, np.array([snrx]), axis=0)

                if not no_data:

                    # First check that all scans are the same
                    azxx[azxx == 360] = 0
                    faz = np.where(np.abs(azxx-azxx[0]) > 1)[0]
                    fel = np.where(np.abs(elxx-elxx[0]) > 1)[0]
                    frng = np.where(np.abs(rngxx-rngxx[0]) > 1)[0]

                    if len(faz) > 0 or len(fel) > 0 or len(frng) > 0:
                        print('Error: Raw lidar data source ' + str(k+1) +
                              ' changed during period retrieval period')
                        return {'success': -999}

                    # We only want to use data that falls in our snr bounds

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))[0]

                    vrxx[:, foo] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(vrxx), len(elxx), len(retz)))*-999
                    vr_varzz = np.ones((len(vrxx), len(elxx), len(retz)))*-999

                    for ii in range(vrxx.shape[0]):
                        for jj in range(vrxx.shape[1]):
                            hgt = rngxx*np.sin(np.deg2rad(elxx[ii, jj]))
                            vrzz[ii, jj, :] = np.interp(
                                retz, hgt, vrxx[ii, jj, :], left=-999, right=-999)

                        temp_sig, thresh_sig = Other_functions.wind_estimate(
                            vrzz[ii], elxx[ii], azxx[ii], retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                        vr_varzz[ii, :, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999
                    
                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vr_varxx)] = -999

                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[vrxx < -100] = -999.
                    vr_varxx[vr_varxx < -100] = -999.

                    vrxx = vrxx.reshape(
                        (vrxx.shape[0]*vrxx.shape[1], vrxx.shape[2]))
                    vr_varxx = vr_varxx.reshape(
                        (vr_varxx[0]*vr_varxx[1], vr_varxx[2]))

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where(vrxx != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid raw Windcube 200s data found')
                else:
                    print('No raw Windcube 200s data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None

        elif vip['raw_lidar_type'][k] == 3:
            if verbose >= 1:
                print('Reading in unprocessed CLAMPS CSM Halo lidar file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(
                            glob.glob(vip['raw_lidar_paths'][k] + '/' + '*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No CLAMPS Halo lidar files found in this directory for this date')
                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    foo = np.where((bt+to >= rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['raw_lidar_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()
                        continue

                    rngx = fid.variables['range'][:]
                    
                    if vip['raw_lidar_fix_heading'][k] == 1:
                        azx = (fid.variables['azimuth'][:] +
                               fid.variables['heading'][:]) % 360
                        hd = fid.variables['heading'][:]
                    else:
                        azx = fid.variables['azimuth'][:]
                        
                    elx = fid.variables['elevation'][:]
                    vrx = fid.variables['velocity'][:, :]
                    snrx = 10*np.log10(fid.variables['intensity'][:, :] - 1)

                    fid.close()
                    
                    # Need to make sure this is usable data
                    if vip['raw_lidar_fix_heading'][k] == 1:
                        fah = np.where(((azx[foo] >= -500) & (hd[foo] >= -500)))[0]
                    else:
                        fah = np.where(azx[foo] >= -500)[0]
                        
                    if len(fah) == 0:
                        continue
                    
                    # Need to fix the azimuths in csm files
                    if vip['raw_lidar_fix_csm_azimuths'][k] == 1:
                        azx = azx[foo[fah]]
                        azimuth_follow = np.concatenate((azx[1:], [azx[-1]]))
                        for j in range(len(azx)):
                            azx[j] = Other_functions.mean_azimuth(
                                azx[j], azimuth_follow[j], .6)
                    else:
                        azx = azx[foo[fah]]

                    if no_data:
                        lsecsx = bt+to[foo[fah]]
                        rngxx = np.array([rngx]*len(to[foo[fah]]))
                        azxx = np.copy(azx)
                        elxx = elx[foo[fah]]
                        vrxx = vrx[foo[fah], :]
                        snrxx = snrx[foo[fah], :]
                        no_data = False

                    else:

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if (len(rngx) != len(rngxx[0])):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue
                        
                        lsecsx = np.append(lsecsx, bt+to[foo[fah]])
                        rngxx = np.append(rngxx, np.array(
                            [rngx]*len(to[foo[fah]])), axis=0)
                        azxx = np.append(azxx, azx)
                        elxx = np.append(elxx, elx[foo[fah]])
                        vrxx = np.append(vrxx, vrx[foo[fah]], axis=0)
                        snrxx = np.append(snrxx, snrx[foo[fah]], axis=0)

                if not no_data:

                    # Set the vr variance to a constant value

                    rngxx = rngxx[0]

                    # We only want to use data that falls in our snr bounds

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan
                    snrxx[foo] = np.nan

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))[0]

                    vrxx[:, foo] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(elxx), len(retz)))*-999
                    vr_varzz = np.ones((len(elxx), len(retz)))*-999

                    for ii in range(len(elxx)):
                        hgt = rngxx*np.sin(np.deg2rad(elxx[ii]))
                        vrzz[ii, :] = np.interp(
                            retz, hgt, vrxx[ii, :], left=-999, right=-999)

                    temp_sig, thresh_sig = Other_functions.wind_estimate(
                        vrzz, elxx, azxx, retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                    
                    vr_varzz[:, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999
                    
                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vr_varxx)] = -999

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where((vrxx != -999) & (vr_varxx < 3000))[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid CLAMPS lidar data found')

                else:
                    print('No raw Halo CSM data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None

        elif vip['raw_lidar_type'][k] == 4:
            if verbose >= 1:
                print('Reading in unprocessed ARM Halo lidar files')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(glob.glob(
                            vip['raw_lidar_paths'][k] + '/' + '*dlppi*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No ARM Halo lidar files found in this directory for this date')
                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                for i in range(len(files)):

                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]
                    elx = fid.variables['elevation'][:]

                    if ((bt+np.nanmean(to) < rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) |
                            (bt+np.nanmean(to) >= rtime+((vip['raw_lidar_timedelta'][k]/2.)*60))):
                        # There are no times we want here so just move on
                        if verbose >= 3:
                            print(f"No scans found in {files[i]} for this time ...")
                        
                        fid.close()
                        continue

                    rngx = fid.variables['range'][:].data/1000.
                    azx = fid.variables['azimuth'][:].data
                    elx = fid.variables['elevation'][:].data
                    vrx = fid.variables['radial_velocity'][:, :].data
                    snrx = 10 * \
                        np.log10(fid.variables['intensity'][:, :].data - 1)
                    
                    fid.close()

                    if no_data:
                        lsecsx = np.array(bt+np.nanmean(to))
                        rngxx = np.array([rngx])
                        azxx = np.array([azx])
                        elxx = np.array([elx])
                        vrxx = np.array([vrx])
                        snrxx = np.array([snrx])
                        no_data = False

                    else:

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if ((len(rngx) != len(rngxx[0])) or
                           (len(azx) != len(azxx[0])) or
                           (len(elx) != len(elxx[0]))):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue

                        lsecsx = np.append(lsecsx, bt+np.nanmean(to))
                        rngxx = np.append(rngxx, np.array([rngx]), axis=0)
                        azxx = np.append(azxx, np.array([azx]), axis=0)
                        elxx = np.append(elxx, np.array([elx]), axis=0)
                        vrxx = np.append(vrxx, np.array([vrx]), axis=0)
                        snrxx = np.append(snrxx, np.array([snrx]), axis=0)

                if not no_data:

                    # First check that all scans are the same
                    azxx[azxx == 360] = 0
                    faz = np.where(np.abs(azxx-azxx[0]) > 1)[0]
                    fel = np.where(np.abs(elxx-elxx[0]) > 1)[0]
                    frng = np.where(np.abs(rngxx-rngxx[0]) > 1)[0]

                    if len(faz) > 0 or len(fel) > 0 or len(frng) > 0:
                        print('Error: Raw lidar data source ' + str(k+1) +
                              ' changed during period retrieval period')
                        return {'success': -999}

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan
                    snrxx[foo] = np.nan

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))

                    vrxx[foo[0], :, foo[1]] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(vrxx), elxx.shape[1], len(retz)))*-999
                    vr_varzz = np.ones(
                        (len(vrxx), elxx.shape[1], len(retz)))*-999

                    for ii in range(vrxx.shape[0]):
                        for jj in range(vrxx.shape[1]):
                            hgt = rngxx[ii]*np.sin(np.deg2rad(elxx[ii, jj]))
                            vrzz[ii, jj, :] = np.interp(
                                retz, hgt, vrxx[ii, jj, :], left=-999, right=-999)

                        temp_sig, thresh_sig = Other_functions.wind_estimate(
                            vrzz[ii], elxx[ii], azxx[ii], retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                        vr_varzz[ii, :, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999
                    
                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vrxx)] = -999

                    vrxx = vrxx.reshape(
                        (vrxx.shape[0]*vrxx.shape[1], vrxx.shape[2]))
                    vr_varxx = vr_varxx.reshape(
                        (vr_varxx.shape[0]*vr_varxx.shape[1], vr_varxx.shape[2]))

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where(vrxx != -999)[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid ARM lidar data found')

                    rngxx = rngxx[0]
                else:
                    print('No raw ARM lidar data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None
    
        elif vip['raw_lidar_type'][k] == 5:
            if verbose >= 1:
                print('Reading in unprocessed ARM CSM Halo lidar file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(
                            glob.glob(vip['raw_lidar_paths'][k] + '/' + '*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No ARM Halo lidar files found in this directory for this date')
                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    foo = np.where((bt+to >= rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['raw_lidar_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()
                        continue

                    rngx = fid.variables['range'][:]/1000.
                    
                    if vip['raw_lidar_fix_heading'][k] == 1:
                        azx = (fid.variables['azimuth'][:] +
                               fid.variables['heading'][:]) % 360
                        hd = fid.variables['heading'][:]
                    else:
                        azx = fid.variables['azimuth'][:]
                        
                    elx = fid.variables['elevation'][:]
                    vrx = fid.variables['radial_velocity'][:, :]
                    snrx = 10*np.log10(fid.variables['intensity'][:, :] - 1)

                    fid.close()
                    
                    # Need to make sure this is usable data
                    if vip['raw_lidar_fix_heading'][k] == 1:
                        fah = np.where(((azx[foo] >= -500) & (hd[foo] >= -500)))
                    else:
                        fah = np.where(azx[foo] >= -500)[0]
                        
                    if len(fah) == 0:
                        continue
                    
                    # Need to fix the azimuths in csm files
                    if vip['raw_lidar_fix_csm_azimuths'][k] == 1:
                        azx = azx[foo[fah]]
                        azimuth_follow = np.concatenate((azx[1:], [azx[-1]]))
                        for j in range(len(azx)):
                            azx[j] = Other_functions.mean_azimuth(
                                azx[j], azimuth_follow[j], .6)
                    else:
                        azx = azx[foo[fah]]

                    if no_data:
                        lsecsx = bt+to[foo[fah]]
                        rngxx = np.array([rngx]*len(to[foo[fah]]))
                        azxx = np.copy(azx)
                        elxx = elx[foo[fah]]
                        vrxx = vrx[foo[fah], :]
                        snrxx = snrx[foo[fah], :]
                        no_data = False

                    else:

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if (len(rngx) != len(rngxx[0])):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue
                        
                        lsecsx = np.append(lsecsx, bt+to[foo[fah]])
                        rngxx = np.append(rngxx, np.array(
                            [rngx]*len(to[foo[fah]])), axis=0)
                        azxx = np.append(azxx, azx)
                        elxx = np.append(elxx, elx[foo[fah]])
                        vrxx = np.append(vrxx, vrx[foo[fah]], axis=0)
                        snrxx = np.append(snrxx, snrx[foo[fah]], axis=0)

                if not no_data:

                    # Set the vr variance to a constant value

                    rngxx = rngxx[0]

                    # We only want to use data that falls in our snr bounds

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan
                    snrxx[foo] = np.nan

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))[0]

                    vrxx[:, foo] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(elxx), len(retz)))*-999
                    vr_varzz = np.ones((len(elxx), len(retz)))*-999

                    for ii in range(len(elxx)):
                        hgt = rngxx*np.sin(np.deg2rad(elxx[ii]))
                        vrzz[ii, :] = np.interp(
                            retz, hgt, vrxx[ii, :], left=-999, right=-999)

                    temp_sig, thresh_sig = Other_functions.wind_estimate(
                        vrzz, elxx, azxx, retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                    vr_varzz[:, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999
                    
                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vr_varxx)] = -999

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where((vrxx != -999) & (vr_varxx < 3000))[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid ARM lidar data found')

                else:
                    print('No raw ARM CSM data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None

        elif vip['raw_lidar_type'][k] == 6:
            if verbose >= 1:
                print('Reading in unprocessed MoDLS lidar file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(
                            glob.glob(vip['raw_lidar_paths'][k] + '/' + '*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No MoDLS lidar files found in this directory for this date')
                lsecsx = None
                rngxx = None
                azxx = None
                elxx = None
                vrxx = None
                vr_varxx = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['epochTime'][0]
                    to = fid.variables['epochTime'][:] - bt

                    foo = np.where((bt+to >= rtime-((vip['raw_lidar_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['raw_lidar_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()
                        continue

                    rngx = fid.variables['height'][:] / 1000
                    elx = fid.variables['elevation'][:]
                    vrx = fid.variables['velocity'][:, :]
                    snrx = 10*np.log10(fid.variables['intensity'][:, :] - 1)
                    azx = fid.variables['azimuth'][:]

                    fid.close()
                


                    if no_data:
                        lsecsx = bt+to[foo]
                        rngxx = np.array([rngx]*len(to[foo]))
                        azxx = np.copy(azx[foo])
                        elxx = elx[foo]
                        vrxx = vrx[foo, :]
                        snrxx = snrx[foo, :]
                        no_data = False

                    else:

                        # Check to make sure the range array is the same length
                        # and azimuth and elevation are the same. If not
                        # abort and tell user

                        if (len(rngx) != len(rngxx[0])):
                            print('Error: Raw lidar data source ' + str(k+1) +
                                  ' changed during period retrieval period')
                            continue
                        
                        lsecsx = np.append(lsecsx, bt+to[foo])
                        rngxx = np.append(rngxx, np.array(
                            [rngx]*len(to[foo])), axis=0)
                        azxx = np.append(azxx, azx[foo])
                        elxx = np.append(elxx, elx[foo])
                        vrxx = np.append(vrxx, vrx[foo], axis=0)
                        snrxx = np.append(snrxx, snrx[foo], axis=0)

                if not no_data:

                    # Set the vr variance to a constant value

                    rngxx = rngxx[0]

                    # We only want to use data that falls in our snr bounds

                    foo = np.where((snrxx < vip['raw_lidar_minsnr'][k]) |
                                   (snrxx > vip['raw_lidar_maxsnr'][k]))

                    vrxx[foo] = np.nan
                    snrxx[foo] = np.nan

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((rngxx < vip['raw_lidar_minrng'][k]) |
                                   (rngxx > vip['raw_lidar_maxrng'][k]))[0]

                    vrxx[:, foo] = np.nan

                    # Now interpolate to the heights of the retrieval
                    vrzz = np.ones((len(elxx), len(retz)))*-999
                    vr_varzz = np.ones((len(elxx), len(retz)))*-999

                    for ii in range(len(elxx)):
                        hgt = rngxx*np.sin(np.deg2rad(elxx[ii]))
                        vrzz[ii, :] = np.interp(
                            retz, hgt, vrxx[ii, :], left=-999, right=-999)

                    temp_sig, thresh_sig = Other_functions.wind_estimate(
                        vrzz, elxx, azxx, retz,vip['raw_lidar_eff_N'], vip['raw_lidar_sig_thresh'][k])
                    
                    vr_varzz[:, :] = temp_sig[None, :]

                    foo = np.where(vr_varzz > 90000)
                    vrzz[foo] = -999
                    
                    vrxx = np.copy(vrzz)
                    vr_varxx = np.copy(vr_varzz)

                    vrxx[np.isnan(vrxx)] = -999
                    vr_varxx[np.isnan(vr_varxx)] = -999

                    vrxx = vrxx.T
                    vr_varxx = vr_varxx.T

                    foo = np.where((vrxx != -999) & (vr_varxx < 3000))[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid MoDLS lidar data found')

                else:
                    print('No raw MoDLS data for retrieval at this time')
                    lsecsx = None
                    rngxx = None
                    azxx = None
                    elxx = None
                    vrxx = None
                    vr_varxx = None

        else:
            lsecsx = None
            rngxx = None
            azxx = None
            elxx = None
            vrxx = None
            vr_varxx = None


        if lsecsx is None:
            lsecs.append(np.copy(lsecsx))
            rng.append(np.copy(rngxx))
            az.append(np.copy(azxx))
            el.append(np.copy(elxx))
            vr.append(np.copy(vrxx))
            vr_var.append(np.copy(vr_varxx))
        else:
            lsecs.append(np.copy(lsecsx))
            rng.append(np.copy(rngxx))
            az.append(np.copy(azxx.ravel()))
            el.append(np.copy(elxx.ravel()))
            vr.append(np.copy(vrxx))
            vr_var.append(np.copy(vr_varxx))

    # Build the output dictionary and return it

    raw_lidar = {'success': 1, 'time': lsecs, 'range': rng, 'z': retz, 'azimuth': az, 'elevation': el,
                 'vr': vr, 'vr_var': vr_var, 'valid': available}

    return raw_lidar


def read_proc_lidar(date, retz, rtime, vip, verbose):
    """


    Parameters
    ----------
    date : int
        Date retrieval is run for
    retz : floats
        1-D array of retrieval heights
    vip : dict
        Dictionary containing all the namelist options
    verbose : int
        Controls the verbosity of the function

    Returns
    -------
    proc_lidar : dict
        Dictionary that contains key variables from the proc lidar files

    """

    lsecs = []
    u = []
    v = []
    u_error = []
    v_error = []

    available = np.zeros(vip['proc_lidar_number'])

    cdf = ['nc', 'cdf']

    # Loop over all user specified lidar types to be used
    for k in range(vip['proc_lidar_number']):
        no_data = True

        # Read in CLAMPS VAD file
        # For the error we are going to use the rms value for both u and v
        if vip['proc_lidar_type'][k] == 1:
            if verbose >= 1:
                print('Reading in unprocessed CLAMPS VAD file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(glob.glob(
                            vip['proc_lidar_paths'][k] + '/' + 'clampsdlvad*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print('No CLAMPS VAD files found in this directory for this date')
                lsecsx = None
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    foo = np.where((bt+to >= rtime-((vip['proc_lidar_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['proc_lidar_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()
                        continue

                    zx = fid.variables['height'][:]
                    wspd = fid.variables['wspd'][foo, :]
                    wdir = fid.variables['wdir'][foo, :]
                    rms = fid.variables['rms'][foo, :]

                    fid.close()

                    ux = -wspd * np.sin(np.deg2rad(wdir))
                    vx = -wspd * np.cos(np.deg2rad(wdir))

                    if no_data:
                        lsecsx = bt+to[foo]
                        zxx = np.copy(np.array([zx]*len(to[foo])))
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        u_errx = np.copy(rms)
                        v_errx = np.copy(rms)
                        no_data = False
                    else:
                        lsecsx = np.append(lsecsx, bt+to[foo])
                        zxx = np.vstack((zxx, np.array([zx]*len(to[foo]))))
                        uxx = np.vstack((uxx, ux))
                        vxx = np.vstack((vxx, vx))
                        u_errx = np.vstack((u_errx, rms))
                        v_errx = np.vstack((v_errx, rms))

                if not no_data:
                    zxx = zxx.T
                    uxx = uxx.T
                    vxx = vxx.T
                    u_errx = u_errx.T
                    v_errx = v_errx.T

                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx[:, 0], uxx, axis=0, bounds_error=False, fill_value=-999)
                    u_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], vxx, axis=0, bounds_error=False, fill_value=-999)
                    v_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], u_errx, axis=0, bounds_error=False, fill_value=-999)
                    uerr_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], v_errx, axis=0, bounds_error=False, fill_value=-999)
                    verr_interp = f(retz.data)

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((retz < vip['proc_lidar_minalt'][k]) |
                                   (retz > vip['proc_lidar_maxalt'][k]))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    foo = np.where(u_interp != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1
                else:
                    print('No CLAMPS VAD data for retrieval at this time')
                    lsecsx = None
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None

        # Read in NCAR (ARM) VAD file
        # For the error we are going to use the sum of wind component uncertainty
        # and the residual of the fit for both u and v
        elif vip['proc_lidar_type'][k] == 2:
            if verbose >= 1:
                print('Reading in processed NCAR (ARM) VAD file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                for j in range(len(cdf)):
                    files = files + \
                        sorted(glob.glob(
                            vip['proc_lidar_paths'][k] + '/' + 'VAD*' + dates[i] + '*.' + cdf[j]))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No NCAR (ARM) VAD files found in this directory for this date')
                lsecsx = None
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    foo = np.where((bt+to >= rtime-((vip['proc_lidar_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['proc_lidar_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()

                    zx = fid.variables['height'][:]
                    ux = fid.variables['u'][foo, :]
                    vx = fid.variables['v'][foo, :]
                    u_err = np.sqrt(fid.variables['u_error'][foo, :])
                    v_err = np.sqrt(fid.variables['v_error'][foo, :])
                    res = fid.variables['residual'][foo, :]

                    fid.close()

                    if no_data:
                        lsecsx = bt+to[foo]
                        zxx = np.copy(np.array([zx]*len(to[foo])))
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        u_errx = np.copy(u_err + res)
                        v_errx = np.copy(v_err + res)
                        no_data = False

                    else:
                        lsecsx = np.append(lsecsx, bt+to[foo])
                        zxx = np.vstack((zxx, np.array([zx]*len(to[foo]))))
                        uxx = np.vstack((uxx, ux))
                        vxx = np.vstack((vxx, vx))
                        u_errx = np.vstack((u_errx, u_err + res))
                        v_errx = np.vstack((v_errx, v_err + res))

                if not no_data:
                    zxx = zxx.T
                    uxx = uxx.T
                    vxx = vxx.T
                    u_errx = u_errx.T
                    v_errx = v_errx.T

                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx[:, 0], uxx, axis=0, bounds_error=False, fill_value=-999)
                    u_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], vxx, axis=0, bounds_error=False, fill_value=-999)
                    v_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], u_errx, axis=0, bounds_error=False, fill_value=-999)
                    uerr_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], v_errx, axis=0, bounds_error=False, fill_value=-999)
                    verr_interp = f(retz.data)

                    # Get rid of NaN values
                    foo = np.where(np.isnan(u_interp))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((retz < vip['proc_lidar_minrng'][k]) |
                                   (retz > vip['proc_lidar_maxrng'][k]))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    foo = np.where(u_interp != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1

                else:
                    print('No NCAR (ARM) VAD data for retrieval at this time')
                    lsecsx = None
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
            
        elif vip['proc_lidar_type'][k] == 3:
            
            if verbose >= 1:
                print('Reading in ZPH file')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]
            
            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(
                        vip['proc_lidar_paths'][k] + '/' + 'Wind_*Y' + dates[i][0:4] + '_M' + dates[i][4:6]
                        + '_D' + dates[i][6:8] + '.*'))
            
            if len(files) == 0:
                if verbose >= 1:
                    print('No ZPH files found in this directory for this date')
                lsecsx = None
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
                
            else:
                for i in range(len(files)):
                    
                    # These files are a pain to use. They are poorly formatted .csv
                    # with a lot of potential script breaking problems.
                    
                    inp = open(files[i]).readlines()[0].split(',')
                    height_array = str.split(inp[-1])[2:]
                    zx = np.array([int(i[:-1]) for i in height_array])[::-1]
                    timestamp = np.loadtxt(files[i], delimiter=',', usecols=(1,),dtype=str, unpack=True,skiprows=2)
                    
                    to_temp = []
                    if (timestamp[0][:-2] == 'AM') or (timestamp[0][:-2] == 'PM'):
                        fmt = '%d/%m/%Y %H:%M:%S %p'
                    else:
                        fmt = '%d/%m/%Y %H:%M:%S'
                        
                    for j in range(len(timestamp)):
                        try:
                            to_temp.append((datetime.strptime(timestamp[j],fmt)
                                           -datetime(1970,1,1)).total_seconds())
                        except:
                            print("Bad row in ZPH file")
                    
                    to_temp = np.array(to_temp)
                    
                    foo = np.where((to_temp >= rtime-((vip['proc_lidar_timedelta'][k]/2.)*60)) &
                                   (to_temp < rtime+((vip['proc_lidar_timedelta'][k]/2.)*60)))[0]
                    
                    # There is not data we want in this file so just move on
                    if len(foo) == 0:
                        continue
                    
                    sfc_wd = np.genfromtxt(files[i], comments = '#@!$', delimiter=',',usecols=(17),dtype=None,skip_header=2,missing_values=('#N/A'),filling_values=(9999))
                    sfc_wd[sfc_wd==9999] = np.nan
                    
                    for j in range(0,len(zx)):
                        try:
                            if j == 0:
                                wd = np.genfromtxt(files[i], comments = '#@!$', delimiter=',',usecols=(19 + j*3),dtype=None,skip_header=2,missing_values=('#N/A'),filling_values=(9999))
                                ws = np.genfromtxt(files[i], comments = '#@!$', delimiter=',',usecols=(20 + j*3),dtype=None,skip_header=2,missing_values=('#N/A'),filling_values=(9999))
                            else:
                                wd = np.vstack((wd,np.genfromtxt(files[i], comments = '#@!$', delimiter=',',usecols=(19 + j*3),dtype=None,skip_header=2,missing_values=('#N/A'),filling_values=(9999))))
                                ws = np.vstack((ws,np.genfromtxt(files[i], comments = '#@!$', delimiter=',',usecols=(20 + j*3),dtype=None,skip_header=2,missing_values=('#N/A'),filling_values=(9999))))
                        
                        except:
                            if j == 0:
                                wd = np.ones(len(to_temp))*np.nan
                                ws = np.ones(len(to_temp))*np.nan
                            else:
                                wd = np.vstack((wd,np.ones(len(to_temp))*np.nan))
                                ws = np.vstack((ws,np.ones(len(to_temp))*np.nan))
                            print("Bad wind data in ZPH file")
                        
                    wd[wd==9999] = np.nan
                    ws[ws==9999] = np.nan
                    
                    # We need to make sure the wd is right by comparing the first level wd
                    # to the surface wd if the difference is greater than 90 then correct it
                    
                    #wd_dif = (np.rad2deg(np.arctan2(np.cos(np.deg2rad(wd[-1])),np.sin(np.deg2rad(wd[-1])))) -
                    #         np.rad2deg(np.arctan2(np.cos(np.deg2rad(sfc_wd)),np.sin(np.deg2rad(sfc_wd)))))
                    
                    #fah = np.where((np.abs(wd_dif) >90))[0]
                    
                    #wd[:,fah] = wd[:,fah] + 180
                    
                    #fah = np.where(wd > 360)
                    #wd[fah] = wd[fah]-360
                    
                    #fah = np.where(np.isnan(sfc_wd))[0]
                    
                    #wd[:,fah] = np.nan
                    
                    ux = np.array(np.sin(np.deg2rad(wd-180))*ws).transpose()
                    vx = np.array(np.cos(np.deg2rad(wd-180))*ws).transpose()
                    
                    ux = ux[:,::-1]
                    vx = vx[:,::-1]
                    
                    if no_data:
                        lsecsx = np.copy(to_temp[foo])
                        zxx = np.copy(np.array([zx]*len(to_temp[foo])))
                        uxx = np.copy(ux[foo,:])
                        vxx = np.copy(vx[foo,:])
                        no_data = False

                    else:
                        lsecsx = np.append(lsecsx, to_temp[foo])
                        zxx = np.vstack((zxx, np.array([zx]*len(to_temp[foo]))))
                        uxx = np.vstack((uxx, ux[foo,:]))
                        vxx = np.vstack((vxx, vx[foo,:]))
                
                if not no_data:
                    zxx = zxx.T/1000.
                    uxx = uxx.T
                    vxx = vxx.T
                    
                    # For this type of data we are going to average the scans
                    # and the variance of the wind speed will be the variance
                    # of the data
                    uxx_mean = np.nanmean(uxx,axis=1)
                    vxx_mean = np.nanmean(vxx,axis=1)
                    
                    u_err = np.nanstd(uxx,axis=1)
                    v_err = np.nanstd(vxx,axis=1)
                    
                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx[:, 0], uxx_mean, axis=0, bounds_error=False, fill_value=-999)
                    u_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], vxx_mean, axis=0, bounds_error=False, fill_value=-999)
                    v_interp = f(retz.data)
                    
                    f = interpolate.interp1d(
                        zxx[:, 0], u_err, axis=0, bounds_error=False, fill_value=-999)
                    uerr_interp = f(retz.data)

                    f = interpolate.interp1d(
                        zxx[:, 0], v_err, axis=0, bounds_error=False, fill_value=-999)
                    verr_interp = f(retz.data)
                    
                    # Get rid of NaN values
                    foo = np.where(np.isnan(u_interp))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((retz < vip['proc_lidar_minalt'][k]) |
                                   (retz > vip['proc_lidar_maxalt'][k]))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    foo = np.where(u_interp != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1

                else:
                    print('No ZPH data for retrieval at this time')
                    lsecsx = None
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                

        lsecs.append(lsecsx)
        u.append(u_interp)
        v.append(v_interp)
        u_error.append(uerr_interp)
        v_error.append(verr_interp)

    # Build the output dictionary and return it

    proc_lidar = {'success': 1, 'time': lsecs, 'height': np.copy(retz), 'u': u, 'v': v,
                  'u_error': u_error, 'v_error': v_error, 'valid': available}

    return proc_lidar


def read_prof_cons(date, retz, rtime, vip, verbose):

    available = 0
    no_data = True
    cdf = ['nc', 'cdf']

    # Read in NCAR
    # For the error we are going to use the variance of the wind or a default value

    if vip['cons_profiler_type'] == 1:
        if verbose >= 1:
            print('Reading in NCAR 449Mhz consensus winds file')

        dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                 str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + \
                    sorted(glob.glob(vip['cons_profiler_path'] +
                           '/' + 'prof449.' + dates[i] + '*.' + cdf[j]))

        if len(files) == 0:
            if verbose >= 1:
                print(
                    'No NCAR 449Mhz consensus wind files found in this directory for this date')
            lsecsx = None
            u_interp = None
            v_interp = None
            uerr_interp = None
            verr_interp = None
        else:
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                bt = fid.variables['base_time'][0]
                to = fid.variables['time_offset'][:]

                foo = np.where((bt+to >= rtime-((vip['cons_profiler_timedelta']/2.)*60)) &
                               (bt+to < rtime+((vip['cons_profiler_timedelta']/2.)*60)))[0]

                # There are no times we want here so just move on
                if len(foo) == 0:
                    fid.close()
                    continue

                zx = fid.variables['height'][foo, :]/1000.
                wspd = fid.variables['wspd'][foo, :]
                wdir = fid.variables['wdir'][foo, :]

                fid.close()

                ux = -wspd * np.sin(np.deg2rad(wdir))
                vx = -wspd * np.cos(np.deg2rad(wdir))

                if no_data:
                    lsecsx = bt+to[foo]
                    zxx = np.copy(zx)
                    uxx = np.copy(ux)
                    vxx = np.copy(vx)
                    u_errx = np.ones(uxx.shape)*2
                    v_errx = np.ones(vxx.shape)*2

                    no_data = False
                else:
                    lsecsx = np.append(lsecsx, bt+to[foo])
                    zxx = np.vstack((zxx, zx))
                    uxx = np.vstack((uxx, ux))
                    vxx = np.vstack((vxx, vx))
                    u_errx = np.ones(uxx.shape)*2
                    v_errx = np.ones(vxx.shape)*2

            if not no_data:
                zxx = zxx.T
                uxx = uxx.T
                vxx = vxx.T
                u_errx = u_errx.T
                v_errx = v_errx.T

                foo = np.where(np.abs(uxx) == 999.)
                uxx[foo] = np.nan
                vxx[foo] = np.nan
                u_errx[foo] = np.nan
                v_errx[foo] = np.nan

                # Interpolate the data to the retrieval vertical grid
                f = interpolate.interp1d(
                    zxx[:, 0], uxx, axis=0, bounds_error=False, fill_value=-999)
                u_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[:, 0], vxx, axis=0, bounds_error=False, fill_value=-999)
                v_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[:, 0], u_errx, axis=0, bounds_error=False, fill_value=-999)
                uerr_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[:, 0], v_errx, axis=0, bounds_error=False, fill_value=-999)
                verr_interp = f(np.copy(retz))

                # Get rid of NaN values
                foo = np.where(np.isnan(u_interp))

                u_interp[foo] = -999.
                v_interp[foo] = -999.
                uerr_interp[foo] = -999.
                verr_interp[foo] = -999.

                # We only want to use data between min range and max range so set
                # everything else to missing

                foo = np.where((retz < vip['cons_profiler_minalt']) |
                               (retz > vip['cons_profiler_maxalt']))

                u_interp[foo] = -999.
                v_interp[foo] = -999.
                uerr_interp[foo] = -999.
                verr_interp[foo] = -999.

                foo = np.where(u_interp != -999.)[0]
                if len(foo) > 0:
                    available = 1
                else:
                    print('No valid consensus wind profiler data found')
            
    elif vip['cons_profiler_type'] == 2:
        if verbose >= 1:
            print('Reading in UAH 915MHz consensus winds file')

        dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                 str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

        files = []
        for i in range(len(dates)):
            for j in range(len(cdf)):
                files = files + \
                    sorted(glob.glob(vip['cons_profiler_path'] +
                           '/' + '*915*' + dates[i] + '*.' + cdf[j]))

        if len(files) == 0:
            if verbose >= 1:
                print(
                    'No UAH 915MHz consensus wind files found in this directory for this date')
            lsecsx = None
            u_interp = None
            v_interp = None
            uerr_interp = None
            verr_interp = None
        else:
            for i in range(len(files)):
                if verbose >= 3:
                    print(f"   Reading RWP file {files[i]}")

                fid = Dataset(files[i], 'r')
                bt = fid.variables['epochTime'][0]
                to = fid.variables['epochTime'][:] - bt

                foo = np.where((bt+to >= rtime-((vip['cons_profiler_timedelta']/2.)*60)) &
                               (bt+to < rtime+((vip['cons_profiler_timedelta']/2.)*60)))[0]

                # There are no times we want here so just move on
                if len(foo) == 0:
                    fid.close()
                    continue

                zx = fid.variables['height'][:]/1000.
                ux = fid.variables['u'][foo, :]
                vx = fid.variables['v'][foo, :]

                # Replace UAH missing value with WINDoe missing value
                # ux = np.where(ux == 999.9, -999, ux)
                # vx = np.where(vx == 999.9, -999, vx)

                fid.close()

                if no_data:
                    lsecsx = bt+to[foo]
                    zxx = np.copy(zx)
                    uxx = np.copy(ux)
                    vxx = np.copy(vx)
                    u_errx = np.ones(uxx.shape)*2
                    v_errx = np.ones(vxx.shape)*2

                    no_data = False
                else:
                    lsecsx = np.append(lsecsx, bt+to[foo])
                    zxx = np.vstack((zxx, zx))
                    uxx = np.vstack((uxx, ux))
                    vxx = np.vstack((vxx, vx))
                    u_errx = np.ones(uxx.shape)*2
                    v_errx = np.ones(vxx.shape)*2

            if not no_data:
                zxx = zxx.T
                uxx = uxx.T
                vxx = vxx.T
                u_errx = u_errx.T
                v_errx = v_errx.T

                foo = np.where(np.abs(uxx) >= 999.)
                uxx[foo] = np.nan
                vxx[foo] = np.nan
                u_errx[foo] = np.nan
                v_errx[foo] = np.nan

                # Interpolate the data to the retrieval vertical grid
                f = interpolate.interp1d(
                    zxx, uxx, axis=0, bounds_error=False, fill_value=-999)
                u_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx, vxx, axis=0, bounds_error=False, fill_value=-999)
                v_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx, u_errx, axis=0, bounds_error=False, fill_value=-999)
                uerr_interp = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx, v_errx, axis=0, bounds_error=False, fill_value=-999)
                verr_interp = f(np.copy(retz))

                # Get rid of NaN values
                foo = np.where(np.isnan(u_interp))

                u_interp[foo] = -999.
                v_interp[foo] = -999.
                uerr_interp[foo] = -999.
                verr_interp[foo] = -999.

                # We only want to use data between min range and max range so set
                # everything else to missing

                foo = np.where((retz < vip['cons_profiler_minalt']) |
                               (retz > vip['cons_profiler_maxalt']))

                u_interp[foo] = -999.
                v_interp[foo] = -999.
                uerr_interp[foo] = -999.
                verr_interp[foo] = -999.

                foo = np.where(u_interp != -999.)[0]
                if len(foo) > 0:
                    available = 1
                else:
                    print('No valid consensus wind profiler data found')

            else:
                print('No consensus wind profiler data for retrieval at this time')
                lsecsx = None
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None

    # Build the output dictionary and return it

    cons_profiler = {'success': 1, 'time': lsecsx, 'height': np.copy(retz), 'u': u_interp, 'v': v_interp,
                     'u_error': uerr_interp, 'v_error': verr_interp, 'valid': available}

    return cons_profiler


def read_raw_prof(date, retz, rtime, vip, verbose):
    
    lsecs = []
    az = []
    el = []
    vr = []
    vr_var = []

    available = np.zeros(vip['raw_profiler_number'])
    

    # Read in ARM wind profiler data
    # For the error we are going to use the variance of the data over the time wind profiler window
    for k in range(vip['raw_profiler_number']):
        no_data = True
        if vip['raw_profiler_type'][k] == 1:
            if verbose >= 1:
                print('Reading in ARM raw wind profiler file')
                
            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]
            
            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['raw_profiler_paths'][k] +
                                     '/' + 'sgp915*' + dates[i] + '*.nc'))
            
            if len(files) == 0:
                if verbose >= 1:
                    print('No ARM raw wind profiler data found for this date')
                lsecsx = None
                vrxx = None
                vr_err = None
                elxx = None
                azxx = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]
                    
                    foo = np.where((bt+to >= rtime-((vip['raw_profiler_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['raw_profiler_timedelta'][k]/2.)*60)))[0]
                        
                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        fid.close()
                        continue
                        
                    zx_tmp = fid.variables['height'][:]/1000.
                    vr_tmp = -fid.variables['radial_velocity'][foo, :]
                    az_tmp = fid.variables['beam_azimuth'][foo]
                    el_tmp = fid.variables['beam_zenith'][foo]
                    width_tmp = fid.variables['spectral_width'][foo, :]
                    
                    if no_data:
                        lsecsx = bt+to[foo]
                        zxx = np.copy(np.array([zx_tmp]*len(to[foo])))
                        vrx = np.copy(vr_tmp)
                        azx = np.copy(az_tmp)
                        elx = np.copy(el_tmp)
                        widthx = np.copy(width_tmp)
                        no_data = False
                    else:
                        lsecsx = np.append(lsecsx, bt+to[foo])
                        zxx = np.vstack((zxx, np.array([zx_tmp]*len(to[foo]))))
                        vrx = np.vstack((vrx, vr_tmp))
                        azx = np.append(azx, az_tmp)
                        elx = np.append(elx, el_tmp)
                        widthx = np.vstack((widthx, width_tmp))

                # Need to sort the radial velocities into arrays by azimuths
                # First need to check the elevations though

                if not no_data:
                    foo = np.where(elx > 5)[0]
                    if len(foo) == 0:
                        no_data = True
                        
                if not no_data:
                    zxx = zxx[0]

                    lsecsx = lsecsx[foo]
                    vrx = vrx[foo, :]
                    azx = azx[foo]
                    elx = elx[foo]
                    widthx = widthx[foo]
                    
                    vrx[np.abs(vrx) > 100.] = np.nan
                    
                    # Find unique azimuths
                    azxx = np.unique(azx)
                    elxx = elx[0]
                    vrxx = []
                    vr_err = []
                    
                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx[:], vrx, axis=1, bounds_error=False, fill_value=-999)
                    vrx = f(np.copy(retz))

                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx[:], widthx, axis=1, bounds_error=False, fill_value=-999)
                    widthx = f(np.copy(retz))
                    
                    for i in range(len(azxx)):
                        foo = np.where(azx == azxx[i])[0]
                        if len(foo) > 0:
                            temp1, temp2 = Other_functions.consensus_average(vrx[foo, :], widthx[foo, :], vip['consensus_cutoff'][k],
                                                                             vip['consensus_min_pct'][k])
                            vrxx.append(np.copy(temp1))
                            vr_err.append(np.copy(np.sqrt(temp2)))
                        else:
                            vrxx.append(np.ones(len(retz))*-999.)
                            vr_err.append(np.ones(len(retz))*-999.)
                            
                    vrxx = np.array(vrxx)
                    vr_err = np.array(vr_err)
                            
                    # Get rid of NaN values
                    foo = np.where(np.isnan(vrxx))
                    vrxx[foo] = -999.
                    vr_err[foo] = -999.
                    
                    # We only want to use data between min range and max range so set
                    # everything else to missing
                        
                    foo = np.where((retz < vip['raw_profiler_minalt'][k]) |
                                   (retz > vip['raw_profiler_maxalt'][k]))[0]
                        
                    vrxx[:, foo] = -999.
                    vr_err[:, foo] = -999.
                        
                    vrxx = vrxx.T
                    vr_err = vr_err.T
                        
                    foo = np.where(vrxx != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1
                    else:
                        print('No valid raw wind profiler data for this time found')

                else:
                    print('No raw wind profiler data for retrieval at this time')
                    lsecsx = None
                    vrxx = None
                    vr_err = None
                    elxx = None
                    azxx = None
    
        if lsecsx is None:
            lsecs.append(np.copy(lsecsx))
            az.append(np.copy(azxx))
            el.append(np.copy(elxx))
            vr.append(np.copy(vrxx))
            vr_var.append(np.copy(vr_err))
        else:
            lsecs.append(np.copy(lsecsx))
            az.append(np.copy(azxx.ravel()))
            el.append(np.copy(elxx.ravel()))
            vr.append(np.copy(vrxx))
            vr_var.append(np.copy(vr_err))
                    
    # Build the output dictionary and return it

    raw_profiler = {'success': 1, 'time': lsecs, 'height': np.copy(retz), 'vr': vr, 'el': el,
                    'vr_error': vr_var, 'az': az, 'valid': available}

    return raw_profiler


def read_insitu(date, retz, rtime, vip, verbose):

    u = []
    v = []
    u_error = []
    v_error = []
    zs = []

    available = np.zeros(vip['insitu_number'])

    # Loop over all user specified insitu types to be used
    for k in range(vip['insitu_number']):
        no_data = True

        if vip['insitu_type'][k] == 1:
            if verbose >= 1:
                print('Reading in NCAR Tower data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths']
                           [k] + '/' + 'isfs_' + dates[i] + '.nc'))

            if len(files) == 0:
                if verbose >= 1:
                    print(
                        'No NCAR Tower data files found in this directory for this date')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
                z_interp = None
            else:
                dif = 1000000
                switch = False
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    # We want the profile closest to the analysis time that fall into the window
                    foo = np.nanargmin(np.abs((bt+to) - rtime))

                    # There are no times we want here so just move on
                    if np.abs((bt+to)[foo] - rtime) > (vip['insitu_timedelta'][k]*60.):
                        fid.close()
                        continue

                    if np.abs((bt+to)[foo] - rtime) < dif:
                        dif = np.abs((bt+to)[foo] - rtime)
                        switch = True

                    zx = fid.variables['alt'][:]/1000.
                    ux = fid.variables['u_wind'][foo, :]
                    vx = fid.variables['v_wind'][foo, :]

                    ux_sigma = fid.variables['sigma_u'][foo, :]
                    vx_sigma = fid.variables['sigma_v'][foo, :]

                    fid.close()

                    if no_data or switch:
                        zxx = np.copy(zx)
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        uxx_sigma = np.copy(ux_sigma)
                        vxx_sigma = np.copy(vx_sigma)
                        no_data = False
                        switch = False

                if not no_data:
                    foo = np.where(np.abs(uxx) == 999.)
                    uxx[foo] = np.nan
                    vxx[foo] = np.nan
                    uxx_sigma[foo] = np.nan
                    vxx_sigma[foo] = np.nan

                    # Interpolate the data to the retrieval vertical grid
                    f = interpolate.interp1d(
                        zxx, uxx, axis=0, bounds_error=False, fill_value=-999)
                    u_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        zxx, vxx, axis=0, bounds_error=False, fill_value=-999)
                    v_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        zxx, uxx_sigma, axis=0, bounds_error=False, fill_value=-999)
                    uerr_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        zxx, vxx_sigma, axis=0, bounds_error=False, fill_value=-999)
                    verr_interp = f(np.copy(retz))

                    z_interp = np.coppy(retz)

                    # We only want to use data between min range and max range so set
                    # everything else to missing

                    foo = np.where((retz < vip['insitu_minalt'][k]) |
                                   (retz > vip['insitu_maxalt'][k]))

                    u_interp[foo] = -999.
                    v_interp[foo] = -999.
                    uerr_interp[foo] = -999.
                    verr_interp[foo] = -999.

                    u_interp[np.isnan(u_interp)] = -999.
                    v_interp[np.isnan(v_interp)] = -999.
                    uerr_interp[np.isnan(uerr_interp)] = -999.
                    verr_interp[np.isnan(verr_interp)] = -999.

                    foo = np.where(u_interp != -999.)[0]
                    if len(foo) > 0:
                        available[k] = 1

                else:
                    print('No NCAR tower data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None

        if vip['insitu_type'][k] == 2:
            if verbose >= 1:
                print('Reading in ARM met station data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths'][k] +
                           '/' + '*met*' + dates[i] + '*.cdf'))

            if len(files) == 0:
                if verbose >= 1:
                    print('No ARM met station data found for this time')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
                z_interp = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    # We want the profile closest to the analysis time that fall into the window
                    foo = np.nanargmin(np.abs((bt+to) - rtime))

                    foo = np.where((bt+to >= rtime-((vip['insitu_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['insitu_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        continue

                    sx = fid.variables['wspd_vec_mean'][foo]
                    wdx = fid.variables['wdir_vec_mean'][foo]

                    fid.close()

                    ux = -sx*np.sin(np.deg2rad(wdx))
                    vx = -sx*np.cos(np.deg2rad(wdx))

                    if no_data:
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        no_data = False

                    else:
                        uxx = np.append(uxx, ux)
                        vxx = np.append(vxx, vx)

                if not no_data:
                    foo = np.where((np.abs(uxx) > 100.) | (np.abs(vxx) > 100.))
                    uxx[foo] = np.nan
                    vxx[foo] = np.nan

                    foo = np.where(~np.isnan(uxx))[0]
                    if len(foo) < 1:
                        u_interp = -999.
                        v_interp = -999.,
                        uerr_interp = -999.
                        verr_interp = -999.

                    # Now average the data and get the variance
                    u_interp = np.nanmean(uxx)
                    v_interp = np.nanmean(vxx)

                    uerr_interp = np.nanstd(uxx)
                    verr_interp = np.nanstd(vxx)

                    if uerr_interp < 1:
                        uerr_interp = 1.0

                    if verr_interp < 1:
                        verr_interp = 1.0

                    # This is all 10-m data
                    z_interp = np.array([retz[0]])

                    u_interp = np.array([u_interp])
                    v_interp = np.array([v_interp])
                    uerr_interp = np.array([uerr_interp])
                    verr_interp = np.array([verr_interp])

                    if u_interp[0] > -100:
                        available[k] = 1

                else:
                    print('No ARM met station data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None

        if vip['insitu_type'][k] == 3:
            if verbose >= 1:
                print('Reading in CLAMPS met station data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths'][k] +
                           '/' + '*mwr*' + dates[i] + '*.cdf'))

            if len(files) == 0:
                if verbose >= 1:
                    print('No CLAMPS met station data found for this time')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
                z_interp = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][:]

                    # We want the profile closest to the analysis time that fall into the window
                    foo = np.nanargmin(np.abs((bt+to) - rtime))

                    foo = np.where((bt+to >= rtime-((vip['insitu_timedelta'][k]/2.)*60)) &
                                   (bt+to < rtime+((vip['insitu_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        continue

                    sx = fid.variables['sfc_wspd'][foo]
                    wdx = fid.variables['sfc_wdir'][foo]

                    fid.close()

                    ux = -sx*np.sin(np.deg2rad(wdx))
                    vx = -sx*np.cos(np.deg2rad(wdx))

                    if no_data:
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        no_data = False

                    else:
                        uxx = np.append(uxx, ux)
                        vxx = np.append(vxx, vx)

                if not no_data:
                    foo = np.where((np.abs(uxx) > 100.) | (np.abs(vxx) > 100.))
                    uxx[foo] = np.nan
                    vxx[foo] = np.nan

                    foo = np.where(~np.isnan(uxx))[0]
                    if len(foo) < 1:
                        u_interp = -999.
                        v_interp = -999.,
                        uerr_interp = -999.
                        verr_interp = -999.

                    # Now average the data and get the variance
                    u_interp = np.nanmean(uxx)
                    v_interp = np.nanmean(vxx)

                    uerr_interp = np.nanstd(uxx)
                    verr_interp = np.nanstd(vxx)

                    # This is all 10-m data
                    z_interp = np.array([retz[0]])

                    u_interp = np.array([u_interp])
                    v_interp = np.array([v_interp])
                    uerr_interp = np.array([uerr_interp])
                    verr_interp = np.array([verr_interp])

                    if u_interp[0] > -100:
                        available[k] = 1

                else:
                    print('No ARM met station data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None

        if vip['insitu_type'][k] == 4:
            if verbose >= 1:
                print('Reading in sounding data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths']
                           [k] + '/' + '*' + dates[i] + '*.nc'))

            files.sort()
            if len(files) == 0:
                if verbose >= 1:
                    print('No sounding data files found in this directory for this date')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
            else:
                for i in range(len(files)):
                    if verbose >= 3:
                        print(f"    Reading file {files[i]}")

                    fid = Dataset(files[i], 'r')
                    bt = fid.variables['base_time'][0]
                    to = fid.variables['time_offset'][0]

                    z = fid.variables['height'][:]/1000.
                    ux = fid.variables['u'][:]
                    vx = fid.variables['v'][:]

                    ux_sigma = np.ones_like(z) * 5
                    vx_sigma = np.ones_like(z) * 5

                    fid.close()
                    
                    # We need this dat interpolated to the retrieval heights
                    f = interpolate.interp1d(
                        z, ux, bounds_error=False, fill_value=np.nan)
                    u_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, vx, bounds_error=False, fill_value=np.nan)
                    v_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, ux_sigma, bounds_error=False, fill_value=np.nan)
                    uerr_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, vx_sigma, bounds_error=False, fill_value=np.nan)
                    verr_interp = f(np.copy(retz))
                    
                    if no_data:
                        secs = np.array([bt+to])
                        uxx = np.array([u_interp])
                        vxx = np.array([v_interp])
                        uxx_sigma = np.array([uerr_interp])
                        vxx_sigma = np.array([verr_interp])
                        no_data = False
                    else:
                        secs = np.append(secs, bt+to)
                        uxx = np.vstack((uxx, u_interp))
                        vxx = np.vstack((vxx, v_interp))
                        uxx_sigma = np.vstack((uxx_sigma, uerr_interp))
                        vxx_sigma = np.vstack((vxx_sigma, verr_interp))
                
                if not no_data:
                    
                    # Now find the profiles closest before and after the retrieval time
                    foo_before = np.where(secs <= rtime)[0]
                    foo_after = np.where(secs > rtime)[0]
                    z_interp = np.copy(retz)

                    # No sounding data before retrieval time
                    if len(foo_before) == 0:

                        # No sounding data after retrieval time either. This is weird
                        if len(foo_after) == 0:
                            print('Something really weird is happening in sounding read.')
                            print('No sounding data used for this time')
                            u_interp = None
                            v_interp = None
                            uerr_interp = None
                            verr_interp = None
                            z_interp = None

                        # We have sounding data after the retrieval time but not before
                        else:
                            after_sec = secs[foo_after[0]]
                            if after_sec > rtime+(vip['insitu_timedelta'][k]/2)*60:
                                print('No sounding data with in timedelta for this time.')
                            else:
                                u_interp = uxx[foo_after[0], :]
                                v_interp = vxx[foo_after[0], :]
                                uerr_interp = uxx_sigma[foo_after[0], :]
                                verr_interp = vxx_sigma[foo_after[0], :]

                                u_interp[np.isnan(u_interp)] = -999.
                                v_interp[np.isnan(v_interp)] = -999.
                                uerr_interp[np.isnan(uerr_interp)] = -999.
                                verr_interp[np.isnan(verr_interp)] = -999.

                                foo = np.where(u_interp != -999)[0]

                                if len(foo) > 0:
                                    available[k] = 1
                                else:
                                    print('No valid model data found')

                    else:
                        # We have sounding data before the retrieval time, but not after
                        if len(foo_after) == 0:
                            before_sec = secs[foo_before[-1]]

                            if before_sec < rtime-(vip['insitu_timedelta'][k]/2)*60:
                                print('No sounding data with in timedelta for this time.')
                            else:
                                u_interp = uxx[foo_before[-1], :]
                                v_interp = vxx[foo_before[-1], :]
                                uerr_interp = uxx_sigma[foo_before[-1], :]
                                verr_interp = vxx_sigma[foo_before[-1], :]

                                u_interp[np.isnan(u_interp)] = -999.
                                v_interp[np.isnan(v_interp)] = -999.
                                uerr_interp[np.isnan(uerr_interp)] = -999.
                                verr_interp[np.isnan(verr_interp)] = -999.

                                foo = np.where(u_interp != -999)[0]

                                if len(foo) > 0:
                                    available[k] = 1
                                else:
                                    print('No valid sounding data found')

                        else:

                            # We have data before and after the retrieval time so interpolate the data in time
                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], uxx[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            u_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], vxx[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            v_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], uxx_sigma[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            uerr_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], vxx_sigma[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            verr_interp = f(rtime)

                            foo = np.where(u_interp != -999.)[0]

                            if len(foo) > 0:
                                available[k] = 1
                            else:
                                print('No valid sounding data found')

                        foo = np.where((retz < vip['insitu_minalt'][k]) |
                                   (retz > vip['insitu_maxalt'][k]))

                        u_interp[foo] = -999.
                        v_interp[foo] = -999.
                        uerr_interp[foo] = -999.
                        verr_interp[foo] = -999.
                else:
                    print('No sounding data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None
            
        if vip['insitu_type'][k] == 5:
            if verbose >= 1:
                print('Reading in mobile mesonet data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths'][k] +
                           '/' + '*' + dates[i] + '_MM.nc'))

            if len(files) == 0:
                if verbose >= 1:
                    print('No mobile mesonet data found for this time')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
                z_interp = None
            else:
                for i in range(len(files)):
                    fid = Dataset(files[i], 'r')
                    t = fid.variables['epochtime'][:]

                    # We want the profile closest to the analysis time that fall into the window
                    foo = np.nanargmin(np.abs((t) - rtime))

                    foo = np.where((t >= rtime-((vip['insitu_timedelta'][k]/2.)*60)) &
                                   (t < rtime+((vip['insitu_timedelta'][k]/2.)*60)))[0]

                    # There are no times we want here so just move on
                    if len(foo) == 0:
                        continue

                    sx = fid.variables['sfc_wspd'][foo]
                    wdx = fid.variables['sfc_wdir'][foo]
                    qcx = np.zeros_like(wdx)

                    fid.close()

                    ux = -sx*np.sin(np.deg2rad(wdx))
                    vx = -sx*np.cos(np.deg2rad(wdx))

                    if no_data:
                        uxx = np.copy(ux)
                        vxx = np.copy(vx)
                        qcxx = np.copy(qcx)
                        no_data = False

                    else:
                        uxx = np.append(uxx, ux)
                        vxx = np.append(vxx, vx)
                        qcxx = np.append(qcxx,qcx)

                if not no_data:
                    foo = np.where((np.abs(uxx) > 100.) | (np.abs(vxx) > 100.))
                    uxx[foo] = np.nan
                    vxx[foo] = np.nan
                    
                    # foo = np.where(qcxx==1)
                    # uxx[foo] = np.nan
                    # vxx[foo] = np.nan

                    foo = np.where(~np.isnan(uxx))[0]
                    if len(foo) < 1:
                        u_interp = None
                        v_interp = None
                        uerr_interp = None
                        verr_interp = None
                        z_interp = None
                        continue

                    # Now average the data and get the variance
                    u_interp = np.nanmean(uxx)
                    v_interp = np.nanmean(vxx)
                    
                    uerr_interp = np.nanstd(uxx)
                    verr_interp = np.nanstd(vxx)

                    # This is all 10-m data
                    z_interp = np.array([retz[0]])

                    u_interp = np.array([u_interp])
                    v_interp = np.array([v_interp])
                    uerr_interp = np.array([uerr_interp])
                    verr_interp = np.array([verr_interp])

                    if u_interp[0] > -100:
                        available[k] = 1

                else:
                    print('No mobile mesonet data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None
        if vip['insitu_type'][k] == 6:
            if verbose >= 1:
                print('Reading in windsonde data')

            dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
                     str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

            files = []
            for i in range(len(dates)):
                files = files + \
                    sorted(glob.glob(vip['insitu_paths']
                           [k] + '/' + '*' + dates[i] + '*.csv'))

            files.sort()
            if len(files) == 0:
                if verbose >= 1:
                    print('No windsonde data files found in this directory for this date')
                u_interp = None
                v_interp = None
                uerr_interp = None
                verr_interp = None
            else:
                for i in range(len(files)):
                    date, time, z, dir, spd = np.loadtxt(files[i], delimiter = ',', usecols = [0, 2, 6, 11, 12], unpack = True, dtype = str, skiprows = 1)
                    
                    # convert the date/times to epochtime
                    epoch = (datetime.strptime(date[0] + time[0], '%Y-%m-%d%H:%M:%S') - datetime(1970,1,1)).total_seconds()

                    # mask missing data as nan
                    spd[spd == ''] = np.nan
                    dir[np.isnan(spd.astype('f')) == True] = np.nan
                    
                    # convert data to float and format properly before interpolation 
                    z = z.astype('f')/1000.
                    ux = -1 * spd.astype('f') * np.sin(dir.astype('f') * (np.pi/180))
                    vx = -1 * spd.astype('f') * np.cos(dir.astype('f') * (np.pi/180))

                    ux_sigma = np.ones_like(z) * 6
                    vx_sigma = np.ones_like(z) * 6
                    
                    # We need this data interpolated to the retrieval heights
                    f = interpolate.interp1d(
                        z, ux, bounds_error=False, fill_value=np.nan)
                    u_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, vx, bounds_error=False, fill_value=np.nan)
                    v_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, ux_sigma, bounds_error=False, fill_value=np.nan)
                    uerr_interp = f(np.copy(retz))

                    f = interpolate.interp1d(
                        z, vx_sigma, bounds_error=False, fill_value=np.nan)
                    verr_interp = f(np.copy(retz))
                    
                    if no_data:
                        secs = epoch
                        uxx = np.array([u_interp])
                        vxx = np.array([v_interp])
                        uxx_sigma = np.array([uerr_interp])
                        vxx_sigma = np.array([verr_interp])
                        no_data = False
                    else:
                        secs = np.append(secs, epoch)
                        uxx = np.vstack((uxx, u_interp))
                        vxx = np.vstack((vxx, v_interp))
                        uxx_sigma = np.vstack((uxx_sigma, uerr_interp))
                        vxx_sigma = np.vstack((vxx_sigma, verr_interp))
                
                if not no_data:
                    
                    # Now find the profiles closest before and after the retrieval time
                    foo_before = np.where(secs <= rtime)[0]
                    foo_after = np.where(secs > rtime)[0]
                    z_interp = np.copy(retz)

                    # No sounding data before retrieval time
                    if len(foo_before) == 0:

                        # No sounding data after retrieval time either. This is weird
                        if len(foo_after) == 0:
                            print('Something really weird is happening in windsonde read.')
                            print('No windsonde data used for this time')
                            u_interp = None
                            v_interp = None
                            uerr_interp = None
                            verr_interp = None
                            z_interp = None

                        # We have sounding data after the retrieval time but not before
                        else:
                            after_sec = secs[foo_after[0]]
                            if after_sec > rtime+(vip['insitu_timedelta'][k]/2)*60:
                                print('No windsonde data with in timedelta for this time.')
                            else:
                                u_interp = uxx[foo_after[0], :]
                                v_interp = vxx[foo_after[0], :]
                                uerr_interp = uxx_sigma[foo_after[0], :]
                                verr_interp = vxx_sigma[foo_after[0], :]

                                u_interp[np.isnan(u_interp)] = -999.
                                v_interp[np.isnan(v_interp)] = -999.
                                uerr_interp[np.isnan(uerr_interp)] = -999.
                                verr_interp[np.isnan(verr_interp)] = -999.

                                foo = np.where(u_interp != -999)[0]

                                if len(foo) > 0:
                                    available[k] = 1
                                else:
                                    print('No valid windsonde data found')

                    else:
                        # We have sounding data before the retrieval time, but not after
                        if len(foo_after) == 0:
                            before_sec = secs[foo_before[-1]]

                            if before_sec < rtime-(vip['insitu_timedelta'][k]/2)*60:
                                print('No windsonde data with in timedelta for this time.')
                            else:
                                u_interp = uxx[foo_before[-1], :]
                                v_interp = vxx[foo_before[-1], :]
                                uerr_interp = uxx_sigma[foo_before[-1], :]
                                verr_interp = vxx_sigma[foo_before[-1], :]

                                u_interp[np.isnan(u_interp)] = -999.
                                v_interp[np.isnan(v_interp)] = -999.
                                uerr_interp[np.isnan(uerr_interp)] = -999.
                                verr_interp[np.isnan(verr_interp)] = -999.

                                foo = np.where(u_interp != -999)[0]

                                if len(foo) > 0:
                                    available[k] = 1
                                else:
                                    print('No valid windsonde data found')

                        else:
                            # print(foo_before, foo_after, uxx.shape)
                            # We have data before and after the retrieval time so interpolate the data in time
                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], uxx[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            u_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], vxx[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            v_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], uxx_sigma[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            uerr_interp = f(rtime)

                            f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], vxx_sigma[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                            verr_interp = f(rtime)

                            foo = np.where(u_interp != -999.)[0]

                            if len(foo) > 0:
                                available[k] = 1
                            else:
                                print('No valid windsonde data found')

                        foo = np.where((retz < vip['insitu_minalt'][k]) |
                                   (retz > vip['insitu_maxalt'][k]))

                        u_interp[foo] = -999.
                        v_interp[foo] = -999.
                        uerr_interp[foo] = -999.
                        verr_interp[foo] = -999.
                else:
                    print('No windsonde data for retrieval at this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None
                    z_interp = None
                    
        u.append(u_interp)
        v.append(v_interp)
        u_error.append(uerr_interp)
        v_error.append(verr_interp)
        zs.append(z_interp)

    # Build the output dictionary and return it

    insitu_data = {'success': 1, 'height': zs, 'u': u, 'v': v,
                   'u_error': u_error, 'v_error': v_error, 'valid': available}

    return insitu_data


def read_model(date, retz, rtime, vip, verbose):

    available = 0
    no_data = True

    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
             str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

    print('Reading in model data')

    files = []
    for i in range(len(dates)):
        files = files + \
            sorted(glob.glob(vip['model_path'] +
                   '/' + '*' + dates[i] + '*.nc'))

    files.sort()
    if len(files) == 0:
        if verbose >= 1:
            print('No model data files found in this directory for this date')
        u_interp = None
        v_interp = None
        uerr_interp = None
        verr_interp = None
    else:
        for i in range(len(files)):
            fid = Dataset(files[i], 'r')
            bt = fid.variables['base_time'][0]
            to = fid.variables['time_offset'][:]

            z = fid.variables['height'][:]
            ux = fid.variables['u'][:, :]
            vx = fid.variables['v'][:, :]

            ux_sigma = fid.variables['u_sigma'][:, :]
            vx_sigma = fid.variables['v_sigma'][:, :]

            fid.close()

            # Quick QC of the model data
            foo = np.where((np.abs(ux) > 100) | (np.abs(vx) > 100))

            ux[foo] = np.nan
            vx[foo] = np.nan
            ux_sigma[foo] = np.nan
            vx_sigma[foo] = np.nan

            if no_data:
                secs = bt+to
                uxx = np.copy(ux)
                vxx = np.copy(vx)
                uxx_sigma = np.copy(ux_sigma)
                vxx_sigma = np.copy(vx_sigma)
                zxx = np.array([z]*ux.shape[0])
                no_data = False
            else:
                secs = np.append(secs, bt+to)
                uxx = np.append(uxx, ux, axis=0)
                vxx = np.append(vxx, vx, axis=0)
                uxx_sigma = np.append(uxx_sigma, ux_sigma, axis=0)
                vxx_sigma = np.append(vxx_sigma, vx_sigma, axis=0)
                zxx = np.append(zxx, np.array([z]*ux.shape[0]), axis=0)

        if not no_data:
            foo = np.where(np.abs(uxx) == -999.)
            uxx[foo] = np.nan
            vxx[foo] = np.nan
            uxx_sigma[foo] = np.nan
            vxx_sigma[foo] = np.nan

            u_interp = np.ones((uxx.shape[0], len(retz)))
            v_interp = np.ones((uxx.shape[0], len(retz)))
            uerr_interp = np.ones((uxx.shape[0], len(retz)))
            verr_interp = np.ones((uxx.shape[0], len(retz)))

            for i in range(uxx.shape[0]):
                # Interpolate the data to the retrieval vertical grid
                f = interpolate.interp1d(
                    zxx[i], uxx[i], axis=0, bounds_error=False, fill_value=np.nan)
                u_interp[i] = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[i], vxx[i], axis=0, bounds_error=False, fill_value=np.nan)
                v_interp[i] = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[i], uxx_sigma[i], axis=0, bounds_error=False, fill_value=np.nan)
                uerr_interp[i] = f(np.copy(retz))

                f = interpolate.interp1d(
                    zxx[i], vxx_sigma[i], axis=0, bounds_error=False, fill_value=np.nan)
                verr_interp[i] = f(np.copy(retz))

            # Inflate the model error
            if vip['model_err_inflation'] != 1:
                print('Apply inflation factor of ' +
                      str(vip['model_err_inflation']) + ' to the model error.')
                uerr_interp *= vip['model_err_inflation']
                verr_interp *= vip['model_err_inflation']

            # Now find the profiles closest before and after the retrieval time
            foo_before = np.where(secs <= rtime)[0]
            foo_after = np.where(secs > rtime)[0]

            # No model data before retrieval time
            if len(foo_before) == 0:

                # No model data after retrieval time either. This is weird
                if len(foo_after) == 0:
                    print('Something really weird is happening in model read.')
                    print('No model data used for this time')
                    u_interp = None
                    v_interp = None
                    uerr_interp = None
                    verr_interp = None

                # We have model data after the retrieval time but not before
                else:
                    after_sec = secs[foo_after[0]]
                    if after_sec > rtime+(vip['model_timedelta']/2)*60:
                        print('No model data with in timedelta for this time.')
                    else:
                        u_interp = u_interp[foo_after[0], :]
                        v_interp = v_interp[foo_after[0], :]
                        uerr_interp = uerr_interp[foo_after[0], :]
                        verr_interp = verr_interp[foo_after[0], :]

                        u_interp[np.isnan(u_interp)] = -999.
                        v_interp[np.isnan(v_interp)] = -999.
                        uerr_interp[np.isnan(uerr_interp)] = -999.
                        verr_interp[np.isnan(verr_interp)] = -999.

                        foo = np.where(u_interp != -999)[0]

                        if len(foo) > 0:
                            available = 1
                        else:
                            print('No valid model data found')

            else:
                # We have model data before the retrieval time, but not after
                if len(foo_after) == 0:
                    before_sec = secs[foo_before[-1]]

                    if before_sec < rtime-(vip['model_timedelta']/2)*60:
                        print('No model data with in timedelta for this time.')
                    else:
                        u_interp = u_interp[foo_before[-1], :]
                        v_interp = v_interp[foo_before[-1], :]
                        uerr_interp = uerr_interp[foo_before[-1], :]
                        verr_interp = verr_interp[foo_before[-1], :]

                        u_interp[np.isnan(u_interp)] = -999.
                        v_interp[np.isnan(v_interp)] = -999.
                        uerr_interp[np.isnan(uerr_interp)] = -999.
                        verr_interp[np.isnan(verr_interp)] = -999.

                        foo = np.where(u_interp != -999)[0]

                        if len(foo) > 0:
                            available = 1
                        else:
                            print('No valid model data found')

                else:

                    # We have data before and after the retrieval time so interpolate the data in time
                    f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], u_interp[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                    u_interp = f(rtime)

                    f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], v_interp[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                    v_interp = f(rtime)

                    f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], uerr_interp[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                    uerr_interp = f(rtime)

                    f = interpolate.interp1d(secs[foo_before[-1]:foo_after[0]+1], verr_interp[foo_before[-1]:foo_after[0]+1], axis=0, bounds_error=False, fill_value=np.nan)
                    verr_interp = f(rtime)

                    foo = np.where(u_interp != -999.)[0]

                    if len(foo) > 0:
                        available = 1
                    else:
                        print('No valid model data found')
        else:
            print('No ensemble data for retrieval at this time')
            u_interp = None
            v_interp = None
            uerr_interp = None
            verr_interp = None

    # Build the output dictionary and return it

    ens_data = {'success': 1, 'height': np.copy(retz), 'u': u_interp, 'v': v_interp,
                'u_error': uerr_interp, 'v_error': verr_interp, 'valid': available}

    return ens_data


def read_windoe(date, retz, rtime, vip, verbose):

    available = 0
    no_data = True

    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
             str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]

    print('Reading in WINDoe data')

    files = []
    for i in range(len(dates)):
        files = files + \
            sorted(glob.glob(vip['windoe_path'] +
                   '/' + 'WINDoe.' + dates[i] + '*.nc'))

    if len(files) == 0:
        if verbose >= 1:
            print('No WINDoe files found in this directory for this date')
    else:
        dif = 1000000
        switch = False
        for i in range(len(files)):
            fid = Dataset(files[i], 'r')
            bt = fid.variables['base_time'][0]
            to = fid.variables['time_offset'][:]

            # We want the profile closest to the analysis time that fall into the window
            foo = np.nanargmin(np.abs((bt+to) - rtime))

            # There are no times we want here so just move on
            if np.abs((bt+to)[foo] - rtime) > (vip['windoe_timedelta']*60.):
                fid.close()
                continue

            if np.abs((bt+to)[foo] - rtime) < dif:
                dif = np.abs((bt+to)[foo] - rtime)
                switch = True

            zx = fid.variables['height'][:]
            ux = fid.variables['u_wind'][foo, :]
            vx = fid.variables['v_wind'][foo, :]

            ux_sigma = fid.variables['sigma_u'][foo, :]
            vx_sigma = fid.variables['sigma_v'][foo, :]

            fid.close()

            if no_data or switch:
                zxx = np.copy(zx)
                uxx = np.copy(ux)
                vxx = np.copy(vx)
                uxx_sigma = np.copy(ux_sigma)
                vxx_sigma = np.copy(vx_sigma)
                no_data = False
                switch = False

        if not no_data:
            foo = np.where(np.abs(uxx) == 999.)
            uxx[foo] = np.nan
            vxx[foo] = np.nan
            uxx_sigma[foo] = np.nan
            vxx_sigma[foo] = np.nan

            # Interpolate the data to the retrieval vertical grid
            f = interpolate.interp1d(
                zxx, uxx, axis=0, bounds_error=False, fill_value=-999)
            u_interp = f(np.copy(retz))

            f = interpolate.interp1d(
                zxx, vxx, axis=0, bounds_error=False, fill_value=-999)
            v_interp = f(np.copy(retz))

            f = interpolate.interp1d(
                zxx, uxx_sigma, axis=0, bounds_error=False, fill_value=-999)
            uerr_interp = f(np.copy(retz))

            f = interpolate.interp1d(
                zxx, vxx_sigma, axis=0, bounds_error=False, fill_value=-999)
            verr_interp = f(np.copy(retz))

            u_interp[np.isnan(u_interp)] = -999.
            v_interp[np.isnan(v_interp)] = -999.
            uerr_interp[np.isnan(uerr_interp)] = -999.
            verr_interp[np.isnan(verr_interp)] = -999.

            foo = np.where(u_interp != -999.)[0]
            if len(foo) > 0:
                available = 1
            else:
                print('No valid WINDoe data found')

        else:
            print('No WINDoe data for retrieval at this time')
            u_interp = None
            v_interp = None
            uerr_interp = None
            verr_interp = None

    # Build the output dictionary and return it

    windoe_data = {'success': 1, 'height': np.copy(retz), 'u': u_interp, 'v': v_interp,
                   'u_error': uerr_interp, 'v_error': verr_interp, 'valid': available}

    return windoe_data

def read_copter(date, retz, rtime, vip, verbose):
    
    available = 0
    no_data = True
    
    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'),
             str(date),  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')]
    
    print('Reading in CopterSonde data')
    
    files = []
    for i in range(len(dates)):
        files = files + \
            sorted(glob.glob(vip['copter_path'] +
                   '/' + '*.a0.' + dates[i] + '*.cdf'))
    
    if len(files) == 0:
        if verbose >= 1:
            print('No CopterSonde files found in this directory for this date')
        
        a_yaw = None
        a_pitch = None
        a_pitch_err = None
        a_roll_err = None
        a_roll = None
        a_alt = None
    
    else:
        # Need a constant for seconds from 1970 to 2010
        sec_70_10 = (datetime(2010,1,1) - datetime(1970,1,1)).total_seconds()
        dif = 1000000
        switch = False
        for i in range(len(files)):
            
            fid = Dataset(files[i], 'r')
            pos_time = fid['pos']['time'][:]
            pos_time_sec = pos_time*1e-6
            
            # There are no times we want here so just move on
            if np.abs(np.nanmean(pos_time_sec)+sec_70_10 - rtime) > (vip['copter_timedelta']*60.):
                fid.close()
                continue
            
            if np.abs(np.nanmean(pos_time_sec)+sec_70_10 - rtime) < dif:
                dif = np.abs(np.nanmean(pos_time_sec)+sec_70_10 - rtime)
                switch = True
            
            rot_time = fid['rotation']['time'][:]
            alt = fid['pos']['alt_rel_home'][:]
            
            yaw = fid['rotation']['yaw'][:]
            pitch = fid['rotation']['pitch'][:]
            roll = fid['rotation']['roll'][:]
            
            fid.close()
            
            if no_data or switch:
                xpos_time = np.copy(pos_time)
                xrot_time = np.copy(rot_time)
                xalt = np.copy(alt)
                xyaw = np.copy(yaw)
                xpitch = np.copy(pitch)
                xroll = np.copy(roll)
                no_data = False
            
        if not no_data:
            # We need to trim these variables so that they only include 
            # the ascending portions of the flights
            
            # Start with finding where the flight starts
            foo = np.where(xalt >= 10)[0]
            if len(foo) == 0:
                print('No CopterSonde data for this retrieval time')
                a_yaw = None
                a_pitch = None
                a_roll = None
                a_pitch_err = None
                a_roll_err = None
                a_alt = None
            
            else:
                
                start_time = xpos_time[foo[0]]
            
                foo = np.where(xrot_time >= start_time)
                
                if len(foo) == 0:
                    print('No CopterSonde data for this retrieval time')
                    a_yaw = None
                    a_pitch = None
                    a_roll = None
                    a_pitch_err = None
                    a_roll_err = None
                    a_alt = None
                
                else:
                    a_rot_time = xrot_time[foo]
                    a_yaw = xyaw[foo]
                    a_pitch = xpitch[foo]
                    a_roll = xroll[foo]
                    
                    # Now find where the ascending portion of the flight ends
                    foo = np.argmax(xalt)
                    end_time = xpos_time[foo]
                    
                    foo = np.where(a_rot_time < end_time)
                    a_rot_time = a_rot_time[foo]
                    a_yaw = a_yaw[foo]
                    a_pitch = a_pitch[foo]
                    a_roll = a_roll[foo]
            
                    # Now thin the data based on the user defined thinning factor
                    
                    thin = vip['copter_thin_factor']
                    a_rot_time = a_rot_time[::thin]
                    a_yaw = a_yaw[::thin]
                    a_pitch = a_pitch[::thin]
                    a_roll = a_roll[::thin]
                
                    a_pitch_err = np.ones(a_pitch.shape)*vip['copter_pitch_err']
                    a_roll_err = np.ones(a_roll.shape)*vip['copter_roll_err']
                    
                    a_alt = ((a_rot_time-a_rot_time[0]) * (vip['copter_ascent_rate']*1e-6) + 10)/1000.
            
                if len(a_yaw) > 1:
                    available = 1
            
        else:
            print('No CopterSonde data for this retrieval time')
            a_yaw = None
            a_pitch = None
            a_roll = None
            a_pitch_err = None
            a_roll_err = None
            a_alt = None
        
        
    copter_data = {'success': 1, 'height': a_alt, 'pitch': a_pitch, 'yaw': a_yaw, 'roll':a_roll,
                   'pitch_error': a_pitch_err, 'roll_error': a_roll_err, 'valid': available}       
    
    return copter_data
    

def read_all_data(date, retz, rtime, vip, verbose):

    fail = 0

    # Get the raw lidar data. Nothing to fancy is happening and there is no
    # averaging of the data with time

    if vip['raw_lidar_number'] > 0:

        raw_lidar_data = read_raw_lidar(date, retz, rtime, vip, verbose)
        fail = 1

    else:
        raw_lidar_data = {'success': -999.}

    # Get the processed lidar data (e.g VADs). This data is interpolated to the
    # the heights retrival heights but there is no time interpolation

    if vip['proc_lidar_number'] > 0:

        proc_lidar_data = read_proc_lidar(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        proc_lidar_data = {'success': -999.}

    if vip['cons_profiler_type'] > 0:

        prof_cons_data = read_prof_cons(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        prof_cons_data = {'success': -999.}

    if vip['raw_profiler_number'] > 0:
        raw_prof_data = read_raw_prof(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        raw_prof_data = {'success': -999.}

    if vip['insitu_number'] > 0:
        insitu_data = read_insitu(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        insitu_data = {'success': -999.}

    if vip['use_model'] == 1:
        mod_data = read_model(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        mod_data = {'success': -999}

    if vip['use_windoe'] == 1:
        windoe_data = read_windoe(date, retz, rtime, vip, verbose)
        fail = 1
    else:
        windoe_data = {'success': -999}
    
    if vip['use_copter'] == 1:
        copter_data = read_copter(date,retz,rtime,vip,verbose)
        fail = 1
    else:
        copter_data = {'success': -999}

    return fail, raw_lidar_data, proc_lidar_data, prof_cons_data, raw_prof_data, insitu_data, mod_data, copter_data, windoe_data
