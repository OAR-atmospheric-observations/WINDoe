# -----------------------------------------------------------------------------
#
# Copyright (C) 2023 by Joshua Gebauer
# All Rights Reserved
#
# This file is part of the "WINDoe" retrieval system.
#
# WINDoe is free software developed while the authors were at the 
# Cooperative Institute for Severe and High-Impact Weather Research and Operations
# at the University of Oklahoma and funded by NOAA/Office of Oceanic and Atmospheric
# Research under NOAA-University of Oklahoma Cooperative Agreement #NA21OAR4320204,
# U.S. Department of Commerce. It is intended to be free software.
# It is made available WITHOUT ANY WARRANTY, For more information, contact the authors.
#
# -----------------------------------------------------------------------------



__version__ = '0.0.1'

import os
import sys
import numpy as np
import shutil
import scipy.io
import scipy.linalg
import copy
import warnings
from netCDF4 import Dataset
from datetime import datetime
from time import gmtime, strftime
from argparse import ArgumentParser

import VIP_Databases_functions
import Output_Functions
import Data_reads
import Jacobian_Functions
import Other_functions

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name if the VIP file (string)")
parser.add_argument("prior_filename", help="Name of the prior input dataset (string)")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last AERI sample]")
parser.add_argument("--verbose",type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")


args = parser.parse_args()

date = args.date
vip_filename = args.vip_filename
prior_filename = args.prior_filename
shour = args.shour
ehour = args.ehour
verbose = args.verbose
debug = args.debug

#Check to see if any of these are set; if not, fall back to default values

if shour is None:
    shour = 0.
if ehour is None:
    ehour = -1.
if verbose is None:
    verbose = 1
if debug is None:
    debug = False

# Initialize
success = False

#Capture the version of this file
globatt = {'algorithm_code': 'WINDoe Retrieval Code',
           'algorithm_author': 'Joshua Gebauer, CIWRO / NOAA National Severe Storms Laboratory (joshua.gebauer@noaa.gov)',
           'algorithm_comment1': 'WINDoe is optimal estimation wind retreival used to obtain wind profiles from ' +
                                 'from remotely sensed wind observations. The algorithm is similar to TROPoe which was  ' +
                                 'created to retrieve thermodynamic wind profiles.',
           'algorithm_isclaimer': 'WINDoe was developed at NOAA and is provided on an as-is basis, with no warranty',
           'algorithm_version': __version__,
           'algorthm_reference': 'Gebauer, JG and TM Bell, 2023: A Flexible, ' +
                    'Multi-Instrument Optimal Estimation Retrieval for Wind Profiles.' +
                    'J. Atmos. Oceanic Tech., Submitted.',
           'datafile_created_on_date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
           'datafile_created_on_machine': os.uname()[-1]}
           
            
                    
# Start the retrieval
print(' ')
print('-------------------------------------------------------------------------')
print('------- WINDoe is a wind retrieval algorithm developed at NOAA ----------')
print('---------------- Contact is joshua.gebauer (@noaa.gov) ------------------')
print('------- The code is provided on an "as-is" basis, with no warranty ------')
print('------------------------------------------------------------------------ ')
print(' ')
print('>>> Starting WINDoe retrieval for ' + str(date) + ' (from ' + str(shour) + ' to ' + str(ehour) + ' UTC) <<<')

#Find the VIP file and read it
vip = VIP_Databases_functions.read_vip_file(vip_filename, globatt = globatt, verbose = verbose)

if vip['success'] != 1:
   VIP_Databases_functions.abort(date)
   sys.exit()

if debug:
    print('Saving the VIP and glabatt structure into "vip.npy" -- for debugging')
    np.save('vip.npy', vip)

# Check if the prior data exists
if not os.path.exists(prior_filename):
    print(('Error: Unable to find the prior data file: ' + prior_filename))
    VIP_Databases_functions.abort(date)
    sys.exit()
    
# House keeping stuff
starttime = datetime.now()
endtime = starttime
print(' ')

if VIP_Databases_functions.check_vip(vip) == 1:
    VIP_Databases_functions.abort(date)
    sys.exit()
 
cvgmult = 0.1            # I will leave this here for now, but later will add this to the vip file. It is a multiplier to apply to the convergence test (0 - 1.0)

# Read in the a priori covariance matrix of u and v for this study
nsonde_prior = -1
try:
    print(('Read prior file: ' + prior_filename))
    fid = Dataset(prior_filename,'r')
except:
    print('Error: Unable to open the XaSa file')
    VIP_Databases_functions.abort(date)
    sys.exit()

z = fid.variables['height'][:]
Xa = np.array(fid.variables['mean_prior'][:])
Sa = np.array(fid.variables['covariance_prior'][:])

# Add in the prior information for w
Xa = np.append(Xa,np.zeros(len(z)))
temp_zeros = np.zeros((len(z)*2,len(z)))
temp_wcov = Other_functions.get_w_covariance(z.data, vip['w_mean'], vip['w_lengthscale'])
temp_Sa = np.append(Sa,temp_zeros.T,axis = 0)
Sa = np.append(temp_Sa,np.append(temp_zeros,temp_wcov,axis = 0),axis=1)

sig_Xa = np.sqrt(np.diag(Sa))


z2 = np.append(z,z)

try:
    nsonde_prior = int(fid.Nsonde.split()[0])
except AttributeError:
    nsonde_prior = int(fid.Nprofiles.split()[0])

comment_prior = str(fid.Comment)
minU = float(fid.QC_limits_U.split()[5])
maxU = float(fid.QC_limits_U.split()[7])
if verbose == 3:
    print('QC limits for U are ' + str(minU) + ' and ' + str(maxU))
minV = float(fid.QC_limits_V.split()[5])
maxV = float(fid.QC_limits_V.split()[7])

if verbose == 3:
    print('QC limits for V are ' + str(minV) + ' and ' + str(maxV))
fid.close()


if verbose >= 1:
    print('Retrived profiles will have ' + str(len(z)) + ' levels (from prior)')
if verbose >= 2:
    print('There were ' + str(nsonde_prior) + ' radiosondes used in the calculation of the prior')

# TODO -- Add inflate prior covariance function here

if ehour < 0:
    ehour = 24
    if verbose >= 2:
        print(('Resetting the processing end hour to ' + str(ehour) + ' UTC'))

# Put all of the prior information into a structure for later
prior = {'comment':comment_prior, 'filename':prior_filename, 'nsonde':nsonde_prior,
         'Xa':np.copy(Xa), 'Sa':np.copy(Sa)}

# Set up array of retrieval times
rtime = ((datetime.strptime(str(date), '%Y%m%d') - datetime(1970,1,1)).total_seconds() +
        np.arange((shour*3600),(ehour*3600)+1,(vip['tres']*60)))

rtime_hour = np.arange((shour*3600),(ehour*3600)+1,(vip['tres']*60))/3600

# Compute inverse of the prior
SaInv = scipy.linalg.pinv(Sa)

# Now loop over the observations and perform the retrievals
xret = []                  #Initialize. Will overwrite this if a succesful ret
already_saved = 0          # Flag to say if saved already or not...
fsample = 0                # Counter for number of spectra processed


# If clobber == 2, then we will try to append. But this requires that
# I perform a check to make sure that we are appending to a file that was
# created by a version of the code that makes sense. I only need to make this
# test once, hence this flag
if vip['output_clobber'] == 2:
    check_clobber = 1
else:
    check_clobber = 0

version = ''
noutfilename = ''

###############################################################################
# This is the main loop for the retrieval! Note this is slightly different than
# TROPoe since the input data can be much much much larger (GBs) so only the data
# needed for each retrieval time is read in each iteration
###############################################################################

for i in range(len(rtime)):
    
    # If we are to append to the file, then I need to find the last valid
    # sample in the file, so I only process after that point...
    if ((vip['output_clobber'] == 2) and (check_clobber == 1)):
        fsample, last_sec, noutfilename = Output_Functions.find_last_time(date,vip,z, shour)
        check_clobber = 0
        
        if fsample < 0:
            VIP_Databases_functions.abort(date)
            sys.exit()
        
        elif fsample == 0:
            vip['output_clobber'] = 0
        
        if ((verbose >= 1) and (fsample > 0)):
            print('Will append output to the file ' + noutfilename)
        
    # If we are in 'append' mode, then skip any samples that are before the 
    # last time in the output file.
    
    if vip['output_clobber'] == 2:
        if rtime[i] <= last_sec:
            print(' ....but was already processed (append_mode)')
            continue
        
    # Read in the data
    fail, raw_lidar, proc_lidar, prof_cons, prof_raw, insitu, model, copter, windoe = Data_reads.read_all_data(date, z, rtime[i], vip, verbose)
    
    # Set iterate once to false here
    iterate_once = False
    
    if fail == 0:
        print('There was no data selected to be used as input. Aborting.')
        VIP_Databases_functions.abort(date)
        sys.exit()
    
    # Make sure we have valid data for performing the retrieval
    if raw_lidar['success'] > 0:
        raw_foo = np.where(raw_lidar['valid'] == 1)[0]
    else:
        raw_foo = np.array([])
    
    if proc_lidar['success'] > 0:
        proc_foo = np.where(proc_lidar['valid'] == 1)[0]
    else:
        proc_foo = np.array([])
    
    if prof_cons['success'] < 0:
        prof_cons['valid'] = 0
    
    if prof_raw['success'] > 0:
        prof_foo = np.where(prof_raw['valid'] == 1)[0]
    else:
        prof_foo = np.array([])
    
    if insitu['success'] > 0:
        insitu_foo = np.where(insitu['valid'] == 1)[0]
    else:
        insitu_foo = np.array([])
        
    if model['success'] < 0:
        model['valid'] = 0
    
    if copter['success'] < 0:
        copter['valid'] = 0
    
    if windoe['success'] < 0:
        windoe['valid'] = 0
    
    # There needs to be some type of observational data
    if ((len(raw_foo) == 0) & (len(proc_foo) == 0) & (len(insitu_foo) == 0) & (prof_cons['valid'] == 0) &
        (len(prof_foo) == 0) & (copter['valid'] == 0)):
        print('Sample ' + str(i) + ' at ' + str(rtime_hour[i]) + ' UTC -- no valid data found.')
        continue
    else:
        print('Sample ' + str(i) + ' at ' + str(rtime_hour[i]) + ' UTC is being processed')
    
    # Set up the observation vector for the retrieval and its covariance matrix
    # The covariance matrix is diagonal. This is important since no matrix inversion
    # will have to be done. If off-diagonal elements are eventually used the entire
    # retrieval structure will have to change.
    
    # A flag is needed for all of the observations for the call out to the forward models
    # The values are:
    #          1 -- Raw lidar data
    #          2 -- Processed lidar data u-wind
    #          3 -- Processed lidar data v-wind
    #          4 -- Consensus profiler data u-wind
    #          5 -- Consensus profiler data v-wind
    #          6 -- Insitu u-wind
    #          7 -- Insitu v-wind
    #          8 -- Model u-wind
    #          9 -- Model v-wind
    #         10 -- WINDoe u-wind
    #         11 -- WINDoe v-wind
    #         12 -- Raw profiler radial velocity
    #         13 -- CopterSonde pitch
    #         14 -- CopterSonde roll
    
    # Also need to keep track of dimensions (elevation, azimuths, height, range, etc.)
    # For raw lidar data height will be range to keep the number of arrays I need to a minimum
    # Processed lidar data will get misssing values for elevatin and azimuth
    
    Y = []
    Sy = []
    sigY = []
    dimY = []
    flagY = []
    azY = []
    elY = []
    if raw_lidar['success'] > 0:
        for j in range(vip['raw_lidar_number']):
            if raw_lidar['valid'][j] == 1:
            
                foo = np.where((raw_lidar['vr'][j] >= -500) & (raw_lidar['vr_var'][j] >= -500))
                if len(foo[0]) == 0:
                    print('Major error when adding raw_lidar data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
                
                Y.extend(raw_lidar['vr'][j][foo].ravel())
                sigY.extend(raw_lidar['vr_var'][j][foo].ravel())
                Sy.extend(raw_lidar['vr_var'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(raw_lidar['vr'][j][foo].ravel())))
                dimY.extend(raw_lidar['z'][foo[0]].ravel())
                azY.extend((raw_lidar['azimuth'][j][foo[1]]).ravel())
                elY.extend((raw_lidar['elevation'][j][foo[1]]).ravel())
                
    if proc_lidar['success'] > 0:
        for j in range(vip['proc_lidar_number']):
            if proc_lidar['valid'][j] == 1:
                
                foo = np.where((proc_lidar['u'][j] >= -500) & (proc_lidar['u_error'][j] >= -500) &
                               (proc_lidar['v'][j] >= -500) & (proc_lidar['v_error'][j] >= -500))
            
                if len(foo[0]) == 0:
                    print('Major error when adding proc_lidar data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
                
                Y.extend(proc_lidar['u'][j][foo].ravel())
                sigY.extend(proc_lidar['u_error'][j][foo].ravel())
                Sy.extend(proc_lidar['u_error'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(proc_lidar['u'][j][foo].ravel()))*2)
                dimY.extend(proc_lidar['height'][foo[0]])
                azY.extend(np.ones(len(proc_lidar['u'][j][foo])).ravel()*-999)
                elY.extend(np.ones(len(proc_lidar['u'][j][foo])).ravel()*-999)
                
        for j in range(vip['proc_lidar_number']):
            if proc_lidar['valid'][j] == 1:
            
                foo = np.where((proc_lidar['u'][j] >= -500) & (proc_lidar['u_error'][j] >= -500) &
                               (proc_lidar['v'][j] >= -500) & (proc_lidar['v_error'][j] >= -500))
            
                if len(foo[0]) == 0:
                    print('Major error when adding proc_lidar data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
                    
                Y.extend(proc_lidar['v'][j][foo].ravel())
                sigY.extend(proc_lidar['v_error'][j][foo].ravel())
                Sy.extend(proc_lidar['v_error'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(proc_lidar['v'][j][foo].ravel()))*3)
                dimY.extend(proc_lidar['height'][foo[0]])
                azY.extend(np.ones(len(proc_lidar['v'][j][foo])).ravel()*-999)
                elY.extend(np.ones(len(proc_lidar['v'][j][foo])).ravel()*-999)
    
    if prof_cons['success'] > 0:
        if prof_cons['valid'] == 1:
            foo = np.where((prof_cons['u'] >= -500) & (prof_cons['u_error'] >= -500) &
                           (prof_cons['v'] >= -500) & (prof_cons['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding consensus wind profiler data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(prof_cons['u'][foo].ravel())
            sigY.extend(prof_cons['u_error'][foo].ravel())
            Sy.extend(prof_cons['u_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(prof_cons['u'][foo].ravel()))*4)
            dimY.extend(prof_cons['height'][foo[0]])
            azY.extend(np.ones(len(prof_cons['u'][foo])).ravel()*-999)
            elY.extend(np.ones(len(prof_cons['u'][foo])).ravel()*-999)
            
        if prof_cons['valid'] == 1:
            foo = np.where((prof_cons['u'] >= -500) & (prof_cons['u_error'] >= -500) &
                           (prof_cons['v'] >= -500) & (prof_cons['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding consensus wind profiler data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(prof_cons['v'][foo].ravel())
            sigY.extend(prof_cons['v_error'][foo].ravel())
            Sy.extend(prof_cons['v_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(prof_cons['v'][foo].ravel()))*5)
            dimY.extend(prof_cons['height'][foo[0]])
            azY.extend(np.ones(len(prof_cons['v'][foo])).ravel()*-999)
            elY.extend(np.ones(len(prof_cons['v'][foo])).ravel()*-999)
    
    if insitu['success'] > 0:
        for j in range(vip['insitu_number']):
            if insitu['valid'][j] == 1:
                
                foo = np.where((insitu['u'][j] >= -500) & (insitu['u_error'][j] >= -500) &
                               (insitu['v'][j] >= -500) & (insitu['v_error'][j] >= -500))

                if len(foo[0]) == 0:
                    print('Major error when adding insitu data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
                
                Y.extend(insitu['u'][j][foo].ravel())
                sigY.extend(insitu['u_error'][j][foo].ravel())
                Sy.extend(insitu['u_error'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(insitu['u'][j][foo].ravel()))*6)
                dimY.extend(insitu['height'][j][foo[0]])
                azY.extend(np.ones(len(insitu['u'][j][foo])).ravel()*-999)
                elY.extend(np.ones(len(insitu['u'][j][foo])).ravel()*-999)
                
        for j in range(vip['insitu_number']):
            if insitu['valid'][j] == 1:
            
                foo = np.where((insitu['u'][j] >= -500) & (insitu['u_error'][j] >= -500) &
                               (insitu['v'][j] >= -500) & (insitu['v_error'][j] >= -500))
            
                if len(foo[0]) == 0:
                    print('Major error when adding proc_lidar data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
                    
                Y.extend(insitu['v'][j][foo].ravel())
                sigY.extend(insitu['v_error'][j][foo].ravel())
                Sy.extend(insitu['v_error'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(insitu['v'][j][foo].ravel()))*7)
                dimY.extend(insitu['height'][j][foo[0]])
                azY.extend(np.ones(len(insitu['v'][j][foo])).ravel()*-999)
                elY.extend(np.ones(len(insitu['v'][j][foo])).ravel()*-999)

    if model['success'] > 0:
        if model['valid'] == 1:
            foo = np.where((model['u'] >= -500) & (model['u_error'] >= -500) &
                           (model['v'] >= -500) & (model['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding ensemble data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(model['u'][foo].ravel())
            sigY.extend(model['u_error'][foo].ravel())
            Sy.extend(model['u_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(model['u'][foo].ravel()))*8)
            dimY.extend(model['height'][foo[0]])
            azY.extend(np.ones(len(model['u'][foo])).ravel()*-999)
            elY.extend(np.ones(len(model['u'][foo])).ravel()*-999)
            
        if model['valid'] == 1:
            foo = np.where((model['u'] >= -500) & (model['u_error'] >= -500) &
                           (model['v'] >= -500) & (model['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding consensus wind profiler data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(model['v'][foo].ravel())
            sigY.extend(np.sqrt(model['v_error'][foo].ravel()))
            Sy.extend(model['v_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(model['v'][foo].ravel()))*9)
            dimY.extend(model['height'][foo[0]])
            azY.extend(np.ones(len(model['v'][foo])).ravel()*-999)
            elY.extend(np.ones(len(model['v'][foo])).ravel()*-999)
    
    if windoe['success'] > 0:
        if windoe['valid'] == 1:
            foo = np.where((windoe['u'] >= -500) & (windoe['u_error'] >= -500) &
                           (windoe['v'] >= -500) & (windoe['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding WINDoe data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(windoe['u'][foo].ravel())
            sigY.extend(windoe['u_error'][foo].ravel())
            Sy.extend(windoe['u_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(windoe['u'][foo].ravel()))*10)
            dimY.extend(windoe['height'][foo[0]])
            azY.extend(np.ones(len(windoe['u'][foo])).ravel()*-999)
            elY.extend(np.ones(len(windoe['u'][foo])).ravel()*-999)
            
        if windoe['valid'] == 1:
            foo = np.where((windoe['u'] >= -500) & (windoe['u_error'] >= -500) &
                           (windoe['v'] >= -500) & (windoe['v_error'] >= -500))
        
            if len(foo[0]) == 0:
                print('Major error when adding WINDoe data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
        
            Y.extend(windoe['v'][foo].ravel())
            sigY.extend(windoe['v_error'][foo].ravel())
            Sy.extend(windoe['v_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(windoe['v'][foo].ravel()))*11)
            dimY.extend(windoe['height'][foo[0]])
            azY.extend(np.ones(len(windoe['v'][foo])).ravel()*-999)
            elY.extend(np.ones(len(windoe['v'][foo])).ravel()*-999)
    
    if prof_raw['success'] > 0:
        for j in range(vip['raw_profiler_number']):
            if prof_raw['valid'][j] == 1:
                
                foo = np.where((prof_raw['vr'][j] >= -500) & (prof_raw['vr_error'][j] >= -500))
        
                if len(foo[0]) == 0:
                    print('Major error when adding raw profiler data to observation vector. This should not happen!')
                    VIP_Databases_functions.abort(date)
                    sys.exit()
        
                Y.extend(prof_raw['vr'][j][foo].ravel())
                sigY.extend(prof_raw['vr_error'][j][foo].ravel())
                Sy.extend(prof_raw['vr_error'][j][foo].ravel()**2)
                flagY.extend(np.ones(len(prof_raw['vr'][j][foo].ravel()))*12)
                dimY.extend(prof_raw['height'][foo[0]])
                azY.extend(prof_raw['az'][j][foo[1]].ravel())
                elY.extend(np.ones(len(prof_raw['vr'][j][foo].ravel()))*(90-prof_raw['el'][j]))
    
    if copter['success'] > 0:
        if copter['valid'] == 1:
            foo = np.where((copter['pitch'] >= -500) & (copter['pitch_error'] >= -500) &
                           (copter['yaw'] >= -500))
            
            if len(foo[0]) == 0:
                print('Major error when adding copter data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            Y.extend(copter['pitch'][foo].ravel())
            sigY.extend(copter['pitch_error'][foo].ravel())
            Sy.extend(copter['pitch_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(copter['pitch'][foo].ravel()))*13)
            dimY.extend(copter['height'][foo[0]])
            azY.extend(copter['yaw'][foo].ravel())
            elY.extend(np.ones(len(copter['pitch'][foo])).ravel()*-999)
            
        if copter['valid'] == 1:
            foo = np.where((copter['roll'] >= -500) & (copter['roll_error'] >= -500) &
                           (copter['yaw'] >= -500))
            
            if len(foo[0]) == 0:
                print('Major error when adding copter data to observation vector. This should not happen!')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            Y.extend(copter['roll'][foo].ravel())
            sigY.extend(copter['roll_error'][foo].ravel())
            Sy.extend(copter['roll_error'][foo].ravel()**2)
            flagY.extend(np.ones(len(copter['roll'][foo].ravel()))*14)
            dimY.extend(copter['height'][foo[0]])
            azY.extend(copter['yaw'][foo].ravel())
            elY.extend(np.ones(len(copter['roll'][foo])).ravel()*-999)
        
    Y = np.array(Y)
    sigY = np.array(sigY)
    flagY = np.array(flagY)
    dimY = np.array(dimY)
    azY = np.array(azY)
    elY = np.array(elY)

    zmin = np.nanmin(dimY)
    zmax = np.nanmax(dimY)
    
    
    # Check to make sure there are observations that are reasonable to do a 
    # retrieval if not move on
    
    foo = np.where(sigY < 50)[0]
    if len(foo) == 0:
        print('No quality observations available for the retrieval. Skipping sample.')
        continue
                   
    Sf = np.zeros((len(Sy),len(Sy)))                       # This is all zeros for now
   
    nY = len(Y)
    nX = len(z)*3             # For U, V, and W
    
    # Start building the first guess vector
    X0 = np.copy(Xa)          # Start with the prior, and overwrite portions of it if desired
    first_guess = 'prior'
    if vip['first_guess'] == 1:
        # Use the prior as the first guess
        if verbose >= 3:
            print('Using prior as first guess')
        
        elif vip['first_guess'] == 2:
            # Get first guess from the previous retrieval. if there is one
            # if there isn't a valid prior retrieval, use prior
            if verbose >= 3:
                print('Using previous good retrieval as first guess')
            print('First guess option 2 is not available yet.')
            VIP_Databases_functions.abort(date)
            sys.exit()
            
            first_guess = 'lastSample'
            # TODO -- Need to build this better once I know output format since
            # I don't want to be keeping previous retrievals in memory forever
            # if it isn't needed.
    
    # Build the first guess vector
    itern = 0
    converged = 0
    Xn = np.copy(X0)
    Fxnm1 = np.array([-999.])

    # Define the gamma factors needed to keep the retrieval sane, we are
    # keeping these at ones for now
    if first_guess == 'lastSample':
        gfactor = np.array([1,1,1,1])
    else:
        #gfactor = np.array([10000.,5000.,1000.,500.,100.,50.,10.,5.,1.])
        gfactor = np.array([1.,1.,1.,1.,1.,1.,1.])
    if len(gfactor) < vip['max_iterations']:
        gfactor = np.append(gfactor, np.ones(vip['max_iterations']-len(gfactor)+3))
    
    continue_next_sample = 0
    while ((itern <= vip['max_iterations']) and (converged == 0)):       # While loop over iter
        if verbose >= 3:
            print(' Making the foward calculation for iteration ' + str(itern))
        
        first_jacobian = True
        first_copter = True
        
        # This code makes the forward calculation for raw lidar data
        foo = np.where(flagY == 1)[0]
        if len(foo) > 0:
            flag, KK, FF= Jacobian_Functions.compute_jacobian_vr(Xn,z, Y[foo], dimY[foo],azY[foo],elY[foo],itern)
        
            if flag == 0:
                print('Problem computing Jacobian for raw lidar data. Have to abort.')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
            bar = np.where(FF < -900)[0]
            if len(bar) > 0:
                Y[foo[bar]] = -999.
                KK[bar,:] = 0.
            
            if first_jacobian:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                first_jacobian = False
            else:
                Kij = np.append(Kij,KK,axis=0)
                FXn = np.append(FXn,FF)
        
        # This code makes the forward calculation for u-component of processed lidar data
        foo = np.where(flagY == 2)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],0)
       
           if flag == 0:
               print('Problem computing Jacobian for processed lidar data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
               
           
        # This code makes the forward calculation for v-component of processed lidar data
        foo = np.where(flagY == 3)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],1)
       
           if flag == 0:
               print('Problem computing Jacobian for processed lidar data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
        
        # This code makes the forward calculation for u-component of consensus profiler data
        foo = np.where(flagY == 4)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],0)
       
           if flag == 0:
               print('Problem computing Jacobian for consensus profiler data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
               
           
        # This code makes the forward calculation for v-component of consensus profiler data
        foo = np.where(flagY == 5)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],1)
       
           if flag == 0:
               print('Problem computing Jacobian for consensus profiler data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
        
        # This code makes the forward calculation for u-component of insitu data
        foo = np.where(flagY == 6)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],0)
       
           if flag == 0:
               print('Problem computing Jacobian for insitu data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
               
           
        # This code makes the forward calculation for v-component of insitu data
        foo = np.where(flagY == 7)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],1)
       
           if flag == 0:
               print('Problem computing Jacobian for insitu data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
        
                
        # This code makes the forward calculation for u-component of model data
        foo = np.where(flagY == 8)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],0)
       
           if flag == 0:
               print('Problem computing Jacobian for model data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
               
           
        # This code makes the forward calculation for v-component of model data
        foo = np.where(flagY == 9)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],1)
       
           if flag == 0:
               print('Problem computing Jacobian for consensus profiler data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
        
        
        # This code makes the forward calculation for u-component of WINDoe data
        foo = np.where(flagY == 10)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],0)
       
           if flag == 0:
               print('Problem computing Jacobian for the WINDoe data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
               
           
        # This code makes the forward calculation for v-component of consensus profiler data
        foo = np.where(flagY == 11)[0]
        if len(foo) > 0:
           flag, KK, FF = Jacobian_Functions.compute_jacobian_uv(Xn,z,dimY[foo],1)
       
           if flag == 0:
               print('Problem computing Jacobian for WINDoe data. Have to abort.')
               VIP_Databases_functions.abort(date)
               sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
           bar = np.where(FF < -900)[0]
           if len(bar) > 0:
               Y[foo[bar]] = -999.
               KK[bar,:] = 0.
            
           if first_jacobian:
               Kij = np.copy(KK)
               FXn = np.copy(FF)
               first_jacobian = False
           else:
               Kij = np.append(Kij,KK,axis=0)
               FXn = np.append(FXn,FF)
        
        # This code makes the forward calculation for raw radar wind profiler
        foo = np.where(flagY == 12)[0]
        if len(foo) > 0:
            flag, KK, FF= Jacobian_Functions.compute_jacobian_vr(Xn,z, Y[foo], dimY[foo],azY[foo],elY[foo],itern)
        
            if flag == 0:
                print('Problem computing Jacobian for raw profiler data. Have to abort.')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside
            # of the prior height bounds? If so we want the observation to have the same value
            # and have no sensitivity in the Jacobian so that the retrieval is unaffected
            
            bar = np.where(FF < -900)[0]
            if len(bar) > 0:
                Y[foo[bar]] = -999.
                KK[bar,:] = 0.
            
            if first_jacobian:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                first_jacobian = False
            else:
                Kij = np.append(Kij,KK,axis=0)
                FXn = np.append(FXn,FF)
        
        # This code makes the forward calculation for CopterSonde pitch data
        foo = np.where(flagY == 13)[0]
        if len(foo) > 0:
            flag, KK, FF, Kbb = Jacobian_Functions.compute_jacobian_copter_pitch(Xn, z, dimY[foo],azY[foo], vip['copter_constants'][0],vip['copter_constants'][1])
            
            if flag == 0:
                print('Problem computing Jacobian for copter pitch data. Have to abort.')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside of the
            # prior height bounds? If so we want the observation to have the same value and have no
            # sensitivity in the Jacobian so that the retreival is unaffected
            
            bar = np.where(FF < -900)[0]
            if len(bar) > 0:
                Y[foo[bar]] = -999.
                KK[bar,:] = 0
                Kbb[bar,:] = 0
            
            if first_jacobian:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                first_jacobian = False
            else:
                Kij = np.append(Kij, KK, axis = 0)
                FXn = np.append(FXn, FF)
                
            if first_copter:
                Kb = np.copy(Kbb)
                first_copter = False
            else:
                Kb = np.append(Kb,Kbb,axis = 0)
                
            
        foo = np.where(flagY == 14)[0]
        if len(foo) > 0:
            flag, KK, FF, Kbb = Jacobian_Functions.compute_jacobian_copter_roll(Xn, z, dimY[foo],azY[foo], vip['copter_constants'][0],vip['copter_constants'][1])
            
            if flag == 0:
                print('Problem computing Jacobian for copter yaw data. Have to abort.')
                VIP_Databases_functions.abort(date)
                sys.exit()
            
            # Are there missing values in the forward operator due to interpolating outside of the
            # prior height bounds? If so we want the observation to have the same value and have no
            # sensitivity in the Jacobian so that the retreival is unaffected
            
            bar = np.where(FF < -900)[0]
            if len(bar) > 0:
                Y[foo[bar]] = -999.
                KK[bar,:] = 0
                Kbb[bar] = 0
            
            if first_jacobian:
                Kij = np.copy(KK)
                FXn = np.copy(FF)
                first_jacobian = False
            else:
                Kij = np.append(Kij, KK, axis = 0)
                FXn = np.append(FXn, FF)
            
            if first_copter:
                Kb = np.copy(Kbb)
                first_copter = False
            else:
                Kb = np.append(Kb,Kbb,axis = 0)
             
        ########
        # Done computing forward calculations and Jacobians. Now the retrieval math.
        ########
        # First we need to calculate the forward model error for the copters
        foo = np.where((flagY == 13) | (flagY == 14))[0]
        if ((len(foo) > 0)):
            Sb = np.diag(np.array([vip['copter_constants_unc'][0]**2, vip['copter_constants_unc'][1]**2, vip['copter_fit_unc']**2]))
            Sff = Kb.dot(Sb).dot(Kb.T)
            Sf[foo[0]:foo[-1]+1,foo[0]:foo[-1]+1] = Sff
        else:
            if vip['run_fast'] == 1:
                iterate_once = True
        
        # compute an obs_hgt_max for each of the data types provided in the run 
        dtypes = [[1], [2,3], [4,5], [6,7], [8,9], [10,11], [12], [13,14]]
        
        # create a max hight dir to store values
        max_hgts = {'raw_lidar': -999, 'proc_lidar': -999, 'con_prof': -999, 'insitu': -999, 'model': -999, 'windoe': -999, 'raw_prof': -999}
        
        # overwrite to nans for datatypes that are supposed to be used in the overall retrieval 
        if vip['raw_lidar_number'] != 0:
            max_hgts['raw_lidar'] = np.nan
        if vip['proc_lidar_number'] != 0:
            max_hgts['proc_lidar'] = np.nan
        if vip['cons_profiler_type'] != 0:
            max_hgts['con_prof'] = np.nan
        if vip['insitu_number'] != 0:
            max_hgts['insitu'] = np.nan
        if vip['use_model'] != 0:
            max_hgts['model'] == np.nan
        if vip['use_windoe'] != 0:
            max_hgts['windoe'] = np.nan
        if vip['raw_profiler_number'] != 0:
            max_hgts['raw_prof'] = np.nan
            
        # loop through the different obs types
        for t in range(len(dtypes)):
            # find where the data of this type exists
            foo = np.where(np.in1d(flagY, dtypes[t]))[0]

            # if there is no data, move forward
            if len(foo) == 0:
                continue
            
            # subselect our uncertainties and heights
            sigy_foo = sigY[foo]
            dimY_foo = dimY[foo]
            
            # determine if we are using insitu obs
            if dtypes[t][0] == 6:
                # whatever the highest height is containing insitu obs is our max height
                max_hgts['insitu'] = np.nanmax(dimY_foo)
            
            if dtypes[t][0] == 1:
                max_hgts['raw_lidar'] = np.nanmax(dimY_foo)
                
            # if we are not working with raw lidar or isitu obs
            else:
                # loop through each height we have data for 
                for zz in np.unique(dimY_foo):
                    # find where we have data at that height where the uncertainty is within our bounds 
                    fah = np.where((dimY_foo == zz) & (sigy_foo < 9))[0]
                    
                    # if we have at least one good data point, this is our max height 
                    if len(fah) > 0:
                        if dtypes[t][0] == 2:
                            max_hgts['proc_lidar'] = zz
                        if dtypes[t][0] == 4:
                            max_hgts['con_prof'] = zz
                        if dtypes[t][0] == 8:
                            max_hgts['model'] = zz
                        if dtypes[t][0] == 10:
                            max_hgts['windoe'] = zz
                        if dtypes[t][0] == 12:
                            max_hgts['con_prof'] = zz           
                 
        # Set an error floor of 1 for all observations except Copter data to 
        # prevent overfitting
        # TODO: Make noise floor part of namelist
        Sy = np.array(Sy)
        foo = np.where((flagY <= 12) & (Sy<1))
        Sy[foo] = 1
        sigY[foo] = 1
        
        # We need to check if the user wants Sm to be diagonal only. If so it
        # changes how we do the matrix math
        if vip['diagonal_covariance'] == 1:
            Sm = Sy + np.diag(Sf)
        else:
            Sm = np.diag(Sy) + Sf
            
        gfac = gfactor[itern]
        
        if vip['diagonal_covariance'] == 1:
            SmInv = 1./Sm
            B = (gfac * SaInv) + (Kij.T*SmInv[None,:]).dot(Kij)
            Binv = scipy.linalg.pinv(B)
            Gain = Binv.dot(Kij.T) * SmInv[None,:]
            Xnp1 = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
            Sop = Binv.dot(gfac*gfac*SaInv + (Kij.T*SmInv[None,:]).dot(Kij)).dot(Binv)
            SopInv = scipy.linalg.inv(Sop)
            Akern = (Binv.dot(Kij.T)*SmInv[None,:]).dot(Kij)
        else:
            SmInv = np.linalg.inv(Sm)
            B = (gfac * SaInv) + Kij.T.dot(SmInv).dot(Kij)
            Binv = scipy.linalg.pinv(B)
            Gain = Binv.dot(Kij.T).dot(SmInv)
            Xnp1 = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
            Sop = Binv.dot(gfac*gfac*SaInv + Kij.T.dot(SmInv).dot(Kij)).dot(Binv)
            SopInv = scipy.linalg.inv(Sop)
            Akern = Binv.dot(Kij.T).dot(SmInv).dot(Kij)
        
        # Look for NaN values in the updated state vector. They should not exist,
        # but if they do, then let's stop the code
        foo = np.where(np.isnan(Xnp1))[0]
        if len(foo) > 0:
            print('Stopping for NaN issue in updated state vector')
            VIP_Databases_functions.abort(date)
            sys.exit()
        
        # Compute information content numbers. The DFS will be computed
        # as the [total, U, V, W]
        tmp = np.diag(Akern)
        dfs=np.array([np.sum(tmp), np.sum(tmp[0:int(nX/3)]), np.sum(tmp[int(nX/3):2*int(nX/3)]), np.sum(tmp[2*int(nX/3):nX])])
        
        sic = 0.5 * np.log(scipy.linalg.det(Sa.dot(SopInv)))
        vres,cdfs = Other_functions.compute_vres_from_akern(Akern,z,do_cdfs=True)
        
        # Compute the N-form and criteria (X space), the M-form is much to computationally expensive
        # so it is not used here.
        if itern == 0:
            # Set the initial RMS and di2 values to large numbers
            
            old_rmsa = 1e20               # RMS for all observations
            old_rmsr = 1e20               # RMS for only the AERI and MWR radiance obs
            old_di2m = 1e20               # di-squared number
        
        #di2n = ((Xn[:,None]-Xnp1[:,None]).T.dot(SopInv).dot(Xn[:,None]-Xnp1[:,None]))[0,0]
        di2n = ((Xn[:,None]-Xnp1).T.dot(SopInv).dot(Xn[:,None]-Xnp1))[0,0]
            
        # Compute the RMS difference between the observation and the
        # forward calculation. However, this will be the relative RMS
        # difference (normalizing by the observation error here), because I
        # am mixing units from all of the different types of observation
        # But I will also compute the chi-square value of the obs vs. F(Xn)
        
        chi2 = np.sqrt(np.sum(((Y - FXn)/ Y)**2) / nY)
        rmsa = np.sqrt(np.sum(((Y - FXn)/sigY)**2) / nY)
        #rmsp = np.mean( (Xa - Xnp1[:,0])/sig_Xa)
        rmsp = np.mean( (Xa - Xnp1[:])/sig_Xa)
        
        # Capture the iteration with the best RMS value
        if rmsa <= old_rmsa:
            old_rmsa = rmsa
            old_iter = itern
        
        # Require the code to go at least 3 iterations before letting it converge
        if itern > 1:
            # Test for "convergence" by looking at the best RMS value
                    
            if ((rmsa > np.sqrt(gfactor[old_iter])*old_rmsa) & (old_iter > 0)):
                converged = 2                   # Converged in "rms increased drastically" sense
                
                Xn = np.copy(xsamp[old_iter]['Xn'])
                FXn = np.copy(xsamp[old_iter]['FXn'])
                Sop = np.copy(xsamp[old_iter]['Sop'])
                Gain = np.copy(xsamp[old_iter]['Gain'])
                Akern = np.copy(xsamp[old_iter]['Akern'])
                vres = np.copy(xsamp[old_iter]['vres'])
                gfac = xsamp[old_iter]['gamma']
                sic = xsamp[old_iter]['sic']
                dfs = np.copy(xsamp[old_iter]['dfs'])
                cdfs = np.copy(xsamp[old_iter]['cdfs'])
                di2n = xsamp[old_iter]['di2n']
                rmsa = xsamp[old_iter]['rmsa']
                rmsp = xsamp[old_iter]['rmsp']
                chi2 = xsamp[old_iter]['chi2']
                itern = old_iter
                
                # But also check for convergence in the normal manner
            if ((gfactor[itern-1] <= 1) & (gfactor[itern] == 1)):
                if di2n < cvgmult * nX:                 # Converged in "classical sense"
                    converged = 1
                
        elif iterate_once:
            
            converged = 4
            
            Xn = np.copy(Xnp1[:,0])
            FXn = np.ones(len(Y))*np.nan
            rmsa = -999
            rmsp = -999
            chi2 = -999
            di2n = -999
            
            
        prev_di2n = di2n
        
        # Place the data into a structure (before we do the update)
        xtmp = {'idx':i, 'secs':rtime[i], 'ymd':date, 'hour':rtime_hour[i], 'nX':nX,
                'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn),
                'Sop':np.copy(Sop), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2n':di2n,
                'rmsa':rmsa, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged, 'max_heights': max_hgts}
        
        if converged == 0:
            Xn = np.copy(Xnp1[:,0])
            Fxnm1 = np.copy(FXn)
            #Xn = np.copy(Xnp1[:])
            if verbose >= 1:
                print('       iter is ' + str(itern) + ' di2n is ' + str(di2n) + ' and RMS is ' + str(rmsa))
            itern += 1
        
                
        # And store each iteration in case we want to investigate how the retrieval
        # function in a sample-by-sampble way

        if itern == 1:
            xsamp = [copy.deepcopy(xtmp)]
        elif converged == 4:
            xsamp = [copy.deepcopy(xtmp)]
        else:
            xsamp.append(copy.deepcopy(xtmp))

        
      
    if converged == 1:
        print('Converged! (di2m << nY')
        print('      final iter is ' + str(itern) + ' di2n is ' + str(di2n) + ' and RMS is ' + str(rmsa))
    elif converged == 2:
        print('Converged (best RMS as RMS drastically increased)')
        print('      final iter is ' + str(old_iter) + ' di2n is ' + str(di2n) + ' and RMS is ' + str(rmsa))
    elif converged == 4:
        if verbose >= 1:
            print('Forward model is linear, no iterations needed')
    else:
        
        # If the retrieval did not converged but performed max_iter iterations
        # means that the RMS didn't really increase drastically at any one step.
        # Let's select the sample that has the best RMS but weight the value
        # so that we are picking it towards the end ofthe iterations (use gamma
        # to do so), and save it
        
        vval = []
        for samp in xsamp:
            vval.append(samp['gamma'] * samp['rmsa'])
        vval = np.array(vval)
        foo = np.where(np.abs(np.min(vval) - vval) < np.min(vval)*1.00001)[0]
        converged = 3           # Converged in "best rms after max_iter" sense
        itern = int(foo[0])
        Xn = np.copy(xsamp[itern]['Xn'])
        FXn = np.copy(xsamp[itern]['FXn'])
        Sop = np.copy(xsamp[itern]['Sop'])
        Gain = np.copy(xsamp[itern]['Gain'])
        Akern = np.copy(xsamp[itern]['Akern'])
        vres = np.copy(xsamp[itern]['vres'])
        gfac = xsamp[itern]['gamma']
        sic = xsamp[itern]['sic']
        dfs = np.copy(xsamp[itern]['dfs'])
        cdfs = np.copy(xsamp[itern]['cdfs'])
        di2n = xsamp[itern]['di2n']
        rmsa = xsamp[itern]['rmsa']
        rmsp = xsamp[itern]['rmsp']
        chi2 = xsamp[itern]['chi2']
                
        xtmp = {'idx':i, 'secs':rtime[i], 'ymd':date, 'hour':rtime_hour[i], 'nX':nX,
                'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(Sm), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn),
                'Sop':np.copy(Sop), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'cdfs':np.copy(cdfs), 'di2n':di2n,
                'rmsa':rmsa, 'rmsp':rmsp, 'chi2':chi2, 'converged':converged, 'max_heights': max_hgts}
        
        xsamp.append(xtmp)
        print('Converged! (best RMS after max_iter)')
    
    endtime = datetime.now()

    # Now store all of the data out. First create a pickle file if debug is turned
    if debug == 1:
        import pickle
        dt = datetime.utcfromtimestamp(rtime[i])
        hr = int(shour)*100+int(((shour-int(shour))*60+.5))
        savename = dt.strftime(f"{vip['output_path']}/{vip['output_rootname']}.%Y%m%d.{hr}.%H%M.pkl")
        out = {'xsamp':xsamp, 'vip':vip, 'Sa':Sa, 'Xa':Xa}
        with open(savename, 'wb') as fh:
            pickle.dump(out,fh)
    
    
    # The retrieval didn't converge
    if xsamp[-1]['converged'] != 1:
        xsamp[-1]['qcflag'] = 2
        
    # The retrieval RMSE is too large
    if xsamp[-1]['rmsa'] > vip['qc_rms_value']:
        xsamp[-1]['qcflag'] = 3
    
    if xsamp[-1]['converged'] == 4:
        xsamp[-1]['qcflag'] = 0
    
    dindices = Other_functions.compute_dindices(xsamp[-1],vip)

    # Write the data into the netCDF file
    success, noutfilename = Output_Functions.write_output(vip, globatt, xsamp[-1], dindices, prior, fsample, (endtime-starttime).total_seconds(),
                                                          noutfilename, shour, verbose)
    
    if success == 0:
        print('Error: Problem saving to output file.')
        VIP_Databases_functions.abort(date)
        sys.exit()
        
    fsample += 1
    
    
totaltime = (endtime - starttime).total_seconds()

print('Processing took ' + str(totaltime) + ' seconds')

# Successful exit
print(('>>> WINDoe retrieval on ' + str(date) + ' ended properly <<<'))
print('--------------------------------------------------------------------')
print(' ')
        
        
        
        
