import sys
import numpy as np
import scipy.io
import scipy.interpolate
from datetime import datetime
from netCDF4 import Dataset

###############################################################################
# This file contains the following functions:
# compute_vres_from_akern()
###############################################################################

def compute_vres_from_akern(akern,z,do_cdfs = False, do_area = False):
    
    vres = np.zeros((3,len(z)))
    cdfs = np.zeros((3,len(z)))
    area = np.zeros((3,len(z)))
    
    # First get the vertical spacing between retrieval levels
    k = len(z)
    zres = [(z[1]-z[0])/2.]
    for i in range(1,k-1):
        zres.append((z[i+1]-z[i-1])/2.)
    
    zres.append((z[i]-z[i-1])*2)
    zres = np.array(zres)
    
    # Now scale the diagonal of the averaging kernal by this 
    # vertical spacing
    mval = 0.0001        # This is the minimum value for the DFS for the first level
    if akern[0,0] > mval:
        uval = akern[0,0]
    else:
        uval = mval
    
    if akern[k,k] > mval:
        vval = akern[k,k]
    else:
        vval = mval
        
    if akern[2*k,2*k] > mval:
        wval = akern[2*k,2*k]
    else:
        wval = mval
    
    for i in range(k):
        # Watch for zeros along the averaging kernal diagonal. If that
        # happens, then use the last good value for the vres calculation
        if akern[i,i] > 0:
            uval = akern[i,i]
        if akern[k+i,k+i] > 0:
            vval = akern[k+i,k+i]
        if akern[2*k+i,2*k+i] > 0:
            wval = akern[2*k+i,2*k+i]
        
        # capture the cumlative DFS profile
        if i == 0:
            cdfs[0,i] = akern[i,i]
            cdfs[1,i] = akern[k+i,k+i]
            cdfs[2,i] = akern[2*k+i,2*k+i]
        else:
            cdfs[0,i] = akern[i,i] + cdfs[0,i-1]
            cdfs[1,i] = akern[k+i,k+i] + cdfs[1,i-1]
            cdfs[2,i] = akern[2*k+i,2*k+i] + cdfs[2,i-1]
            
        # This is the Hewison method
        vres[0,i] = zres[i] / uval             # U-component
        vres[1,i] = zres[i] / vval             # V-component
        vres[2,i] = zres[i] / wval             # W-component
        
    # Now compute the area of the averaging kernal (pg 56 in Rodger)
    tmp = np.copy(akern[0:k,0:k])
    area[0,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    tmp = np.copy(akern[k:2*k,k:2*k])
    area[1,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    tmp = np.copy(akern[2*k:3*k,2*k:3*k])
    area[2,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    
    if do_cdfs and do_area:
        return vres, area, cdfs
    elif do_cdfs:
        return vres, cdfs
    elif do_area:
        return vres, area
    else:
        return vres

# This is the ARM VAD function that is used to get error estimates for the radial
# velocity values
    
def wind_estimate(vr,el,az,ranges,eff_N,sig_thresh = 9,default_sigma=100000,missing=-999):
    
    # Initialize the arrays
    sigma = np.ones(len(ranges))*np.nan
    thresh_sigma = np.ones(len(ranges))*np.nan
    
    coords = az + 1j*el
    # Loop over all the ranges
    for i in range(len(ranges)):

        # Search for missing data
        
        foo = np.where((~np.isnan(vr[:,i])) & (vr[:,i]!=missing))[0]
        junk, count = np.unique(coords[foo],return_counts=True)
        
        # Need at least 4 unique positions
        if len(count) < 4:
            sigma[i] = default_sigma
            continue
        
        if ((eff_N > 0) & (eff_N < len(foo)-3)):
            N = np.copy(eff_N)
        else:
            N = len(foo)-3
            
        A = np.ones((len(foo),3))
        A[:,0] = np.sin(np.deg2rad(az[foo]))*np.cos(np.deg2rad(el[foo]))
        A[:,1] = np.cos(np.deg2rad(az[foo]))*np.cos(np.deg2rad(el[foo]))
        A[:,2] = np.sin(np.deg2rad(el[foo]))
        
        
        # Solve for the wind components
        v0 = (np.linalg.pinv(A.T.dot(A))).dot(A.T).dot(vr[foo,i])
        
        #sigma[i] = np.sqrt(np.nansum((vr[foo,i] - A.dot(v0))**2)/(len(foo)-3))
        
        sigma[i] = np.sqrt(np.nansum((vr[foo,i] - A.dot(v0))**2)/N)
        thresh_sigma[i] = np.sqrt(np.nansum((vr[foo,i] - A.dot(v0))**2)/(len(foo) - 3))

        # We are going to try this QC. If there is a significant w component
        # determine the uncertainty with no w component
        
        if np.abs(v0[2]) >= 3:
            vh_0 = (np.linalg.pinv(A[:,:-1].T.dot(A[:,:-1]))).dot(A[:,:-1].T).dot(vr[foo,i])
            
            temp_sigma = np.sqrt(np.nansum((vr[foo,i] - A[:,:-1].dot(vh_0))**2))/(N+1)
            temp_thresh_sigma = np.sqrt(np.nansum((vr[foo,i] - A[:,:-1].dot(vh_0))**2))/((len(foo) - 3) + 1)
            
            if temp_sigma > sigma[i]:
                sigma[i] = np.copy(temp_sigma)
                thresh_sigma[i] = np.copy(temp_thresh_sigma)
    
    sigma[thresh_sigma > sig_thresh] = default_sigma
    
    return sigma, thresh_sigma

def consensus_average(x, width, cutoff, min_percentage,missing=-999.):
    
    data = np.copy(x)
    data[data==missing] = np.nan
    
    data_length = data.shape[0]
    
    mean = np.nanmean(x,axis=0)
    
    consensus = np.ones(mean.shape)*np.nan
    variance = np.ones(mean.shape)*np.nan
    for i in range(len(mean)):
        foo = np.where((data[:,i] >= mean[i]-cutoff) & (data[:,i] <= mean[i]+cutoff))[0]
        if ((len(foo) > 2) and ((len(foo)/data_length)*100 >= min_percentage)):
            consensus[i] = np.nanmean(data[foo,i])
            # We are going to combine the varariance of the winds and the average spectrum width
            variance[i] = np.nanvar(data[:,i]) + np.nanmean(width[foo,i])
    
    return consensus, variance

def mean_azimuth(a, b, weight_a=0.6):
        """ Return weighted mean azimuth in the middle of a and b
          Args:
            a: starting azimuth.
            b: ending azimuth.
            weight_a: weight of a; so (1-weight_a) is the weight of b. The value
                      0.6 was empirically chosen and is still questionable.
          Returns:
            w_mean: weighted mean.
          """
        
        if abs(a - b) > 180:
            if a < b:
                return ((a + 360) * weight_a + b * (1 - weight_a)) % 360
            else:
                return ((b + 360) * (1 - weight_a) + a * weight_a) % 360
        else:
            return a * weight_a + b * (1 - weight_a)

def compute_storm_motion(u_winds, v_winds, z):
        
    # First make sure that the wind profile goes from ~0 to 6000 m.
    
    max_z = np.nanmax(z)
    min_z = np.nanmin(z)
    
    if ((max_z < 6) or (min_z > 50)):
        return {'bunkers_right_u': -999., 'bunkers_right_v': -999.}
    
    # Now find mean wind between 0 and 6 km
    foo = np.where(z <= 6)[0]
    
    if len(foo) < 2:
        return {'bunkers_right_u': -999., 'bunkers_right_v': -999.}
    
    mean_u = np.nanmean(u_winds[foo])
    mean_v = np.nanmean(v_winds[foo])
    
    # Now find mean wind between 0-0.5 km
    foo = np.where(z <= 0.5)[0]
    if len(foo) == 0:
        return {'bunkers_right_u': -999., 'bunkers_right_v': -999.}
    
    mean_u_low = np.nanmean(u_winds[foo])
    mean_v_low = np.nanmean(v_winds[foo])
    
    # Now find mean wind between 0-0.5 km
    foo = np.where((z >= 5.5) & (z <= 6))[0]
    if len(foo) == 0:
        return {'bunkers_right_u': -999., 'bunkers_right_v': -999.}
    
    mean_u_high = np.nanmean(u_winds[foo])
    mean_v_high = np.nanmean(v_winds[foo])
    
    # Calculate the shear vector
    u_shear = mean_u_high - mean_u_low
    v_shear = mean_v_high - mean_v_low
    
    # Find the storm motion
    bunkers_right_u = mean_u + 7.5*(v_shear/(np.sqrt(u_shear**2+v_shear**2)))
    bunkers_right_v = mean_v + 7.5*(-u_shear/(np.sqrt(u_shear**2+v_shear**2)))
    
    return {'bunkers_right_u': bunkers_right_u, 'bunkers_right_v': bunkers_right_v}

def compute_srh(u_winds,v_winds,storm_motion, z, z_top):
    
    # Find heights below z_top
    foo = np.where(z <= z_top)[0]
    
    if len(foo) < 2:
        return -999.
    
    u = u_winds[foo]
    v = v_winds[foo]
    
    # Compute the integral
    integral = ((u[1:]-storm_motion['bunkers_right_u'])*(v[:-1]-storm_motion['bunkers_right_v']) -
                (u[:-1]-storm_motion['bunkers_right_u'])*(v[1:]-storm_motion['bunkers_right_v']))
    
    srh = np.sum(integral)
    
    return srh  

def compute_dindices(xret, vip, num_mc=20):
    
    dindex_name = ['srh_1km', 'srh_3km']
    
    dindex_units = ['m^2/s^2', 'm^2/s^2']
    
    indices = np.zeros(len(dindex_name))
    sigma_indices = np.zeros(len(dindex_name))
    
    k = len(xret['z'])
    
    z = np.copy(xret['z'])
    
    u_wind = np.copy(xret['Xn'][0:k])
    v_wind = np.copy(xret['Xn'][k:2*k])
    
    # Get the posterior covariance matrix and 
    # Extract out the posterior covariance matrix, and get the 1-sigma uncertainties
    Sop_tmp = np.copy(xret['Sop'][0:2*k,0:2*k])
    
    # First get the storm motion vector
    storm_motion = compute_storm_motion(u_wind, v_wind, z)
    
    # Calculate 0-1 km srh first
    
    if (storm_motion['bunkers_right_u'] < -500):
        indices[0] = -9999.
    else:
        indices[0] = compute_srh(u_wind, v_wind, storm_motion,z,1)
    
    # Calculate 0-3 km srh
    
    if (storm_motion['bunkers_right_u'] < -500):
        indices[1] = -9999.
    else:
        indices[1] = compute_srh(u_wind, v_wind, storm_motion,z,3)
    
    # Only compute the uncertainties in the indices if the number of Monte Carlo 
    # samples is strictly positive; otherwise, return -9999 as the uncertainties
    if num_mc <= 0:
        print('      Not computing the uncertainties in the derived indices')
        sigma_indices = np.zeros_like(indices) - 9999.
    else:
        # Perform SVD of posterior covariance matrix
        u, w, v, = scipy.linalg.svd(Sop_tmp.T,False)
        
        # Generate the Monte Carlo profiles
        b = np.random.default_rng().normal(size=(2*k,num_mc))
        pert = u.dot(np.diag(np.sqrt(w))).dot(b)
        uprofs = u_wind[:,None] + pert[0:k,:]
        vprofs = v_wind[:,None] + pert[k:2*k,:]
        
        # allocate room to compute the indices for the various profiles in the MC sampling
        tmp_srh1 = np.zeros(num_mc)
        tmp_srh3 = np.zeros(num_mc)
        
        for j in range(num_mc):
            
            # Compute the storm_motion
            tmp_storm_motion = compute_storm_motion(uprofs[:,j], vprofs[:,j],z)
            
            if tmp_storm_motion['bunkers_right_u'] < -500:
                tmp_srh1[j] = -9999.
                tmp_srh3[j] = -9999.
            
            else:
                tmp_srh1[j] = compute_srh(uprofs[:,j],vprofs[:,j],tmp_storm_motion,z,1)
                tmp_srh3[j] = compute_srh(uprofs[:,j],vprofs[:,j],tmp_storm_motion,z,3)
        
        # srh1
        foo = np.where(tmp_srh1 > -9000)[0]
        if ((len(foo) > 1) & (indices[0] > -9000)):
            sigma_indices[0] = np.nanstd(indices[0]-tmp_srh1[foo])
        else:
            sigma_indices[0] = -9999.
        
        # shr3
        foo = np.where(tmp_srh3 > -9000)[0]
        if ((len(foo) > 1) & (indices[1] > -9000)):
            sigma_indices[1] = np.nanstd(indices[1]-tmp_srh3[foo])
        else:
            sigma_indices[1] = -9999.
    
    return {'indices':indices, 'sigma_indices':sigma_indices, 'name':dindex_name,
            'units':dindex_units}
                 
        
def get_w_covariance(z,mean,lengthscale):
    
    cov = mean**2*np.exp(-(np.subtract.outer(z,z))**2/(2*(lengthscale**2)))
    
    return cov
    
    
    
    
