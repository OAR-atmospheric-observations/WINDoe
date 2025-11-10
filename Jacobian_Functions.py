import numpy as np
from scipy import interpolate

###############################################################################
# This file contains the following functions:
# compute_jacobian_raw_lidar()
# compute_jacobian_proc_lidar()
###############################################################################

def compute_jacobian_vr(Xn, z, Y, zY, az,el,itern,do_kij=True):
    
    
    flag = 0         # Failure
    
    k = len(z)
    u = np.copy(Xn[0:k])          # U-component
    v = np.copy(Xn[k:2*k])        # V-component
    w = np.copy(Xn[2*k:3*k])
    
    
    # Find the height of each observation
    rng = zY/np.sin(np.deg2rad(el))
    lx = rng*np.cos(np.deg2rad(el)) * np.sin(np.deg2rad(az))
    ly = rng*np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(az))
    
    A = np.array([np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el)),
                  np.cos(np.deg2rad(az))*np.cos(np.deg2rad(el)),
                  np.sin(np.deg2rad(el))])
    
    # Now get U and V at each height
    f = interpolate.interp1d(z,u,bounds_error = False, fill_value = np.nan)
    u_lz = f(zY)
    
    f = interpolate.interp1d(z,v,bounds_error = False, fill_value = np.nan)
    v_lz = f(zY)
    
    f = interpolate.interp1d(z,w,bounds_error = False, fill_value = np.nan)
    w_lz = f(zY)
    
    V = np.array([u_lz,v_lz,w_lz])
    
    # Calculate radial velocity from u and v
    FXn = np.squeeze(np.matmul(A.T[:,None,:],V.T[:,:,None]))
    
    foo = np.where(Y == -999)
    FXn[foo] = -999.
            
        
    if do_kij:
        Kij = np.zeros((len(FXn),len(Xn)))
        
        for i in range(k):
            foo = np.where(z[i] == zY)[0]
            if len(foo) > 0:
                Kij[foo,i] = np.sin(np.deg2rad(az[foo]))*np.cos(np.deg2rad(el[foo]))
                Kij[foo,k+i] = np.cos(np.deg2rad(az[foo]))*np.cos(np.deg2rad(el[foo]))
                Kij[foo,2*k+i] = np.sin(np.deg2rad(el[foo]))
                
        # Make sure we are not using bad stuff for the jacobian 
        Kij[np.abs(Kij)<1e-6] = 0
        Kij[np.isnan(Kij)] = -999.
        
    FXn[np.isnan(FXn)] = -999.
    flag = 1
    
    
    if do_kij:
        return flag, Kij, FXn
    else:
        return flag, FXn
    
def compute_jacobian_uv(Xn,z,height,uv):
    
    flag = 0
    
    k = len(z)
    u = np.copy(Xn[0:k])          # U-component
    v = np.copy(Xn[k:2*k])        # V-component
    
    # Jacobian for U
    
    if uv == 0:
        f = interpolate.interp1d(z,u,bounds_error = False, fill_value = -999)
        FXn = f(height)
    
        Kij = np.zeros((len(FXn),len(Xn)))
    
        for i in range(k):
            foo = np.where(np.abs(z[i] - height) < 0.00001)[0]
            if len(foo) > 0:
                Kij[foo,i] = 1
    
    if uv == 1:
        f = interpolate.interp1d(z,v,bounds_error = False, fill_value = -999)
        FXn = f(height)
    
        Kij = np.zeros((len(FXn),len(Xn)))
    
    
        for i in range(k):
            foo = np.where(np.abs(z[i] - height) < 0.00001)[0]
            if len(foo) > 0:
                Kij[foo,k+i] = 1
    
    flag = 1
    
    return flag, Kij, FXn

    
def compute_jacobian_copter_pitch(Xn,z,height,yaw,c0,c1):

    flag = 0

    k = len(z)
    u = np.copy(Xn[0:k])
    v = np.copy(Xn[k:2*k])
    
    u_f = interpolate.interp1d(z,u,bounds_error = False, fill_value = -999)
    u_interp = u_f(height)

    v_f = interpolate.interp1d(z,v,bounds_error = False, fill_value = -999)
    v_interp = v_f(height)

    speed = np.sqrt(u_interp**2+v_interp**2)

    # This is the how we get tilt angle from the wind speed.
    tilt_angle = np.rad2deg(np.arctan(((speed-c1)/c0)**2))
    
    wdir = (270 - np.rad2deg(np.arctan2(v_interp,u_interp)))%360
    
    # Find the angle between wind direction and yaw. This is probably overkill
    # but I want to make sure the signs with this are okay and this is the most
    # explicit way to do it
    angle = (np.rad2deg(np.arctan2(np.cos(np.deg2rad(wdir)),np.sin(np.deg2rad(wdir)))) -
             np.rad2deg(np.arctan2(np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)))))
    
    foo = np.where(angle > 180)
    angle[foo] -= 360
        
    foo = np.where(angle < -180)
    angle[foo] += 360
    
    # Here is the forward model for ptich
    FXn = np.rad2deg(np.arcsin(-np.sin(np.deg2rad(tilt_angle))*np.cos(np.deg2rad(angle))))
    
    # Initialize the Jacobian matrix
    Kij = np.zeros((len(FXn),len(Xn)))
    
    for i in range(2*k):
        tmpXn = np.copy(Xn[0:2*k])
        tmpXn[i] += 1
        
        u_tmp = tmpXn[0:k]
        v_tmp = tmpXn[k:2*k]
        
        u_f = interpolate.interp1d(z,u_tmp,bounds_error = False, fill_value = -999)
        u_tmp_interp = u_f(height)
        
        v_f = interpolate.interp1d(z,v_tmp,bounds_error = False, fill_value = -999)
        v_tmp_interp = v_f(height)
        
        tmp_speed = np.sqrt(u_tmp_interp**2+v_tmp_interp**2)
        
        tilt_tmp = np.rad2deg(np.arctan(((tmp_speed-c1)/c0)**2))
        wdir_tmp = (270 - np.rad2deg(np.arctan2(v_tmp_interp,u_tmp_interp)))%360
        
        angle_tmp = (np.rad2deg(np.arctan2(np.cos(np.deg2rad(wdir_tmp)),np.sin(np.deg2rad(wdir_tmp)))) -
                 np.rad2deg(np.arctan2(np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)))))
        
        pitch_tmp = np.rad2deg(np.arcsin(-np.sin(np.deg2rad(tilt_tmp))*np.cos(np.deg2rad(angle_tmp))))
        
        
        Kij[:,i] = (pitch_tmp - FXn)/(tmpXn[i]-Xn[i])
    
    flag = 1
    
    foo = np.where(np.abs(Kij) < 1e-10)
    Kij[foo] = 0
    
    # Now calculate the errors associated with the forward model
    Kb = np.zeros((len(FXn),3))
    
    # First for C0
    Kb[:,0] = ((360*np.cos(np.deg2rad(angle))*(speed-c1)**2) /
               (np.pi*c0**3*((speed-c1)**4/(c0**4)+1)**1.5 *
               (1 - ((np.cos(np.deg2rad(angle))**2*(speed-c1)**4) /
                     (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    # Now for C1
    Kb[:,1] = ((-360*np.cos(np.deg2rad(angle))*(c1-speed)) /
           (np.pi*c0**2*((speed-c1)**4/(c0**4)+1)**1.5 * 
           (1 - ((np.cos(np.deg2rad(angle))**2*(speed-c1)**4) /
                 (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    # Now for the uncertainty of the fit
    Kb[:,2] = ((-360*np.cos(np.deg2rad(angle))*(speed-c1)) /
           (np.pi*c0**2*((speed-c1)**4/(c0**4)+1)**1.5 * 
           (1 - ((np.cos(np.deg2rad(angle))**2*(speed-c1)**4) /
                 (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    return flag, Kij, FXn, Kb

def compute_jacobian_copter_roll(Xn,z,height,yaw,c0,c1):
    
    flag = 0

    k = len(z)
    u = np.copy(Xn[0:k])
    v = np.copy(Xn[k:2*k])
    
    u_f = interpolate.interp1d(z,u,bounds_error = False, fill_value = -999)
    u_interp = u_f(height)

    v_f = interpolate.interp1d(z,v,bounds_error = False, fill_value = -999)
    v_interp = v_f(height)

    speed = np.sqrt(u_interp**2+v_interp**2)

    # This is the how we get tilt angle from the wind speed.
    tilt_angle = np.rad2deg(np.arctan(((speed-c1)/c0)**2))
    
    wdir = (270 - np.rad2deg(np.arctan2(v_interp,u_interp)))%360
    
    # Find the angle between wind direction and yaw. This is probably overkill
    # but I want to make sure the signs with this are okay and this is the most
    # explicit way to do it
    angle = (np.rad2deg(np.arctan2(np.cos(np.deg2rad(wdir)),np.sin(np.deg2rad(wdir)))) -
             np.rad2deg(np.arctan2(np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)))))
    
    foo = np.where(angle > 180)
    angle[foo] -= 360
        
    foo = np.where(angle < -180)
    angle[foo] += 360
    
    # Here is the forward model for roll
    FXn = np.rad2deg(np.arcsin(-np.sin(np.deg2rad(tilt_angle))*np.sin(np.deg2rad(angle))))
    
    # Initialize the Jacobian matrix
    Kij = np.zeros((len(FXn),len(Xn)))
    
    for i in range(2*k):
        tmpXn = np.copy(Xn[0:2*k])
        tmpXn[i] += 1
        
        u_tmp = tmpXn[0:k]
        v_tmp = tmpXn[k:2*k]
        
        u_f = interpolate.interp1d(z,u_tmp,bounds_error = False, fill_value = -999)
        u_tmp_interp = u_f(height)
        
        v_f = interpolate.interp1d(z,v_tmp,bounds_error = False, fill_value = -999)
        v_tmp_interp = v_f(height)
        
        tmp_speed = np.sqrt(u_tmp_interp**2+v_tmp_interp**2)
        
        tilt_tmp = np.rad2deg(np.arctan(((tmp_speed-c1)/c0)**2))
        wdir_tmp = (270 - np.rad2deg(np.arctan2(v_tmp_interp,u_tmp_interp)))%360
        
        angle_tmp = (np.rad2deg(np.arctan2(np.cos(np.deg2rad(wdir_tmp)),np.sin(np.deg2rad(wdir_tmp)))) -
                 np.rad2deg(np.arctan2(np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)))))
        
        roll_tmp = np.rad2deg(np.arcsin(-np.sin(np.deg2rad(tilt_tmp))*np.sin(np.deg2rad(angle_tmp))))
        
        Kij[:,i] = (roll_tmp - FXn)/(tmpXn[i]-Xn[i])
        
    flag = 1
    
    foo = np.where(np.abs(Kij) < 1e-10)
    Kij[foo] = 0
    
    # Now calculate the errors associated with the forward model
    Kb = np.zeros((len(FXn),3))
    
    # First for C0
    Kb[:,0] = ((360*np.sin(np.deg2rad(angle))*(speed-c1)**2) /
               (np.pi*c0**3*((speed-c1)**4/(c0**4)+1)**1.5 *
               (1 - ((np.sin(np.deg2rad(angle))**2*(speed-c1)**4) /
                     (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    # Now for C1
    Kb[:,1] = ((-360*np.sin(np.deg2rad(angle))*(c1-speed)) /
           (np.pi*c0**2*((speed-c1)**4/(c0**4)+1)**1.5 * 
           (1 - ((np.sin(np.deg2rad(angle))**2*(speed-c1)**4) /
                 (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    # Now for the uncertainty of the fit
    Kb[:,2] = ((-360*np.sin(np.deg2rad(angle))*(speed-c1)) /
           (np.pi*c0**2*((speed-c1)**4/(c0**4)+1)**1.5 * 
           (1 - ((np.sin(np.deg2rad(angle))**2*(speed-c1)**4) /
                 (c0**4 * ((speed-c1)**4/(c0**4)+1))))**0.5))
    
    return flag, Kij, FXn, Kb
        