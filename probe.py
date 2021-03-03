"""This file contains functions for reading TS probe data."""
import numpy as np

def read_dat(fname, shape):
    """Load flow data from a .dat file"""

    # Get raw data
    raw = np.genfromtxt(fname, skip_header=1, delimiter=' ')

    # Reshape to correct size
    nvar = 8
    shp = np.append(shape, -1)
    shp = np.append(shp, nvar)
    raw = np.reshape(raw, shp, order='F')

    # Split the columns into named variables
    Dat = {}
    varnames = ['x', 'r', 'rt', 'ro', 'rovx', 'rovr', 'rorvt', 'roe']
    for i, vi in enumerate(varnames):
        Dat[vi] = raw[:,:,:,:,i]

    return Dat

def secondary(d, rpm, cp, ga, Pref, Tref):
    # Calculate other variables

    # Velocities
    d['vx'] = d['rovx']/d['ro']
    d['vr'] = d['rovr']/d['ro']
    d['vt'] = d['rorvt']/d['ro']/d['r']
    d['U'] = d['r'] * rpm / 60. * np.pi * 2.
    d['vtrel'] = d['vt'] - d['U']
    d['v'] = np.sqrt(d['vx']**2. + d['vr']**2. + d['vt']**2.)
    d['vrel'] = np.sqrt(d['vx']**2. + d['vr']**2. + d['vtrel']**2.)

    # Total energy for temperature
    E = d['roe']/d['ro']
    cv = cp/ga
    d['tstat'] = (E - 0.5*d['v']**2.)/cv

    # Pressure from idea gas law
    rgas = cp - cv
    d['pstat'] = d['ro'] * rgas * d['tstat']

    # Entropy change wrt reference
    d['ds'] = cp * np.log(d['tstat']/Tref) - rgas*np.log(d['pstat']/Pref)

    # Pressure fluc wrt time mean
    d['pfluc'] = d['pstat'] - np.mean(d['pstat'],3,keepdims=True)

    # Angular velocity
    d['omega'] = rpm / 60. * 2. * np.pi

    # Blade speed
    d['U'] = d['omega'] * d['r']

    # Save the parameters
    d['rpm'] = rpm
    d['cp'] = cp
    d['ga'] = ga

    return d

def render_frame(a, d, varname, it, lev, Omega, dt, nstep_cycle, sector_size, norm=None):
    """Plot out contours of a variable at time step it."""

    for i, di in enumerate(d):
        xnow = di['x'][:,0,:,it]
        rtnow = di['rt'][:,0,:,it]
        rnow = di['r'][:,0,:,it]
        varnow = di[varname][:,0,:,it]+0.

        if norm is not None:
            varnow = (varnow - norm[0])/norm[1]
            print(varnow.min())
            print(varnow.max())

        print(np.shape(varnow))

        # If this is a stator, offset backwards by del_theta
        if di['rpm']==0.:
            rtnow = (rtnow/rnow - Omega*it*dt) * rnow

        # Duplicate plots so the screen is always full
        for ii in range(-2,6):
            if lev is None:
                a.contourf(xnow, (rtnow/rnow + ii*sector_size)*rnow, varnow)
            else:
                a.contourf(xnow, (rtnow/rnow + ii*sector_size)*rnow, varnow, lev)

