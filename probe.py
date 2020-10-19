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

def secondary(d, rpm, cp, ga):
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

    return d

