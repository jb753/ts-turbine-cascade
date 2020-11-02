"""This script reads and plots the blade-to-blade probes."""
import numpy as np  # Multidimensional array library
import probe  # Code for reading TS probe output
import matplotlib.pyplot as plt  # Plotting library
from ts import ts_tstream_reader, ts_tstream_cut  # TS grid reader

#
# Set variables here
#

output_file_name = 'output_2'  # Location of TS output file

# We identify a region of the grid using block and patch IDs
bid_probe = [4, 5, 6, 7 ,8]  # Block ID of probes
pid_probe = [8, 8, 8, 8, 8]   # Patch ID of probes

#
# This next section contains code to read in the data and process it into a
# convenient form. Only a vague undestanding of this section is needed.
#

# Load the grid 
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read(output_file_name + '.hdf5')

# Store all probe patches in a list
Dat = []
for i in range(len(bid_probe)):

    bpi = bid_probe[i]
    ppi = pid_probe[i]

    print('Reading probe %d of %d' % (i+1, len(bid_probe)))

    # Determine the number of grid points on probe patches
    # (We index the TS grid using i = streamwise, j = spanwise, k =
    # pitchwise)
    p = g.get_patch(bpi,ppi)
    di = p.ien - p.ist
    dj = p.jen - p.jst
    dk = p.ken - p.kst
    probe_shape = [di, dj, dk]  # Numbers of points in i, j, k directions

    # Assemble file names for the probes using % substitution
    probe_name = (output_file_name + '_probe_%d_%d.dat' % (bpi,ppi))

    # Read the probes: The dictionary is keyed by variable name; the values
    # are numpy arrays with indexes 
    # [i = streamwise, j = spanwise, k = pitchwise, n = timewise] 
    Dat.append(probe.read_dat(probe_name, probe_shape))

# Here we extract some parameters from the TS grid to use later
rpm = g.get_bv('rpm',1)  # RPM in rotor row
cp = g.get_av('cp')  # Specific heat capacity at const p
ga = g.get_av('ga')  # Specific heat ratio
rgas = cp * (1.-1./ga)

# Get information about time discretisation from TS grid
freq = g.get_av('frequency')  # Blade passing frequency
ncycle = g.get_av('ncycle')  # Number of cycles
nstep_cycle = g.get_av('nstep_cycle')  # Time steps per cycle
# Individual time step in seconds = blade passing period / steps per cycle
dt = 1./freq/float(nstep_cycle)
# Number of time steps = num cycles * steps per cycle
# nt = ncycle * nstep_cycle
nt = np.shape(Dat[0]['ro'])[-1]
print(nt)
# Make non-dimensional time vector = time in seconds * blade passing frequency
ft = np.linspace(0.,float(nt-1)*dt,nt) * freq

# Get secondary vars, things like static pressure, rotor-relative Mach, etc.
Dat = [probe.secondary(Di, rpm, cp, ga) for Di in Dat]

# Arbitrary datum temperatures for entropy level
Pdat=16e5
Tdat=1600.
bid_rotor = 5
b_rotor = g.get_block(bid_rotor)

# Get cuts at rotor inlet and exit to form Cp
rotor_inlet = ts_tstream_cut.TstreamStructuredCut()
rotor_inlet.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = 0, ien=1,  # First streamwise
        jst = 0, jen=b_rotor.nj,  # All radial
        kst = 0, ken=b_rotor.nk  # All pitchwise
        )
rotor_outlet = ts_tstream_cut.TstreamStructuredCut()
rotor_outlet.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = b_rotor.ni-1, ien=b_rotor.ni,  # Last streamwise
        jst = 0, jen=b_rotor.nj,  # All radial
        kst = 0, ken=b_rotor.nk  # All pitchwise
        )

# Pressure references
_, Po1 = rotor_inlet.mass_avg_1d('pstag')
_, P1 = rotor_inlet.area_avg_1d('pstat')
_, P2 = rotor_outlet.area_avg_1d('pstat')

# Temperature references
_, T1 = rotor_inlet.area_avg_1d('tstat')
_, T2 = rotor_outlet.area_avg_1d('tstat')

# Finished reading data, now make some plots
#

# Static pressure
f,a = plt.subplots()  # Create a figure and axis to plot into
plt.set_cmap('cubehelix')
lev = np.linspace(-1.4,0.,21)
# Loop over all blocks
for Di in Dat:
    # Indices
    # :, all x
    # 0, probe is at constant j
    # :, all rt
    # -1, last time step
    xnow = Di['x'][:,0,:,-1]
    rtnow = Di['rt'][:,0,:,-1]
    Pnow = Di['pstat'][:,0,:,-1]
    Cpnow = (Pnow - Po1)/(Po1-P2)
    a.contourf(xnow, rtnow, Cpnow, lev)
a.axis('equal')
plt.grid(False)
plt.tight_layout()  # Remove extraneous white space
plt.show()  # Render the plot
plt.savefig('unst_Cp_cont.pdf')  # Write out a pdf file

# Entropy
f,a = plt.subplots()  # Create a figure and axis to plot into
plt.set_cmap('cubehelix_r')
lev = np.linspace(-8.,25.0,21)
# Loop over all blocks
for Di in Dat:
    # Indices
    # :, all x
    # 0, probe is at constant j
    # :, all rt
    # -1, last time step
    xnow = Di['x'][:,0,:,-1]
    rtnow = Di['rt'][:,0,:,-1]
    Pnow = Di['pstat'][:,0,:,-1]
    Tnow = Di['tstat'][:,0,:,-1]
    # Change in entropy relative to mean upstream state
    Dsnow = cp * np.log(Tnow/T1) - rgas*np.log(Pnow/P1)
    a.contourf(xnow, rtnow, Dsnow, lev)
a.axis('equal')
plt.grid(False)
plt.tight_layout()  # Remove extraneous white space
plt.show()  # Render the plot
plt.savefig('unst_s_cont.pdf')  # Write out a pdf file

