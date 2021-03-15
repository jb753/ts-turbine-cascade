"""This script reads and plots the blade-to-blade probes."""
import numpy as np  # Multidimensional array library
import probe  # Code for reading TS probe output
import matplotlib.pyplot as plt  # Plotting library
from ts import ts_tstream_reader, ts_tstream_cut, ts_tstream_patch_kind  # TS grid reader

#
# Set variables here
#

output_file_name = 'output_2'  # Location of TS output file

# Load the grid 
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read(output_file_name + '.hdf5')


# Set the probe locations manually
bid_probe = range(20)
pid_probe = [8,]*9 + [8,]*11
print(np.array([bid_probe,pid_probe]).astype(int))

# Verify that these probes exist
for bid, pid in zip(bid_probe,pid_probe):
    try:
        pnow = g.get_patch(bid, pid)
        if not pnow.kind ==  ts_tstream_patch_kind.probe:
            raise Exception()
    except:
        raise Exception(
            "Probe patch bid=%d, pid=%d not found!\n"
            "List the patches using `list_probes.py`, and try again."% (bid, pid)
            )


# This next section contains code to read in the data and process it into a
# convenient form. Only a vague undestanding of this section is needed.
#

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
rpm = g.get_bv('rpm',g.get_nb()-1)  # RPM in rotor row
print(rpm)
Omega = rpm / 60. * np.pi * 2.
print(Omega)
cp = g.get_av('cp')  # Specific heat capacity at const p
ga = g.get_av('ga')  # Specific heat ratio
rgas = cp * (1.-1./ga)

# Get information about time discretisation from TS grid
freq = g.get_av('frequency')  # Blade passing frequency
print(freq)
ncycle = g.get_av('ncycle')  # Number of cycles
nstep_cycle = g.get_av('nstep_cycle')  # Time steps per cycle
print(nstep_cycle)
nstep_save_probe = g.get_av('nstep_save_probe')  # Time steps per cycle
# Individual time step in seconds = blade passing period / steps per cycle
dt = 1./freq/float(nstep_cycle)

# Number of time steps = num cycles * steps per cycle
# nt = ncycle * nstep_cycle
nt = np.shape(Dat[0]['ro'])[-1]
print(dt)
# Make non-dimensional time vector = time in seconds * blade passing frequency


# Arbitrary datum temperatures for entropy level
Pdat=16e5
Tdat=1600.
bid_rotor = g.get_nb()-1
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
_, Po1 = rotor_inlet.mass_avg_1d('pstag_rel')
_, P1 = rotor_inlet.area_avg_1d('pstat')
_, P2 = rotor_outlet.area_avg_1d('pstat')

# Temperature references
_, T1 = rotor_inlet.area_avg_1d('tstat')
_, T2 = rotor_outlet.area_avg_1d('tstat')

rpms = [g.get_bv('rpm',b) for b in g.get_block_ids()]
print(rpms)
# Get secondary vars, things like static pressure, rotor-relative Mach, etc.
Dat = [probe.secondary(Di, rpmi, cp, ga, P1, T1) for Di, rpmi in zip(Dat,rpms)]

# Determine sector size
dtheta_sector = 2. * np.pi * g.get_bv('fracann',0)

# Angular movement per time step
dtheta_step = Omega * dt

# We must add this as an offset between the blade rows because the mesh is
# moved to the next time step before the probes are written out!

# Finished reading data, now make some plots

# Time-avg

f,a = plt.subplots(figsize=(4.,4.))  # Create a figure and axis to plot into
plt.set_cmap('cubehelix_r')
lev = np.linspace(-8.,35.0,21) # For entropy
probe.render_frame(a, Dat,'ds', 0, lev, Omega, dt*nstep_save_probe,nstep_cycle/nstep_save_probe, dtheta_sector, norm=(0.,1.),avg=True)
plt.grid(False)
a.axis('square')
a.axis('off')
# a.set_ylim([0.,dtheta_sector*Dat[0]['r'][0,0,0,0]])
a.set_xlim([-0.06,0.34])
a.set_ylim([0.,0.4])
plt.tight_layout(pad=0.1)  # Remove extraneous white space
plt.savefig('avg.png',dpi=100)
plt.close(f)

quit()

# Movie
for it in range(nt):
    print('Rendering frame %d/%d'%(it,nt))
    f,a = plt.subplots(figsize=(4.,4.))  # Create a figure and axis to plot into
    plt.set_cmap('RdBu')
    plt.set_cmap('cubehelix_r')
    lev = np.linspace(-8.,35.0,21) # For entropy
    # lev = np.linspace(-.04,.04,20)
    # probe.render_frame(a, Dat,'pfluc', it, lev, Omega, dt*nstep_save_probe,nstep_cycle/nstep_save_probe, dtheta_sector, norm=(0,Po1-P2))
    probe.render_frame(a, Dat,'ds', it, lev, Omega, dt*nstep_save_probe,nstep_cycle/nstep_save_probe, dtheta_sector, norm=(0.,1.))
    plt.grid(False)
    a.axis('square')
    a.axis('off')
    # a.set_ylim([0.,dtheta_sector*Dat[0]['r'][0,0,0,0]])
    a.set_xlim([-0.06,0.34])
    a.set_ylim([0.,0.4])
    plt.tight_layout(pad=0.1)  # Remove extraneous white space
    plt.savefig('%d.png'%it,dpi=100)
    plt.close(f)

plt.show()  # Render the plot
