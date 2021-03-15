"""This script reads in an unsteady solution and runs the simple hole model."""
import numpy as np  # Multidimensional array library
import probe  # Code for reading TS probe output
import model  # Simple hole model
import matplotlib.pyplot as plt  # Plotting library
from ts import ts_tstream_reader  # TS grid reader
from ts import ts_tstream_cut  # TS cutter

#
# Set variables here
#

output_file_name = 'output_2'  # Location of TS output file

# We identify a region of the grid using block and patch IDs
pid_probe_ps = 9  # Patch ID of surface probe on pressure side
pid_probe_ss = 10  # Patch ID of surface probe on suction side
pid_probe_ps_free = 11  # Patch ID of free-stream probe on pressure side
pid_probe_ss_free = 12  # Patch ID of free-stream probe on suction side

#
# This next section contains code to read in the data and process it into a
# convenient form. Only a vague undestanding of this section is needed.
#

# Load the grid 
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read(output_file_name + '.hdf5')

# Determine number of blades in each row
bids = [0,g.get_nb()-1]
fracann = np.array([g.get_bv('fracann',bi) for bi in bids])
nblade = np.array([g.get_bv('nblade',bi) for bi in bids])
nb_row = np.round(fracann * nblade)
bid_probe = int(nb_row[0])  # Block ID where probes are located


# Determine the number of grid points on probe patches
# (We index the TS grid using i = streamwise, j = spanwise, k = pitchwise)
p = g.get_patch(bid_probe,pid_probe_ps)
di = p.ien - p.ist
dj = p.jen - p.jst
probe_shape = [di, dj, 1]  # Numbers of points in i, j, k directions

# Index for the mid-span
jmid = int(dj/2)

# Assemble file names for the probes using % substitution
probe_name_ps = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ps)
probe_name_ss = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ss)
probe_name_ps_free = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ps_free)
probe_name_ss_free = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ss_free)

# Read the probes
# The probe data are separate dictionary for each surface of the blade The
# dictionary is keyed by variable name; the values are numpy arrays with
# indexes [i = streamwise, j = spanwise, k = pitchwise, n = timewise]
# For example, to get density at time instant n as a function of
# axial distance at mid-radius:
#   Dat_ps['ro'][:,jmid,0,n]
Dat_ps = probe.read_dat(probe_name_ps, probe_shape)
Dat_ss = probe.read_dat(probe_name_ss, probe_shape)
Dat_ps_free = probe.read_dat(probe_name_ps_free, probe_shape)
Dat_ss_free = probe.read_dat(probe_name_ss_free, probe_shape)

# Here we extract some parameters from the TS grid to use later
rpm = g.get_bv('rpm',1)  # RPM in rotor row
cp = g.get_av('cp')  # Specific heat capacity at const p
ga = g.get_av('ga')  # Specific heat ratio

# Get information about time discretisation from TS grid
freq = g.get_av('frequency')  # Blade passing frequency
ncycle = g.get_av('ncycle')  # Number of cycles
nstep_cycle = g.get_av('nstep_cycle')  # Time steps per cycle
nstep_save_probe = g.get_av('nstep_save_probe')  # Time steps per cycle
# Individual time step in seconds = blade passing period / steps per cycle
dt = 1./freq/float(nstep_cycle)*float(nstep_save_probe)
# Number of time steps = num cycles * steps per cycle
# nt = ncycle * nstep_cycle
nt = np.shape(Dat_ps['ro'])[-1]
# Make non-dimensional time vector = time in seconds * blade passing frequency
ft = np.linspace(0.,float(nt-1)*dt,nt) * freq

# Get secondary vars, things like static pressure, rotor-relative Mach, etc.

Pdat = 1e5
Tdat = 300.

Dat_ps = probe.secondary(Dat_ps, rpm, cp, ga, Pdat, Tdat)
Dat_ss = probe.secondary(Dat_ss, rpm, cp, ga, Pdat, Tdat)
Dat_ps_free = probe.secondary(Dat_ps_free, rpm, cp, ga, Pdat, Tdat)
Dat_ss_free = probe.secondary(Dat_ss_free, rpm, cp, ga, Pdat, Tdat)

# Cut the rotor inlet
b = g.get_block(bid_probe)
rotor_inlet = ts_tstream_cut.TstreamStructuredCut()
rotor_inlet.read_from_grid(
        g, Pdat, Tdat, bid_probe,
        ist = 0, ien=1,  # First streamwise
        jst = 0, jen=b.nj,  # All radial
        kst = 0, ken=b.nk # All pitchwise
        )

# Get mass averaged rotor inlet relative stagnation conditions
_, P1 = rotor_inlet.mass_avg_1d('pstat')
_, Mrel1 = rotor_inlet.mass_avg_1d('mach_rel')
_, To1 = rotor_inlet.mass_avg_1d('tstag_rel')
_, x1 = rotor_inlet.mass_avg_1d('x')
Po1 = model.cf.from_Ma('Po_P',Mrel1,ga)*P1

#
# Set up the simple hole model
#

# Choose a constant "pressure margin", or percentage increase of coolant
# relative to the inlet stagnation condition
PM = 0.04
Poc = (1. + PM) * Po1

# Fix a stagnation temperature ratio, i.e. the coolant is this much colder than
# the main-stream inlet stagnation condition
TR = 0.5
Toc = TR * To1

# Choose a hole position
ihole_ps = 20
ihole_ss = 40

# Pull out data for model

roinf = np.stack((Dat_ps_free['ro'][ihole_ps,jmid,0,:],
                  Dat_ss_free['ro'][ihole_ss,jmid,0,:]))
Vinf = np.stack((Dat_ps_free['vrel'][ihole_ps,jmid,0,:],
                  Dat_ss_free['vrel'][ihole_ss,jmid,0,:]))
Pinf = np.stack((Dat_ps_free['pstat'][ihole_ps,jmid,0,:],
                  Dat_ss_free['pstat'][ihole_ss,jmid,0,:]))

# Nondimensionalise data
Pinf_Poc, roVinf_Po_cpToc = model.normalise(Poc, Toc, Pinf, roinf, Vinf, cp)

# Assume constant Cd
Cd = 0.7

# Calculate BR
BR = model.evaluate( Pinf_Poc, roVinf_Po_cpToc, Cd, ga )

#
# Finished reading data, now make some plots
#

# Plot the hole position
f,a = plt.subplots()  # Create a figure and axis to plot into

x = Dat_ps['x'][:,jmid,0,0]
rt_ps = Dat_ps['rt'][:,jmid,0,0]
rt_ss = Dat_ss['rt'][:,jmid,0,0]
a.plot(x,rt_ps,'-k')  # Blade pressure surface
a.plot(x,rt_ss,'-k')  # Blade suction surface
a.plot(x[ihole_ss],rt_ss[ihole_ss],'b*')  # SS hole location
a.plot(x[ihole_ps],rt_ps[ihole_ps],'r*')  # PS hole location 
plt.axis('equal')
plt.axis('off')
plt.tight_layout()  # Remove extraneous white space
plt.savefig('hole_posn.pdf')  # Write out a pdf file

# Plot the Blowing ratios
f,a = plt.subplots()  # Create a figure and axis to plot into
a.plot(ft, BR.T)
a.set_ylabel('Hole Blowing Ratio, $BR$')
a.set_xlabel('Time, Vane Periods')
plt.tight_layout()  # Remove extraneous white space
plt.savefig('BR.pdf')  # Write out a pdf file

plt.show()  # Render the plots
quit()
#
# Other things to try
#
#   See https://numpy.org/doc/stable/ for documentation on Numpy
#
#   * Frequency spectrum of unsteady pressure at a point, use the Fast
#   Fourier Transform function np.fft.fft( pressure, axis=?) to get Fourier
#   coefficients for a series expansion in time. Get frequencies for the bins
#   using np.fft.fftfreq( len(pressure) , dt)
#   * Time-mean pressure distribution on pressure and suction sides using
#   np.mean with the correct axis argument
#   * Minimum and maximum pressure at each axial location. Use function
#   np.amax( pressure, axis=? ) to take the maximum value over one index (time)
#   There is a counterpart np.amin
#   * Vary the Mach number in `make_design.py` and compare the above for
#   different Mach numbers


