"""This script reads in an unsteady solution and graphs the results."""
import numpy as np  # Multidimensional array library
import probe  # Code for reading TS probe output
import matplotlib.pyplot as plt  # Plotting library
from ts import ts_tstream_reader  # TS grid reader

#
# Set variables here
#

output_file_name = 'output_2'  # Location of TS output file

# We identify a region of the grid using block and patch IDs
pid_probe_ps = 9  # Patch ID of probe on pressure side
pid_probe_ss = 8  # Patch ID of probe on suction side
bid_probe = 1  # Block ID where probes are located

#
# This next section contains code to read in the data and process it into a
# convenient form.
#

# Load the grid 
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read(output_file_name + '.hdf5')

# Determine the number of grid points on probe patches
# (We index the TS grid using i = streamwise, j = spanwise, k = pitchwise)
p = g.get_patch(1,pid_probe_ps)
di = p.ien - p.ist
dj = p.jen - p.jst
probe_shape = [di, dj, 1]  # Numbers of points in i, j, k directions

# Index for the mid-span
jmid = int(dj/2)

# Assemble file names for the probes using % substitution
probe_name_ps = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ps)
probe_name_ss = output_file_name + '_probe_%d_%d.dat' % (bid_probe,pid_probe_ss)

# Read the probes
# The probe data are separate dictionary for each surface of the blade The
# dictionary is keyed by variable name; the values are numpy arrays with
# indexes [i = streamwise, j = spanwise, k = pitchwise, n = timewise]
# For example, to get density at time instant n as a function of
# axial distance at mid-radius:
#   Dat_ps['ro'][:,jmid,0,n]
Dat_ps = probe.read_dat(probe_name_ps, probe_shape)
Dat_ss = probe.read_dat(probe_name_ss, probe_shape)

# Here we extract some parameters from the TS grid to use later
rpm = g.get_bv('rpm',1)  # RPM in rotor row
cp = g.get_av('cp')  # Specific heat capacity at const p
ga = g.get_av('ga')  # Specific heat ratio

# Get information about time discretisation from TS grid
freq = g.get_av('frequency')  # Blade passing frequency
ncycle = g.get_av('ncycle')  # Number of cycles
nstep_cycle = g.get_av('nstep_cycle')  # Time steps per cycle
# Individual time step in seconds = blade passing period / steps per cycle
dt = 1./freq/float(nstep_cycle)
# Number of time steps = num cycles * steps per cycle
nt = ncycle * nstep_cycle
# Make non-dimensional time vector = time in seconds * blade passing frequency
ft = np.linspace(0.,float(nt-1)*dt,nt) * freq

# Get secondary vars, things like static pressure, rotor-relative Mach, etc.
Dat_ps = probe.secondary(Dat_ps, rpm, cp, ga)
Dat_ss = probe.secondary(Dat_ss, rpm, cp, ga)

#
# Finished reading data, now make some plots
#

#
# Plot static pressure at mid-chord on pressure side edge as function of time
#

# Streamwise index at mid-chord = number of points in streamwise dirn / 2
imid = int(di/2)

# Get static pressure at coordinates of interest
# i = imid for mid-chord axial location
# j = jmid for mid-span radial location
# k = 0 because the patch is at const pitchwise position, on pressure surface
# n = : for all instants in time 
P = Dat_ps['pstat'][imid,jmid,0,:]

# Divide pressure by mean value
# P is a one-dimensional vector of values of static pressure at each instant in
# time; np.mean is a function that returns the mean of an array
P_hat = P / np.mean(P)

# Generate the graph
f,a = plt.subplots()  # Create a figure and axis to plot into
a.plot(ft,P_hat,'-')  # Plot our data as a new line
plt.xlabel('Time, Rotor Periods, $ft$')  # Horizontal axis label
plt.ylabel('Static Pressure, $p/\overline{p}$')  # Vertical axis label
plt.tight_layout()  # Remove extraneous white space
plt.show()  # Render the plot
plt.savefig('unsteady_P.pdf')  # Write out a pdf file

#
# Plot time-mean density on pressure side as function of axial location
#

# Get density at coordinates of interest
# i = : for all axial locations
# j = jmid for mid-span radial location 
# k = 0 because the patch is at const pitchwise position, on pressure surface
# n = : for all instants in time
ro = Dat_ps['ro'][:,jmid,0,:]

# Take the time-mean of the density at each axial location
# ro is a 2D matrix of density values over all axial positions and time steps
# The first index is i (axial), second index is n (time)
# We use the np.mean funtion with a keyword argument `axis=1` to specify that we
# want to take the mean over the second index, i.e. in time and not in space.
ro_av = np.mean(ro, axis=1)

# Make non-dimensional with the density at leading edge, at index i=0
ro_hat = ro_av / ro_av[0]

# Get axial coordinates on pressure side 
# i = : for all axial locations
# j = jmid for mid-span radial location 
# k = 0 because the patch is at const pitchwise position, on pressure surface
# n = 0 for first time step, arbitrary because x is not a function of time.
x = Dat_ps['x'][:,jmid,0,0]

# Convert to axial chord fraction; use the array min and max functions to get
# the coordinates at leading and trailing edges respectively.
x_hat = (x - x.min())/(x.max() - x.min())

# Generate the graph
f,a = plt.subplots()  # Create a figure and axis to plot into
a.plot(x_hat,ro_hat,'-')  # Plot our data as a new line
plt.xlabel('Axial Chord Fraction, $\hat{x}$')  # Horizontal axis label
# Vertical axis label, start string with r so that \r is not interpreted as a
# special escape sequence for carriage return
plt.ylabel(
        r'Time-averaged Density, $\overline{\rho}/\overline{\rho}_\mathrm{LE}$')
plt.tight_layout()  # Remove extraneous white space
plt.show()  # Render the plot
plt.savefig('ro_x.pdf')  # Write out a pdf file

