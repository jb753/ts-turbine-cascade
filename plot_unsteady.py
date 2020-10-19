import numpy as np
import probe
import matplotlib.pyplot as plt
from ts import ts_tstream_reader

# Load the grid from hdf5 file
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read("output_2.hdf5")

# Load probe dat
p = g.get_patch(1,8)
di = p.ien - p.ist
dj = p.jen - p.jst
probe_shape = [di, dj, 1]
Dat_ps = probe.read_dat('output_2_probe_1_9.dat', probe_shape)
Dat_ss = probe.read_dat('output_2_probe_1_8.dat', probe_shape)

# The probe data are separate for each surface of the blade
# A dictionary keyed by variable name
# With numpy arrays with indexes [streamwise, spanwise, pitchwise, timewise]

# Get secondary vars
rpm = g.get_bv('rpm',1)
cp = g.get_av('cp')
ga = g.get_av('ga')
freq = g.get_av('frequency')
ncycle = g.get_av('ncycle')
nstep_cycle = g.get_av('nstep_cycle')
dt = 1./freq/float(nstep_cycle)
nt = ncycle * nstep_cycle
Dat_ps = probe.secondary(Dat_ps, rpm, cp, ga)
Dat_ss = probe.secondary(Dat_ss, rpm, cp, ga)

# Plot pressure on leading edge as function of time
f,a = plt.subplots()
t_nondim = np.linspace(0.,float(nt-1)*dt,nt)* freq
P = Dat_ps['pstat'][0,3,0,:]
a.plot(t_nondim,P / np.mean(P),'-')
plt.xlabel('Time, Rotor Periods, $ft$')
plt.ylabel('Static Pressure, $p/\overline{p}$')
plt.tight_layout()
plt.savefig('unsteady_P.pdf')
plt.show()
