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

# Get secondary vars
rpm = g.get_bv('rpm',1)
cp = g.get_av('cp')
ga = g.get_av('ga')
Dat_ps = probe.secondary(Dat_ps, rpm, cp, ga)
Dat_ss = probe.secondary(Dat_ss, rpm, cp, ga)

f,a = plt.subplots()
# a.plot(Dat_ss['x'][:,3,0,0],Dat_ss['rt'][:,3,0,0])
# a.plot(Dat_ps['x'][:,3,0,0],Dat_ps['rt'][:,3,0,0])
d.plot(Dat_ps['pstat'][0,3,0,:],'-x')
plt.savefig('test.pdf')
plt.show()
