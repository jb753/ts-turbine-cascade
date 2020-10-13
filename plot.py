"""This script reads in a converged solution and plots some results."""
from ts import ts_tstream_reader
from ts import ts_tstream_cut
import matplotlib.pyplot as plt
import numpy as np
import compflow as cf

# Arbitrary datum temperatures for entropy level
Pdat=16e5
Tdat=1600.

# Load the grid from hdf5 file
tsr = ts_tstream_reader.TstreamReader()
g = tsr.read("output_1.hdf5")

# Block ids
bid_stator = 0
bid_rotor = 1
bid = [bid_stator, bid_rotor]

# Numbers of points
blk = [g.get_block(bidn) for bidn in bid]
ni = [blki.ni for blki in blk]
nj = [blki.nj for blki in blk]
nk = [blki.nk for blki in blk]
jmid = [int((njn+1)/2) for njn in nj]
print(ni)
print(nj)
print(nk)

# Locate leading and trailing edges
ile = []
ite = []
pitch = []
cx = []
for blki in blk:
    rtnow = g.get_bp('rt',blki.bid)
    rnow = g.get_bp('r',blki.bid)
    xnow = g.get_bp('x',blki.bid)
    theta = rtnow/rnow
    dt = theta[-1,0,:] - theta[0,0,:]
    tol = 1e-6
    ile.append(int(np.argmax(dt<dt[0]-tol)))
    ite.append(ni[blki.bid] - int(np.argmax(dt[::-1]<dt[0]-tol)))
    pitch.append(rtnow[0,jmid[blki.bid],0]-rtnow[-1,jmid[blki.bid],0])
    cx.append(xnow[0,jmid[blki.bid],ite[-1]]
            -xnow[0,jmid[blki.bid],ile[-1]] )

# Extract data at planes of interest

stator_inlet = ts_tstream_cut.TstreamStructuredCut()
stator_inlet.read_from_grid(
        g, Pdat, Tdat, bid_stator,
        ist = 0, ien=1,  # First streamwise
        jst = 0, jen=nj[bid_stator],  # All radial
        kst = 0, ken=nk[bid_stator]  # All pitchwise
        )
stator_outlet = ts_tstream_cut.TstreamStructuredCut()
stator_outlet.read_from_grid(
        g, Pdat, Tdat, bid_stator,
        ist = ni[bid_stator]-1, ien=ni[bid_stator],  # Last streamwise
        jst = 0, jen=nj[bid_stator],  # All radial
        kst = 0, ken=nk[bid_stator]  # All pitchwise
        )

rotor_inlet = ts_tstream_cut.TstreamStructuredCut()
rotor_inlet.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = 0, ien=1,  # First streamwise
        jst = 0, jen=nj[bid_rotor],  # All radial
        kst = 0, ken=nk[bid_rotor]  # All pitchwise
        )

rotor_outlet = ts_tstream_cut.TstreamStructuredCut()
rotor_outlet.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = ni[bid_rotor]-1, ien=ni[bid_rotor],  # Last streamwise
        jst = 0, jen=nj[bid_rotor],  # All radial
        kst = 0, ken=nk[bid_rotor]  # All pitchwise
        )

# Stator surface
stator_ps = ts_tstream_cut.TstreamStructuredCut()
stator_ps.read_from_grid(
        g, Pdat, Tdat, bid_stator,
        ist = ile[bid_stator], ien=ite[bid_stator]+1,  # Blade location
        jst = jmid[bid_stator], jen=jmid[bid_stator]+1, # Midspan
        kst = 0, ken=1  # 
        )
stator_ss = ts_tstream_cut.TstreamStructuredCut()
stator_ss.read_from_grid(
        g, Pdat, Tdat, bid_stator,
        ist = ile[bid_stator], ien=ite[bid_stator]+1,  # Blade location
        jst = jmid[bid_stator], jen=jmid[bid_stator]+1, # Midspan
        kst = nk[bid_stator]-1, ken=nk[bid_stator]  # 
        )

# rotor surface
rotor_ss = ts_tstream_cut.TstreamStructuredCut()
rotor_ss.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = ile[bid_rotor], ien=ite[bid_rotor]+1,  # Blade location
        jst = jmid[bid_rotor], jen=jmid[bid_rotor]+1, # Midspan
        kst = 0, ken=1  # 
        )

rotor_ps = ts_tstream_cut.TstreamStructuredCut()
rotor_ps.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = ile[bid_rotor], ien=ite[bid_rotor]+1,  # Blade location
        jst = jmid[bid_rotor], jen=jmid[bid_rotor]+1, # Midspan
        kst = nk[bid_rotor]-1, ken=nk[bid_rotor]  # 
        )

# Blade to blade plane

rotor_b2b = ts_tstream_cut.TstreamStructuredCut()
rotor_b2b.read_from_grid(
        g, Pdat, Tdat, bid_rotor,
        ist = 0, ien=ni[bid_rotor],  # All streamwise
        jst = jmid[bid_rotor], jen=jmid[bid_rotor]+1, # Midspan
        kst = 0, ken=nk[bid_rotor]  # All pitchwise 
        )

stator_b2b = ts_tstream_cut.TstreamStructuredCut()
stator_b2b.read_from_grid(
        g, Pdat, Tdat, bid_stator,
        ist = 0, ien=ni[bid_stator],  # All streamwise
        jst = jmid[bid_stator], jen=jmid[bid_stator]+1, # Midspan
        kst = 0, ken=nk[bid_stator]  # All pitchwise 
        )

# Averaged properties
_, Po1 = stator_inlet.mass_avg_1d('pstag')
_, P2 = stator_outlet.area_avg_1d('pstat')
_, P3 = rotor_inlet.area_avg_1d('pstat')
_, P4 = rotor_outlet.area_avg_1d('pstat')

ga = g.get_av('ga')
_, Ma3 = rotor_inlet.mass_avg_1d('mach_rel')
Po3 = P3 * cf.from_Ma('Po_P',Ma3,ga)

# Calculate pressure coefficient
x_stator = stator_ss.get_bp('x').T
x_stator = x_stator/np.ptp(x_stator)
P_stator = np.vstack((stator_ss.get_bp('pstat'),
                      stator_ps.get_bp('pstat'))).T
Cp_stator = (P_stator - Po1)/(Po1 - P2)

x_rotor = rotor_ss.get_bp('x').T
x_rotor = x_rotor/np.ptp(x_rotor)
P_rotor = np.vstack((rotor_ss.get_bp('pstat'),
                     rotor_ps.get_bp('pstat'))).T
Cp_rotor = (P_rotor - Po3)/(Po3 - P4)

# Plot pressure coefficient
plt.figure(1)
plt.plot(x_stator, Cp_stator, '-x')
plt.xlabel('Axial Chord Fraction, $x/c_x$')
plt.ylabel('Static Pressure Coefficient, $C_p$')
plt.legend(['SS','PS'])
plt.title('Vane Presure Distribution')
plt.tight_layout()
plt.savefig('Cp_vane.pdf')

# Plot pressure coefficient
plt.figure(2)
plt.plot(x_rotor, Cp_rotor, '-x')
plt.xlabel('Axial Chord Fraction, $x/c_x$')
plt.ylabel('Static Pressure Coefficient, $C_p$')
plt.legend(['SS','PS'])
plt.title('Blade Presure Distribution')
plt.tight_layout()
plt.savefig('Cp_blade.pdf')

# Get Cp in blade to blade plane and plot

x_b2b_stator = stator_b2b.get_bp('x')/cx[bid_stator]
rt_b2b_stator = stator_b2b.get_bp('rt')/cx[bid_stator]
P_b2b_stator = stator_b2b.get_bp('pstat')
Cp_b2b_stator = (P_b2b_stator - Po1)/(Po1 - P2)

x_b2b_rotor = rotor_b2b.get_bp('x')/cx[bid_rotor]
rt_b2b_rotor = rotor_b2b.get_bp('rt')/cx[bid_rotor]
P_b2b_rotor = rotor_b2b.get_bp('pstat')
Cp_b2b_rotor = (P_b2b_rotor - Po1)/(Po1 - P2)

plt.figure(3)
lev = np.linspace(-2.2,0.,23)
plt.contourf(x_b2b_stator, rt_b2b_stator, Cp_b2b_stator, lev)
plt.contourf(x_b2b_rotor, rt_b2b_rotor, Cp_b2b_rotor, lev)
plt.contourf(x_b2b_stator,
        rt_b2b_stator+pitch[bid_stator]/cx[bid_stator], Cp_b2b_stator, lev)
plt.contourf(x_b2b_rotor, rt_b2b_rotor+pitch[bid_rotor]/cx[bid_rotor], Cp_b2b_rotor, lev)
plt.grid('off')
plt.title('Static Pressure Coefficient, $C_p$')
plt.xlabel(r'Axial Coordinate, $x/c_x$')
plt.ylabel(r'Pitchwise Coordinate, $r\theta/c_x$')
plt.tight_layout()

plt.savefig('Cp_cont.pdf')
