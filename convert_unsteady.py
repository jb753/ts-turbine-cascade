"""This script reads a steady TS solution and creates an unsteady input file."""
import numpy as np  # Multidimensional array library
# Load some TS librays
from ts import ts_tstream_reader, ts_tstream_steady_to_unsteady, ts_tstream_type
from ts import ts_tstream_load_balance, ts_tstream_patch_kind

if __name__ == "__main__":

    # Number of rotor blade passing periods to run for
    # Change me so that the computaion reaches a periodic state
    ncycle = 16

    # Time steps per cycle 
    # Increased a bit because usually have fewer rotor blades
    nstep_cycle = 96

    # Which time step to start saving probes
    nstep_save_start =  (ncycle-1)*nstep_cycle

    # Number of time steps between probes
    nstep_save_probe = 1

    # File name of the steady soln to read in
    fname = "output_1.hdf5"

    # File name of the new unsteady input file to write out
    fname_out = "input_2.hdf5"

    # Put blade-to-blade probes on vane?
    vane_b2b_probe = True

    #
    # Should not need to change below this line
    # It is complicated and not neccesary to understand the below!
    #

    # Read in the converged steady solution
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(fname)
    bids = g.get_block_ids()

    # add blade-to-blade probe patches
    if vane_b2b_probe:
        bid_b2b = [0, 1]
    else:
        bid_b2b = [1]

    for bid in bid_b2b:
        b = g.get_block(bid)
        jmid = int(b.nj/2)
        p = ts_tstream_type.TstreamPatch()
        p.kind = ts_tstream_patch_kind.probe
        p.bid = bid
        p.ist = 0
        p.ien = b.ni
        p.jst = jmid
        p.jen = jmid+1
        p.kst = 0
        p.ken = b.nk
        p.nxbid = 0
        p.nxpid = 0
        p.idir = 0
        p.jdir = 1
        p.kdir = 2
        p.pid = g.add_patch(bid, p)
        g.set_pv('probe_append', ts_tstream_type.int, p.bid, p.pid, 1)

    # find the sector size required
    nb = np.array([g.get_bv('nblade',bid) for bid in bids])
    dt = 2. * np.pi / nb

    # loop over possible sector sizes
    sect_max = nb.max()
    sect_max = 9
    cost = np.empty((sect_max,))
    for i in range(sect_max):

        # Get scaling factor needed to fit the blades in this sector
        dt_sector = 2. * np.pi / float(i+1)
        nb_sector = np.round(dt_sector / dt)
        sf_sector = dt_sector / nb_sector / dt

        cost[i] = np.sum(np.abs(sf_sector - 1.))

    # pick smallest sector with acceptable cost
    thresh = 0.1
    sect = (np.nonzero(cost<thresh))[-1][-1]
    dt_sect = 2. * np.pi / sect


    # get blade numbers
    dup = np.round(dt_sect/ dt)
    scale = dt_sect / dup / dt

    # set frequency 
    rpm = g.get_bv('rpm',bids[-1])
    freq = rpm / 60. * (sect * dup)[0]
    print 'frequency', freq, 'Hz'

    # perodic patches
    periodic = {}
    periodic[0] = (0,0)
    periodic[1] = (0,2)
    periodic[2] = (1,0)
    periodic[3] = (1,2)
    print(periodic)

    dup_int = [int(dupi) for dupi in dup]
    g2 = ts_tstream_steady_to_unsteady.steady_to_unsteady(
            g, dup_int, scale, periodic
            )

    # Get blade indices from circumferential periodics
    p1 = g.get_patch(1,0)
    p2 = g.get_patch(1,2)
    ist = p1.ien
    ien = p2.ist

    # add blade probe patches
    bid_pr = int(dup_int[0])
    b = g2.get_block(bid_pr)
    dk_free = 12
    for k in [0, b.nk-1, dk_free, b.nk-1-dk_free]:
        p = ts_tstream_type.TstreamPatch()
        p.kind = ts_tstream_patch_kind.probe
        p.bid = bid_pr
        p.ist = ist
        p.ien = ien
        p.jst = 0
        p.jen = b.nj
        p.kst = k
        p.ken = k+1
        p.nxbid = 0
        p.nxpid = 0
        p.idir = 0
        p.jdir = 1
        p.kdir = 2
        p.pid = g2.add_patch(bid_pr, p)
        g.set_pv('probe_append', ts_tstream_type.int, p.bid, p.pid, 1)

    # variables for unsteady run
    g2.set_av("ncycle", ts_tstream_type.int, ncycle)
    g2.set_av("frequency", ts_tstream_type.float,  freq)

    g2.set_av("nstep_cycle", ts_tstream_type.int, nstep_cycle)
    g2.set_av("nstep_inner", ts_tstream_type.int, 100)

    # disable saving of snapshots
    g2.set_av("nstep_save", ts_tstream_type.int, 999999)

    # Save probes for last few period
    g2.set_av("nstep_save_start", ts_tstream_type.int, nstep_save_start)
    g2.set_av("nstep_save_start_probe", ts_tstream_type.int, nstep_save_start)

    # save probes every  nth step, from the beginning
    g2.set_av("nstep_save_probe", ts_tstream_type.int, nstep_save_probe)

    # other configuration variables
    g2.set_av("dts_conv", ts_tstream_type.float, 0.0)
    g2.set_av("facsafe", ts_tstream_type.float, 0.2)
    g2.set_av("dts", ts_tstream_type.int, 1)

    for bid in g2.get_block_ids():
        g2.set_bv("fmgrid", ts_tstream_type.float, bid, 0.0)

    # g2.set_av("sfin",ts_tstream_type.float,0.5)
    # g2.set_av("facsecin",ts_tstream_type.float,0.005)
    g2.set_av("dampin",ts_tstream_type.float,10.0)

    # use mixing lengths and flow guess from steady calculation
    g2.set_av("restart", ts_tstream_type.int, 1)
    g2.set_av("poisson_restart", ts_tstream_type.int, 1)
    g2.set_av("poisson_nstep", ts_tstream_type.int, 0)

    # load balance for 
    ts_tstream_load_balance.load_balance(g2, 4)

    # Reset spurious application variable
    g2.set_av("if_ale", ts_tstream_type.int, 0)

    # write out unsteady input file
    g2.write_hdf5(fname_out)

    print 'Old blade counts', nb
    print 'Sector size', sect, 'th annulus'
    print 'Scaled blade counts', dup, 'x',sect
    print 'Scaling factors', scale
    nb_new = (np.array(dup)*sect).astype(int)

    print 'Checking for Tyler/Sofrin modes...'
    mV = nb_new[0]*np.arange(1,7)
    nB = nb_new[1]*np.arange(1,7)
    if np.intersect1d(mV,nB):
        print('Warning: mode found')
        print(mV)
        print(nB)
    else:
        print('OK.')

