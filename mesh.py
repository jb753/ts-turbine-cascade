from ts import ts_tstream_grid, ts_tstream_type, ts_tstream_default
from ts import ts_tstream_patch_kind, ts_tstream_check_grid

if __name__ == "__main__":

    # Grid point numbers
    ni = [97, 97, 41]
    nj = 3
    nk = 49

    # Make stator and rotor blocks
    for bid in range(2):
        b = ts_tstream_type.TstreamBlock()
        b.bid = bid
        b.np = 0
        b.ni = np.sum(ni)
        b.nj = nj
        b.nk = nk
        b.procid = 0
        b.threadid = 0
        g.add_block(b)
        
        g.set_bp("x", ts_tstream_type.float, bid, x)
        g.set_bp("r", ts_tstream_type.float, bid, y)
        g.set_bp("rt", ts_tstream_type.float, bid, z)

