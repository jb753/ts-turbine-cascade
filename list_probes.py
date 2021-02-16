#!/usr/bin/env python
"""This script reads a TS grid and lists all probe patches."""
from ts import ts_tstream_reader, ts_tstream_patch_kind  # TS grid reader
import sys

if __name__ == "__main__":

    # Load the grid 
    tsr = ts_tstream_reader.TstreamReader()
    fname = sys.argv[1]
    print('Selected file: %s ' % fname)
    g = tsr.read(fname)

    # Print probe patch locations
    print('Listing available probe patches...')
    for bid in g.get_block_ids():
        for pid in g.get_patch_ids(bid):
            patch = g.get_patch(bid,pid)
            if patch.kind == ts_tstream_patch_kind.probe:
                rpm = g.get_bv('rpm',bid)
                row_str = 'STATOR' if rpm==0 else 'ROTOR '
                di = patch.ien- patch.ist
                dj = patch.jen- patch.jst
                dk = patch.ken- patch.kst
                print('%s, bid=%3d, pid=%3d, di=%3d, dj=%3d, dk=%3d'
                        % (row_str, bid, pid, di, dj, dk))
