#!/bin/bash
# Delete the unwanted probe files
find -regex '.*_probe_[0-9]+.hdf5$' -delete
