This repository contains code to generate a 2D Turbostream CFD model for a set
of turbine stage design parameters.

Usage instructions:

1. Log in to the HPC and obtain an interactive node on the GPU cluster,
```
sintr_0 -p pascal -t 1:00:00 -N 1 -A pullan-mhi-sl2-gpu --gres=gpu:1 --qos=intr 
```

2. Set up the Turbostream environment,
```
source /usr/local/software/turbostream/ts3610/bashrc_module_ts3610
```

3. Edit `make_design.py` to choose your turbine aerodynamic design, and generate the input file by running,
```
python make_design.py
```

4. Run Turbostream on the input file (this will print lots of numbers to the
   screen as the calculation progresses, and save an output file at the end),
```
mpirun -npernode 1 -np 1 turbostream input_1.hdf5 output_1 1
```
You can capture the output and save it to a file by using,
```
mpirun -npernode 1 -np 1 turbostream input_1.hdf5 output_1 1 > log_1.txt &
```
and view it in the terminal using `less log_1.txt`.

5. Create plots of interest by editing `plot.py` and running,
```
python plot.py
```

6. To create an unsteady CFD from the converged steady solution model, use
```
python convert_unsteady.py
```

7. Run Turbostream on the unsteady input file,
```
mpirun -npernode 1 -np 1 turbostream input_2.hdf5 output_2 1 > log_2.txt &
```

8. Turbostream writes out a probe hdf5 file every time step, but we do not need them, so use this command to clear them up,
```
./clean_probes.sh
```

9. Finally, read the unsteady data and plot using,
```
python plot_unsteady.py
```

JB Oct 2020
