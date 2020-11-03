"""
Choose parameters for a turbine stage and generate a corresponding CFD model.
"""
import design

fname = 'input_1'  # Desired name of the TS input hdf5
phi = 0.8  # Flow coefficient (0.4 to 1.2)
psi = 1.6  # Stage loading coefficient (0.8 to 2.4)
Lam = 0.5  # Degree of reaction (0.4 to 0.6)
# Ma = 0.9  # Vane exit Mach number (0.6 to 0.9)
eta = 0.9  # Polytropic efficiency (leave this for now)
gap_chord = 0.5  # Spacing between stator and rotor

# Check that the code works for many Mach
for Mai in [0.6, 0.65, 0.7, 0.75, 0.81, 0.85, 0.9]:

    # Create a file name for this Mach using % substitution
    fname_now = fname + '_Ma_%.2f.hdf5' % Mai

    # Call out to the design generation code
    # It is complicated so best to think of it as a black box!
    design.generate(fname_now, phi, psi, Lam, Mai, eta, gap_chord )
