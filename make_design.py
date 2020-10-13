"""
Choose parameters for a turbine stage and generate a corresponding CFD model.
"""
import design

fname = 'input_1.hdf5'
phi = 0.8  # Flow coefficient (0.4 to 1.2)
psi = 1.6  # Stage loading coefficient (0.8 to 2.4)
Lam = 0.5  # Degree of reaction (0.4 to 0.6)
Ma = 0.7  # Vane exit Mach number (0.6 to 0.9)
eta = 0.9  # Polytropic efficiency (leave this for now)
gap_chord = 0.5  # Spacing between stator and rotor

design.generate(fname, phi, psi, Lam, Ma, eta, gap_chord )
