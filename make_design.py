"""
Choose parameters for a turbine stage and generate a corresponding CFD model.
"""
import design

fname = 'input_1.hdf5'
phi = 0.8  # Flow coefficient
psi = 1.6  # Stage loading coefficient
Lam = 0.5  # Degree of reaction
Ma = 0.7  # Vane exit Mach number
eta = 0.85  # Polytropy efficiency
gap_chord = 0.5  # Spacing between stator and rotor

design.generate(fname, phi, psi, Lam, Ma, eta, gap_chord )
