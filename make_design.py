"""
Choose parameters for a turbine stage and generate a corresponding CFD model.
"""
import design

fname = 'input_1.hdf5'
phi = 0.6  # Flow coefficient
psi = 2.0  # Stage loading coefficient
Lam = 0.5  # Degree of reaction
Ma = 0.8  # Vane exit Mach number
eta = 0.9  # Polytropic efficiency
gap_chord = 0.5  # Spacing between stator and rotor

design.generate(fname, phi, psi, Lam, Ma, eta, gap_chord )
