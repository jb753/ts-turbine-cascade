"""Definitions for the simple film cooling hole model."""
import compflow as cf  # Compressible flow relations
import numpy as np

def evaluate( Pinf_Poc, roVinf_Po_cpToc, Cd, ga ):
    """Evaluate blowing ratio using non-dimensional flow conditions.

    Coolant conditions are characterised by steady Poc, Toc. Main-stream
    conditions are characterised by instantaneous unsteady static pressure
    Pinf, and mass flux roVinf.

    Normalise main-stream mass flux by Poc/sqrt(cp Toc).

    """

    # Isentropic Mach number of coolant
    Macs = cf.to_Ma('Po_P', 1./Pinf_Poc, ga)

    # Isentropic flow function of coolant
    Qs = cf.from_Ma('mcpTo_APo', Macs, ga)

    # Blowing ratio
    BR = Cd * Qs / roVinf_Po_cpToc

    return BR


def normalise( Poc, Toc, Pinf, roinf, Vinf, cp ):
    """Convert dimensional flow conditions to non-dimensional ones.

    Parameters
    ----------
    Poc : float [Pa]
        Coolant stagnation pressure.
    Toc : float [K]
        Coolant stagnation temperature.
    Pinf : array_like [Pa]
        Main-stream static pressure.
    roinf : array_like [kg/m^3]
        Main-stream static density.
    Vinf : array_like [m/s]
        Main-stream velocity.
    cp : float [J/kgK]
        Specific heat capacity at constant pressure.

    Returns
    -------
    Pinf_Poc : array_like [-]
        Main-stream static pressure normalised by coolant stagnation pressure.
    roVinf_Po_cpToc : array_like [-]
        Main-stream mass flux normalised by Poc/sqrt(cpToc) .
    """

    return Pinf/Poc, roinf*Vinf/Poc*np.sqrt(cp * Toc)
