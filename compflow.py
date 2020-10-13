#
# Functions to calculate one-dimensional compressible flow quantities
#
import numpy as np
from scipy.optimize import newton


# Invert the Mach number relations by solving iteratively
def to_Ma(var, Y_in, ga, supersonic=False):
    #
    # Validate input data
    Y = np.asarray_chkfinite(Y_in)
    if ga <= 0.:
        raise ValueError('Specific heat ratio must be positive.')
    if np.any(Y < 0.):
        raise ValueError('Input quantity must be positive.')

    # Choose initial guess on sub or supersonic branch
    if supersonic or var in ['Posh_Po', 'Mash']:
        Ma_guess = np.array(1.5)
    else:
        Ma_guess = np.array(0.3)

    Ma_out = np.ones_like(Y)

    # Get indices for non-physical values
    if var == 'mcpTo_APo':
        ich = Y > from_Ma('mcpTo_APo', 1., ga, validate=False)
    elif var in ['Po_P', 'To_T', 'rhoo_rho', 'A_Acrit']:
        ich = Y < 1.
    elif var in ['Posh_Po', 'Mash']:
        ich = Y > 1.
    else:
        ich = np.full(np.shape(Y), False)

    # Return NaN if the value is not physical
    Ma_out[ich] = np.nan

    # Don't try to solve if all are non-physical
    if np.any(~ich):

        # Check if an explicit inversion exists
        if var == 'To_T':
            Ma_out[~ich] = np.sqrt((Y[~ich] - 1.) * 2. / (ga - 1.))

        elif var == 'Po_P':
            Ma_out[~ich] = np.sqrt((Y[~ich] ** ((ga - 1.) / ga) - 1.) * 2. / (ga - 1.))

        elif var == 'rhoo_rho':
            Ma_out[~ich] = np.sqrt((Y[~ich] ** (ga - 1.) - 1.) * 2. / (ga - 1.))

        else:

            # Wrapper functions for the iterative solve
            def err(Ma_i):
                return from_Ma(var, Ma_i, ga, validate=False) - Y[~ich]

            def jac(Ma_i):
                return derivative_from_Ma(var, Ma_i, ga, validate=False)

            # Newton iteration
            Ma_out[~ich] = newton(err, np.ones_like(Y[~ich]) * Ma_guess, fprime=jac)

    return Ma_out


# Quantities as explicit functions of Ma
def from_Ma(var, Ma_in, ga_in, validate=True):
    #
    # Validate Ma and ga
    if validate:
        ga = np.asarray_chkfinite(ga_in)
        Ma = np.asarray_chkfinite(Ma_in)
        if ga <= 0.:
            raise ValueError('Specific heat ratio must be positive.')
        if np.any(Ma < 0.):
            raise ValueError('Mach number must be positive.')
        Malimsh = 1.  # Physical limit on pre-shock Ma
    else:
        ga = np.asarray(ga_in)
        Ma = np.asarray(Ma_in)
        Malimsh = np.sqrt((ga - 1.) / ga / 2.) + 0.001  # Mathematical limit on pre-shock Ma

    # Define shorthand gamma combinations
    gm1 = ga - 1.
    gp1 = ga + 1.
    gm1_2 = gm1 / 2.
    gp1_2 = gp1 / 2.
    g_gm1 = ga / gm1
    sqr_gm1 = np.sqrt(gm1)
    gp1_gm1 = (ga + 1.) / gm1

    # Stagnation temperature ratio appears in every expression
    To_T = 1. + gm1_2 * Ma ** 2.

    # Safe reciprocal of Ma
    with np.errstate(divide='ignore'):
        recip_Ma = 1. / Ma

    # Simple ratios
    if var == 'To_T':
        return To_T

    elif var == 'Po_P':
        return To_T ** g_gm1

    elif var == 'rhoo_rho':
        return To_T ** (1. / gm1)

    # Velocity and mass flow functions
    elif var == 'V_cpTo':
        return sqr_gm1 * Ma * To_T ** -0.5

    elif var == 'mcpTo_APo':
        return ga / sqr_gm1 * Ma * To_T ** (-0.5 * gp1_gm1)

    elif var == 'mcpTo_AP':
        return ga / sqr_gm1 * Ma * To_T ** 0.5

    # Choking area
    elif var == 'A_Acrit':
        return recip_Ma * (2. / gp1 * To_T) ** (0.5 * gp1_gm1)

    # Post-shock Mach
    elif var == 'Mash':
        Mash = np.asarray(np.ones_like(Ma) * np.nan)
        Mash[Ma >= Malimsh] = (To_T[Ma >= Malimsh] / (ga * Ma[Ma >= Malimsh] ** 2. - gm1_2)) ** 0.5
        return Mash

    # Shock pressure ratio
    elif var == 'Posh_Po':
        Posh_Po = np.asarray(np.ones_like(Ma) * np.nan)
        A = gp1_2 * Ma ** 2. / To_T
        B = 2. * ga / gp1 * Ma ** 2. - 1. / gp1_gm1
        Posh_Po[Ma >= Malimsh] = A[Ma >= Malimsh] ** g_gm1 * B[Ma >= Malimsh] ** (-1. / gm1)
        return Posh_Po

    # Throw an error if we don't recognise the requested variable
    else:
        raise ValueError('Invalid quantity requested: {}.'.format(var))


# Quantity derivatives as explict functions of Ma
def derivative_from_Ma(var, Ma_in, ga_in, validate=True):
    #
    # Validate Ma and ga
    if validate:
        ga = np.asarray_chkfinite(ga_in)
        Ma = np.asarray_chkfinite(Ma_in)
        if ga <= 0.:
            raise ValueError('Specific heat ratio must be positive.')
        if np.any(Ma < 0.):
            raise ValueError('Mach number must be positive.')
    else:
        ga = np.asarray(ga_in)
        Ma = np.asarray(Ma_in)

    # Define shorthand gamma combinations
    gm1 = ga - 1.
    gp1 = ga + 1.
    gm1_2 = gm1 / 2.
    g_gm1 = ga / gm1
    sqr_gm1 = np.sqrt(gm1)
    gp1_gm1 = (ga + 1.) / gm1
    Malimsh = np.sqrt(0.5 / g_gm1) + 0.001  # Limit when denominator goes negative

    # Stagnation temperature ratio appears in every expression
    To_T = 1. + gm1_2 * Ma ** 2.

    # Safe reciprocal of Ma
    with np.errstate(divide='ignore'):
        recip_Ma = 1. / Ma

    # Simple ratios
    if var == 'To_T':
        return gm1 * Ma

    elif var == 'Po_P':
        return ga * Ma * To_T ** (g_gm1 - 1.)

    elif var == 'rhoo_rho':
        return Ma * To_T ** (1. / gm1 - 1.)

    # Velocity and mass flow functions
    elif var == 'V_cpTo':
        return sqr_gm1 * (To_T ** -0.5 - 0.5 * gm1 * Ma ** 2. * To_T ** -1.5)

    elif var == 'mcpTo_APo':
        return ga / sqr_gm1 * (To_T ** (-0.5 * gp1_gm1) - 0.5 * gp1 * Ma ** 2. * To_T ** (-0.5 * gp1_gm1 - 1.))

    elif var == 'mcpTo_AP':
        return ga / sqr_gm1 * (To_T ** 0.5 + 0.5 * gm1 * Ma ** 2. * To_T ** -0.5)

    # Choking area
    elif var == 'A_Acrit':
        return (2. / gp1 * To_T) ** (0.5 * gp1_gm1) * (-recip_Ma ** 2. + 0.5 * gp1 * To_T ** -1.)

    # Post-shock Mack number
    elif var == 'Mash':
        der_Mash = np.asarray(np.ones_like(Ma) * np.nan)
        A = gp1 ** 2. * Ma / np.sqrt(2.)
        C = ga * (2 * Ma ** 2. - 1.) + 1
        der_Mash[Ma >= Malimsh] = -A[Ma >= Malimsh] * To_T[Ma >= Malimsh] ** -.5 * C[Ma >= Malimsh] ** -1.5
        return der_Mash

    # Shock pressure ratio
    elif var == 'Posh_Po':
        der_Posh_Po = np.asarray(np.ones_like(Ma) * np.nan)
        A = ga * Ma * (Ma ** 2. - 1.) ** 2. / To_T ** 2.
        B = gp1 * Ma ** 2. / To_T / 2.
        C = 2. * ga / gp1 * Ma ** 2. - 1. / gp1_gm1
        der_Posh_Po[Ma >= Malimsh] = -A[Ma >= Malimsh] * B[Ma >= Malimsh] ** (1. / gm1) * C[Ma >= Malimsh] ** -g_gm1
        return der_Posh_Po

    # Throw an error if we don't recognise the requested variable
    else:
        raise ValueError('Invalid quantity requested: {}.'.format(var))
