"""
This file contains methods for the one-dimensional design of turbine stages.
"""
import scipy.optimize
import compflow as cf
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

# Define a namedtuple to store all information about a vane design
VaneParams = [
    'Ma_in',  # Relative Mach numbers at stator, rotor inlet
    'Ma_out',  # Relative Mach numbers at stator, rotor outlet
    'Yp',  # Stagnation P loss coefficients for stator, rotor
    'chi',  # Metal angles for stator in, out, rotor in, out
    'Ax_Axin',  # Normalised areas
    'PR',  # Stagnation pressure ratio
    'Vx_rat',  # Axial velocity ratios w.r.t. rotor inlet
    'rm_rat',  # Mean radius ratios w.r.t. rotor inlet
]
NonDimVane = namedtuple('NonDimVane', VaneParams)

# Define a namedtuple to store all information about a stage design
# Note that the reference blade speed is taken at rotor inlet
StageParams = [
    'phi',  # Flow coefficient
    'psi',  # Stage loading coefficient
    'Ma_in',  # Relative Mach numbers at stator, rotor inlet
    'Ma_out',  # Relative Mach numbers at stator, rotor outlet
    'Yp',  # Stagnation P loss coefficients for stator, rotor
    'Al',  # Absolute flow angles
    'Al_rel',  # Relative frame flow angles
    'Lam',  # Reaction
    'chi',  # Metal angles for stator in, out, rotor in, out
    'Ax_Axin',  # Normalised areas
    'U_cpToin',  # Non-dimensional blade speed
    'PR',  # Stagnation pressure ratio
    'TR',  # Stagnation temperature ratio
    'eta',  # Polytropic efficiency
    'Vx_rat',  # Axial velocity ratios w.r.t. rotor inlet
    'rm_rat',  # Mean radius ratios w.r.t. rotor inlet
]
NonDimStage = namedtuple('NonDimStage', StageParams)

def nondim_vane(*, Al, Ma_out, ga, Yp, rr, drr=None, const_Vx=None):
    """Generate a non-dimensional vane design."""

    if drr is None and const_Vx is None:
        raise ValueError("Need to specify span or Vx is const.")

    elif const_Vx:
        """Adjust span to produce an N-D vane with Vx=constant."""
        def iter_drr(x):
            vane_now = nondim_vane(Al=Al, Ma_out=Ma_out, ga=ga, Yp=Yp, rr=rr,
                                drr=x, const_Vx=False)
            return vane_now.Vx_rat - 1.
        res = scipy.optimize.root_scalar(iter_drr, x0=1., x1=1.1)
        if not res.converged:
            raise Exception('Root not found')
        vane_out = nondim_vane(Al=Al, Ma_out=Ma_out, ga=ga, Yp=Yp, rr=rr,
                                drr=res.root, const_Vx=False)
        return vane_out

    # Geometry
    dr_dr_in = np.array([1., drr])
    cosAl = np.cos(np.radians(Al))

    # With known Ma there is an algebraic equation for Po_out/Po_in
    Po_P_out = cf.from_Ma('Po_P', Ma_out, ga)
    Por = (1. - Yp) / (1. - Yp / Po_P_out )

    # We know what exit capacity should be
    # Use it to work out inlet Ma if lossless
    Q_out = cf.from_Ma('mcpTo_APo', Ma_out, ga)
    Q_in = Q_out * rr * drr * cosAl[1] / cosAl[0] * Por
    Ma_in = cf.to_Ma('mcpTo_APo', Q_in, ga)
    V_cpTo = cf.from_Ma('V_cpTo', [Ma_in, Ma_out], ga)

    # Get axial velocity ratios
    Vx_rat = V_cpTo[1] / V_cpTo[0] * cosAl[1] / cosAl[0]

    # Assemble all of the data into the output object
    dat = NonDimVane(
        Ma_in=float(Ma_in),
        Ma_out=Ma_out,
        Yp=Yp,
        chi=Al,
        Ax_Axin=dr_dr_in,
        PR=Por,
        Vx_rat=Vx_rat,
        rm_rat=rr
    )
    return dat

def nondim_vane_const_Vx(*, Al, Ma_out, ga, Yp, rr):
    """Adjust span to produce an N-D vane with Vx=constant."""
    def iter_drr(x):
        vane_now = nondim_vane(Al=Al, Ma_out=Ma_out, ga=ga, Yp=Yp, rr=rr,
                               drr=x)
        return vane_now.Vx_rat - 1.
    res = scipy.optimize.root_scalar(iter_drr, x0=1., x1=1.1)
    if not res.converged:
        raise Exception('Root not found')
    vane_out = nondim_vane(Al=Al, Ma_out=Ma_out, ga=ga, Yp=Yp, rr=rr,
                            drr=res.root)
    return vane_out

def nondim_stage_from_Al(*,
                         phi, psi, Al,  # Velocity triangles
                         Ma, ga,  # Compressibility
                         eta,  # Loss
                         Vx_rat=(1., 1.),  # Design philosophy: is Vx const?
                         limit_Ma=False
                         ):
    """Get N-D geometry, blade speed for set of N-D aerodynamic parameters.

    This compact routine is the only stage design code you will ever need. It
    uses Euler's equation, definitions of the N-D parameters, and standard
    perfect gas compressible flow relations.

    Assumes constant mean radius, angular velocity, hence blade speed.

    From the output of this function, arbitrarily choosing one of Omega or rm
    and providing an inlet stagnation state will completely define the stage in
    dimensional terms.

    TODO - allow for varying the mean radius."""

    # Unpack input
    Ala = Al[0]
    Alc = Al[1]

    # First, construct velocity triangles

    # Euler work equation sets tangential velocity upstream of rotor
    # This gives us absolute flow angles everywhere
    tanAlb = np.tan(np.radians(Alc)) * Vx_rat[1] + psi / phi
    Alb = np.degrees(np.arctan(tanAlb))
    Al_all = np.hstack((Ala, Alb, Alc))

    # Get non-dimensional velocities
    Vx_U = np.array([Vx_rat[0], 1., Vx_rat[1]]) * phi
    Vt_U = Vx_U * np.tan(np.radians(Al_all))
    Vtrel_U = Vt_U - 1.
    V_U = np.sqrt(Vx_U**2. + Vt_U**2.)
    Vrel_U = np.sqrt(Vx_U**2. + Vtrel_U**2.)
    Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Use Mach number to get U/cpTo
    V_cpTob = cf.from_Ma('V_cpTo', Ma, ga)
    U_cpToin = V_cpTob / V_U[1]

    cpToin_Usq = 1./U_cpToin**2
    cpToout_Usq = cpToin_Usq - psi

    # N-D temperatures from input blade Ma and psi
    cpTo_Usq = np.array([cpToin_Usq, cpToin_Usq, cpToout_Usq])

    # Ma nos 
    Ma_all = cf.to_Ma('V_cpTo', V_U / np.sqrt(cpTo_Usq), ga)
    Ma_rel = Ma_all * Vrel_U / V_U

    # If blade exit mach is supersonic, bump mach down by 
    if limit_Ma and (Ma_rel[-1] > 0.99):
        print('Dropping Ma')
        return nondim_stage_from_Al(phi=phi, psi=psi, Al=Al,
                                   Ma=Ma*0.995, ga=ga, eta=eta,
                                   Vx_rat=Vx_rat,limit_Ma=limit_Ma)

    # capacity everywhere
    Q = cf.from_Ma('mcpTo_APo', Ma_all, ga)
    Q_Qin = Q / Q[0]

    # Second, construct annulus line

    # Use polytropic effy to get entropy change
    To_Toin = cpTo_Usq / cpTo_Usq[0]

    # Split evenly between rotor and stator
    Ds_cp = -(1. - 1. / eta) * np.log(To_Toin[-1])
    s_cp = np.hstack((0., 0.5, 1.)) * Ds_cp

    # Convert to stagnation pressures
    Po_Poin = np.exp((ga / (ga - 1.)) * (np.log(To_Toin) + s_cp))

    # Area ratios = span ratios because rm = const
    cosAl = np.cos(np.radians(Al_all))
    Dr_Drin = np.sqrt(To_Toin) / Po_Poin / Q_Qin * cosAl[0] / cosAl

    # Evaluate some other parameters that we might want to target
    T_Toin = To_Toin / cf.from_Ma('To_T', Ma_all, ga)
    P_Poin = Po_Poin / cf.from_Ma('Po_P', Ma_all, ga)
    Porel_Poin = P_Poin * cf.from_Ma('Po_P', Ma_rel, ga)
    Lam = (T_Toin[2] - T_Toin[1]) / (T_Toin[2] - T_Toin[0])

    Yp_vane = (Po_Poin[0] - Po_Poin[1]) / (Po_Poin[0] - P_Poin[1])
    Yp_blade = (Porel_Poin[1] - Porel_Poin[2]) / (Porel_Poin[1] - P_Poin[2])
    PR = Po_Poin[2] / Po_Poin[0]
    TR = To_Toin[2] / To_Toin[0]

    chi = np.hstack((Al_all[:2], Alrel[1:]))

    # Assemble all of the data into the output object
    dat = NonDimStage(
        phi=phi,
        psi=psi,
        Ma_in=np.array([Ma_all[0], Ma_rel[1]]),
        Ma_out=np.array([Ma_all[1], Ma_rel[2]]),
        Yp=np.array([Yp_vane, Yp_blade]),
        Al=Al_all,
        Al_rel=Alrel,
        Lam=Lam,
        chi=chi,
        Ax_Axin=Dr_Drin,
        U_cpToin=U_cpToin,
        PR=PR,
        TR=TR,
        eta=eta,
        Vx_rat=np.array(Vx_rat),
        rm_rat=np.ones((2,))
    )

    return dat

def nondim_stage_from_Lam(*,
                          phi, psi, Lam, Alin,  # Velocity triangles
                          Ma, ga,  # Compressibility
                          eta,  # Loss
                          Vx_rat=(1., 1.),  # Design philosophy: is Vx const?
                          limit_Ma=False
                          ):
    """This function produces an N-D stage design at fixed reaction.

    It is most convenient to evaluate a design in terms of exit flow angle, and
    not reaction. So iterate on the exit flow angle until the target reaction
    is achieved."""

    # Inner function
    def iter_Al(x):
        stg_now = nondim_stage_from_Al(phi=phi, psi=psi, Al=[Alin, x],
                                       Ma=Ma, ga=ga, eta=eta,
                                       Vx_rat=Vx_rat)
        return stg_now.Lam - Lam

    # First pass map out a coarse reaction graph
    Al_guess = np.linspace(-89.,89.0,20)
    with np.errstate(invalid='ignore'):
        Lam_guess = np.array([iter_Al(Ali) for Ali in Al_guess])
    Al_guess = Al_guess[~np.isnan(Lam_guess)]
    Lam_guess = Lam_guess[~np.isnan(Lam_guess)]

    # f,a = plt.subplots()
    # a.plot(Al_guess,Lam_guess)
    # plt.show()

    Al1 = Al_guess[np.argmax(Lam_guess)]
    Al2 = Al_guess[np.argmin(Lam_guess)]

    res = scipy.optimize.root_scalar(iter_Al,bracket=(Al1,Al2),xtol=1e-10)
    if not res.converged:
        raise Exception('Root not found')

    stg_out = nondim_stage_from_Al(phi=phi, psi=psi, Al=[Alin, res.root],
                                   Ma=Ma, ga=ga, eta=eta,
                                   Vx_rat=Vx_rat,limit_Ma=limit_Ma)
    if np.any(stg_out.Ma_out>1.):
        raise Exception('Choked design')

    return stg_out

def nondim_stage_const_span(*,
                          phi, psi, Lam, Alin,  # Velocity triangles
                          Ma, ga,  # Compressibility
                          eta,  # Loss
                          ):
    """Generate non-dimensional stage with constant axial velocity."""
    def iter_Vx(x):
        stg_now = nondim_stage_from_Lam(phi=phi, psi=psi, Lam=Lam, Alin=Alin,
                                       Ma=Ma, ga=ga, eta=eta,
                                       Vx_rat=x)
        return np.sum((stg_now.Ax_Axin - 1.)**2.)
    res = scipy.optimize.minimize(iter_Vx, x0=np.ones((2,)))
    if not res.success:
        raise Exception('Root not found')

    stg_out = nondim_stage_from_Lam(phi=phi, psi=psi, Lam=Lam, Alin=Alin,
                                   Ma=Ma, ga=ga, eta=eta,
                                   Vx_rat=res.x)
    return stg_out

def nondim_stage(*, phi, psi, Lam, Alin, Ma, ga, eta, const_Vx):
    if const_Vx:
        return nondim_stage_from_Lam(
            phi=phi, psi=psi, Lam=Lam, Alin=Alin, Ma=Ma, ga=ga, eta=eta)
    else:
        return nondim_stage_const_span(
            phi=phi, psi=psi, Lam=Lam, Alin=Alin, Ma=Ma, ga=ga, eta=eta)

def scale_vane(nd_vane, inlet, htr):
    """Scale an N-D vane to a particular dimensional condition."""

    # Pick mean radius arbitrarily
    rm = np.ones((2,))

    # Use hub-to-tip ratio to set span (mdot will therefore float)
    Dr_rm = 2. * (1. - htr) / (1. + htr)
    Dr = rm * Dr_rm * nd_vane.Ax_Axin[:2]
    chi = nd_vane.chi[:2]

    return rm, Dr, chi

def scale_stage(nd_stage, inlet, Omega, htr):
    """Scale an N-D stage to a particular dimensional condition."""

    # Use N-D blade speed to get U, hence mean radius
    U = nd_stage.U_cpToin * np.sqrt(inlet.To * inlet.cp)
    rm = U / Omega * np.ones((4,))

    # Use hub-to-tip ratio to set span (mdot will therefore float)
    Dr_rm = 2. * (1. - htr) / (1. + htr)
    Dr = rm * Dr_rm * np.hstack((nd_stage.Ax_Axin[:2], nd_stage.Ax_Axin[1:]))

    return rm, Dr, nd_stage.chi, Omega

def calc_Yp(states):
    """For a pair of inlet and exit states, determine the loss coefficient."""
    return (states[0].Po - states[1].Po) / (states[0].Po - states[1].P)

class State:
    """A set of flow quantities at a particular location"""

    def __init__(self, ga, rgas, Po, To, Al=0.0, Ma=0.0):
        """Initialise a new state from a gas model, two thermodynamic
        properties, and optional flow angle and Mach number."""

        # Save input data
        self.ga = ga
        self.rgas = rgas
        self.Po = Po
        self.To = To
        self.Ma = Ma
        self.Al = Al

        # Gas properties
        self.gam1 = self.ga - 1.0
        self.gap1 = self.ga + 1.0
        self.gae = self.ga / self.gam1
        self.cp = self.rgas * self.gae
        self.cv = self.cp / self.ga

        # Secondary vars
        self.secondary()

    def secondary(self):
        """Calculate all secondary variables from the basic ones."""

        # Resolve Mach
        self.Max = self.Ma * np.cos(np.radians(self.Al))
        self.Mat = self.Ma * np.sin(np.radians(self.Al))

        # Get velocities
        V_cpTo = cf.from_Ma('V_cpTo', self.Ma, self.ga)
        self.V = V_cpTo * np.sqrt(self.cp * self.To)
        self.Vx = self.V * np.cos(np.radians(self.Al))
        self.Vt = self.V * np.sin(np.radians(self.Al))

        # Get static quantities
        self.P = self.Po / cf.from_Ma('Po_P', self.Ma, self.ga)
        self.T = self.To / cf.from_Ma('To_T', self.Ma, self.ga)

        # Other secondary vars
        self.a = np.sqrt(self.ga * self.rgas * self.T)
        self.rho = self.P / self.rgas / self.T

    def __str__(self):
        """If we print this object, show the basic variables."""

        return ("To = %f, Po = %f, Al = %f, Ma = %f\nVx = %f, Vt = %f"
                % (self.To, self.Po, self.Al, self.Ma, self.Vx, self.Vt))

    def change_frame(self, Delta_U):
        """Return a new state in a reference frame with different U.

        The two blade speeds are related by:
            Unew = Uold + DeltaU
        So that DeltaU is +ve when going from stator to rotor, and
        -ve when going from rotor to stator."""

        # Velocity relative to new reference frame
        Vt_rel = self.Vt - Delta_U

        # Get relative Mach
        V_rel = np.sqrt(Vt_rel**2. + self.Vx**2.)
        Ma_rel = V_rel / self.a

        # New Po, To in relative frame
        Po_rel = self.P * cf.from_Ma('Po_P', Ma_rel, self.ga)
        To_rel = self.T * cf.from_Ma('To_T', Ma_rel, self.ga)

        # New flow angle in relative frame
        Al_rel = np.degrees(np.arctan2(Vt_rel, self.Vx))

        return State(self.ga, self.rgas, Po_rel, To_rel, Al_rel, Ma_rel)

    def update(self, Po=None, To=None, Ma=None, Al=None):
        """Update any of the basic variables in place (no new object)."""

        if Po is not None:
            self.Po = Po

        if Al is not None:
            self.Al = Al

        if To is not None:
            self.To = To

        if Ma is not None:
            self.Ma = Ma

        self.secondary()


class Duct:
    """Representation of an unbladed annulus."""

    def __init__(self, rm, Delta_r):
        """Initialise with pure geometry."""

        # Save input
        self.rm = np.asarray(rm)
        self.Delta_r = np.asarray(Delta_r)

        # Assume zero loss unless modified later
        self.Delta_Po = 0.

        # Assume no cooling
        self.TRc = 0.
        self.PRc = 0.
        self.fc = 0.

        # Ducts never rotate
        self.Omega = 0.

    def get_Ax(self):
        """Reset the axial flow area, after geometry modified."""
        return self.rm * 2. * np.pi * self.Delta_r

    def get_To_out(self, inlet):
        """Use rothalpy and prescribed cooling to get outlet To"""

        # Conserve rothalpy to get To uncooled
        To_uc = (inlet.To
                 + (self.rm[1] ** 2 - self.rm[0] ** 2)
                 * (self.Omega ** 2.0) / 2.0 / inlet.cp)

        # Inject cooling flow at trailing edge
        To_out = To_uc * (1. + self.fc * self.TRc) / (1. + self.fc)

        return To_out

    def get_Po_out(self, inlet):
        """Use appropriate ideal pressure, prescribed loss to get outlet Po."""

        # Ideal stagnation pressure - Young and Horlock Wp
        if self.fc > 0.:
            Pos = inlet.Po * self.PRc ** (self.fc / (self.fc + 1.))
        else:
            Pos = inlet.Po * (self.get_To_out(inlet) / inlet.To) ** inlet.gae

        # Subtract prescribed pressure loss
        Po_out = Pos - self.Delta_Po

        return Po_out

    def set_loss(self, inlet, Yp):
        """Iterate the delta Po until a loss coefficent is reached."""

        # Use the guess dynamic head to iterate until exit Po converged
        DPo_old = self.Delta_Po
        err = np.inf
        tol = inlet.Po * 1e-9
        while err > tol:
            # Get outlet state
            outlet = self.transfer(inlet)

            # Use dynamic head to set Po loss for known Yp
            self.Delta_Po = Yp * (inlet.Po - outlet.P)

            # Calc error and store
            err = np.abs(DPo_old - self.Delta_Po)
            DPo_old = self.Delta_Po

        return outlet

    def transfer(self, inlet):
        """Conserve angular momentum to get outlet state."""

        Po_out = self.get_Po_out(inlet)
        To_out = self.get_To_out(inlet)

        # Conservation of angular momentum sets tangential velocity
        Vt_out = inlet.Vt * self.rm[0] / self.rm[1]

        # Get inlet mass flow using known inlet conditions
        Ax = self.get_Ax()
        A_in = Ax[0] * np.cos(np.radians(inlet.Al))
        capacity_in = cf.from_Ma('mcpTo_APo', inlet.Ma, inlet.ga)
        mdot_in = (capacity_in * A_in * inlet.Po
                   / np.sqrt(inlet.cp * inlet.To))

        # Addition of mass due to cooling
        mdot_out = mdot_in * (1. + self.fc)

        # Guess Al_out = Al_in to estimate Ma_out
        Al_out = inlet.Al
        err = np.inf
        tol = 1e-9
        while err > tol:
            # Get outlet capacity hence estimate of Ma_out
            Q_out = (mdot_out * np.sqrt(inlet.cp * To_out)
                     / Ax[1] / np.cos(np.radians(Al_out)) / Po_out)
            Ma_out = cf.to_Ma('mcpTo_APo', Q_out, inlet.ga,use_lookup=False)

            # Get velocity and new guess for flow angle
            V_out = (cf.from_Ma('V_cpTo', Ma_out, inlet.ga)
                     * np.sqrt(inlet.cp * To_out))
            Al_out_new = np.degrees(np.arcsin(Vt_out / V_out))

            # Calculate error
            err = np.abs(Al_out - Al_out_new)
            Al_out = Al_out_new

        return State(inlet.ga, inlet.rgas,
                     Po_out, To_out, Al_out, Ma_out)

    def get_mdot_in(self, inlet):
        """Return the mass flow resulting from an inlet state."""
        Q_in = cf.from_Ma('mcpTo_APo', inlet.Ma, inlet.ga)
        Ax_in = self.rm[0] * 2. * np.pi * self.Delta_r[0]
        mdot = (Q_in / np.sqrt(inlet.cp * inlet.To)
                * Ax_in * np.cos(np.radians(inlet.Al)) * inlet.Po)
        return mdot


class Row(Duct):
    """Representation of a bladed annulus."""

    def __init__(self, rm, Delta_r, chi, Omega=0.):
        """Initialise a row with geometry and rotational speed."""

        # Call parent init
        super().__init__(rm, Delta_r)

        # Add flow angles and angular velocity
        self.chi = np.asarray(chi)
        self.Omega = Omega

        # Default to no deviation
        self.dev = 0.

    def transfer(self, inlet):
        """Use prescribed exit flow angle to get outlet state.
        Overrides the Duct transfer method which only knows about conservation
        of angular momentum."""

        # Exit angle = metal + deviation
        Al_out = self.chi[1] + self.dev

        # Areas
        Ax = self.get_Ax()
        A_in = Ax[0] * np.cos(np.radians(inlet.Al))
        A_out = Ax[1] * np.cos(np.radians(Al_out))

        # Exit total properties
        Po_out = self.get_Po_out(inlet)
        To_out = self.get_To_out(inlet)

        # Evaluate outlet Mach
        capacity_in = cf.from_Ma('mcpTo_APo', inlet.Ma, inlet.ga)
        mdot_out = (capacity_in * A_in * inlet.Po
                    / np.sqrt(inlet.cp * inlet.To)) * (1. + self.fc)
        capacity_out = (mdot_out * np.sqrt(inlet.cp * To_out)
                        / A_out / Po_out)
        Ma_out = cf.to_Ma('mcpTo_APo', capacity_out, inlet.ga)

        return State(inlet.ga, inlet.rgas,
                     Po_out, To_out, Al_out, Ma_out)

    @classmethod
    def from_nondim(cls, Ma, Al, ga, Yp, rr, htr):
        """Generate a stator using non-dimensional parameters.

        This function simply transforms the input non-dimensional variables
        into appropriate dimensional variables to initialise the class. Note
        that arbitrary values for some dimensional quantites are chosen, but
        according to dimensional analysis their particular values do not
        matter."""

        # Choose arbitrary values
        rm_in = 1.
        To_in = 1700.
        Po_in = 20e5
        rgas = 287.058

        cp = rgas * ga / (ga - 1.)
        rm = rm_in * np.array([1., rr])

        # Use inlet area and Ma to set mass flow
        dr_in = rm_in * 2. * (1. - htr) / (1. + htr)
        Ax_in = rm_in * 2. * np.pi * dr_in
        Q_in = cf.from_Ma('mcpTo_APo', Ma[0], ga)
        mdot = (Q_in / np.sqrt(cp * To_in)
                * Ax_in * np.cos(np.radians(Al[0])) * Po_in)

        # Assume lossless to get first guess of exit span
        Q_out = cf.from_Ma('mcpTo_APo', Ma[1], ga)
        Ax_out = (mdot * np.sqrt(cp * To_in) / Po_in
                  / np.cos(np.radians(Al[1])) / Q_out)
        dr_out = Ax_out / (2. * np.pi * rm_in * rr)

        # Now generate the row structure
        dr = np.array([dr_in, dr_out])
        row = cls(rm, dr, Al)

        # Use the guess dynamic head to iterate until exit Po converged
        DPo_old = 0.
        err = np.inf
        tol = Po_in * 1e-9
        inlet = State(ga, rgas, Po_in, To_in, Al[0], Ma[0])
        while err > tol:
            # Get outlet state
            outlet = row.transfer(inlet)

            # Use dynamic head to set Po loss for known Yp
            row.Delta_Po = Yp * (inlet.Po - outlet.P)

            # Update span
            Ax_out = (mdot * np.sqrt(cp * To_in) / (Po_in - row.Delta_Po)
                      / np.cos(np.radians(Al[1])) / Q_out)
            row.Delta_r[1] = Ax_out / (2. * np.pi * rm[1])

            # Calc error and store
            err = np.abs(DPo_old - row.Delta_Po)
            DPo_old = row.Delta_Po

        return row, inlet

    @classmethod
    def from_nondim_2(cls, Ma_out, Al, ga, Yp, rr, htr, drr):
        """Generate a stator using non-dimensional parameters.

        This function simply transforms the input non-dimensional variables
        into appropriate dimensional variables to initialise the class. Note
        that arbitrary values for some dimensional quantites are chosen, but
        according to dimensional analysis their particular values do not
        matter."""

        # Choose arbitrary values
        rm_in = 1.0
        To_in = 1700.
        Po_in = 20e5
        rgas = 287.058

        # Deduce inlet span from hub-to-tip ratio
        dr_in = rm_in * 2. * (1. - htr) / (1. + htr)

        # Generate the object
        rm = rm_in * np.array([1., rr])
        dr = dr_in * np.array([1., drr])
        row = cls(rm, dr, Al)

        # We know what exit capacity should be
        # Use it to work out inlet Ma if lossless
        Q_out = cf.from_Ma('mcpTo_APo', Ma_out, ga)
        Q_in = (Q_out * rr * drr * np.cos(np.radians(Al[1])) /
                np.cos(np.radians(Al[0])))
        Ma_in = cf.to_Ma('mcpTo_APo', Q_in, ga)

        # Use the guess dynamic head to iterate until exit Po converged
        DPo_old = 0.
        err = np.inf
        tol = Po_in * 1e-9
        inlet = State(ga, rgas, Po_in, To_in, Al[0], Ma_in)
        while err > tol:
            # Get outlet state
            outlet = row.transfer(inlet)

            # Use dynamic head to set Po loss for known Yp
            row.Delta_Po = Yp * (inlet.Po - outlet.P)

            # Update inlet Ma accouting for loss
            Q_in = (Q_out * rr * drr * np.cos(np.radians(Al[1])) /
                    np.cos(np.radians(Al[0])) * (1. - row.Delta_Po / Po_in))
            Ma_in = cf.to_Ma('mcpTo_APo', Q_in, ga)
            inlet.update(Ma=Ma_in)

            # Calc error and store
            err = np.abs(DPo_old - row.Delta_Po)
            DPo_old = row.Delta_Po

        return row, inlet

    def to_tad(self, AR, inlet=None):
        """Format a dictionary of data to input to TAD."""

        rh = self.rm - self.Delta_r / 2.
        rc = self.rm + self.Delta_r / 2.
        typ = "STATOR" if self.Omega == 0. else "ROTOR"

        if inlet is not None:

            d = {
                "rh": rh[0],
                "rc": rc[0],
                "Po": inlet.Po,
                "To": inlet.To,
                "Alpha": inlet.Al
            }

        else:

            cx = np.mean(self.Delta_r) / AR

            d = {
                "xin": 0.,
                "xout": cx,
                "rh": rh[1],
                "rc": rc[1],
                "stag": self.chi,
                "Al1": self.chi[0],
                "Al2": self.chi[1],
                "type": typ,
            }

        return d

    def to_tad_machine(self, inlet, AR):

        d_vane = self.to_tad(AR)
        rows = {n: [d_vane[n]] for n in d_vane}
        states = (inlet, self.transfer(inlet))
        Yp_vane = (states[0].Po - states[1].Po) / (states[0].Po - states[1].P)
        rows['Ypmin'] = [-Yp_vane]

        mdot = self.get_mdot_in(inlet)
        rpm = self.Omega / 2. / np.pi * 60.

        # Get the inlet stuff
        d = {
            "title": "Rows",
            "nv": 2,
            "ga": inlet.ga,
            "mdot_des": mdot,
            "rpm_des": rpm,
            "inlet": self.to_tad(None, inlet),
            "row": rows
        }

        return d


class Stage:
    """Representation of a pair of Rows and a duct connecting them."""

    # Define bounds on iteration variables
    eps = 0.1
    bnd_Alc = (-90., 90.)
    bnd_phi = (eps, None)
    bnd_psi = (eps, None)
    bnd_DVx = (0.5, 1.5)

    def __init__(self, rm, Delta_r, chi, Omega):
        """Initialise using geometry at four stations and angular velocity."""

        # Make objects for the stator, inter-row duct, and rotor
        self.stator = Row(rm[:2], Delta_r[:2], chi[:2], 0.)
        self.duct = Duct(rm[1:3], Delta_r[1:3])
        self.rotor = Row(rm[2:], Delta_r[2:], chi[2:], Omega)

    def transfer(self, inlet):
        """Transfer the inlet state through the stage.
        Return the stator exit and rotor exit states in abs frame"""

        # Get blade velocities to change frames
        U = self.rotor.rm * self.rotor.Omega

        stator_exit = self.stator.transfer(inlet)
        rotor_inlet = self.duct.transfer(stator_exit).change_frame(U[0])
        rotor_exit = (self.rotor.transfer(rotor_inlet).change_frame(-U[1]))

        return inlet, stator_exit, rotor_inlet, rotor_exit

    def set_loss(self, inlet, Yp):
        """Iterate the delta Po until two loss coefficents are reached."""

        # Use the guess dynamic head to iterate until exit Po converged
        DPo_old = np.array([self.stator.Delta_Po, self.rotor.Delta_Po])
        err = np.inf
        tol = inlet.Po * 1e-9
        while err > tol:

            states = self.transfer(inlet)

            # Use dynamic head to set Po loss for known Yp
            DPo_now = np.array([Yp[0] * (states[0].Po - states[1].P),
                                Yp[1] * (states[2].Po - states[3].P)])
            self.stator.Delta_Po = DPo_now[0]
            self.rotor.Delta_Po = DPo_now[1]

            # Calc error and store
            err = np.max(np.abs(DPo_old - DPo_now))
            DPo_old = DPo_now

        return states

    def get_Uref(self):
        return self.rotor.rm[0] * self.rotor.Omega

    def to_tad(self, inlet, AR):
        """Format a dictionary of data for this stage to input to TAD."""

        mdot = self.stator.get_mdot_in(inlet)
        rpm = self.rotor.Omega / 2. / np.pi * 60.

        states = self.transfer(inlet)
        Yp_vane = calc_Yp(states[:2])
        Yp_blade = calc_Yp((states[2],
                            states[3].change_frame(self.get_Uref())))

        d_vane = self.stator.to_tad(AR[0])
        d_blade = self.rotor.to_tad(AR[1])
        cx = d_vane['xout']
        spacing = cx*1.0
        d_blade['xin'] = d_blade['xin'] + cx + spacing
        d_blade['xout'] = d_blade['xout'] + cx + spacing
        rows = {n: [d_vane[n], d_blade[n]] for n in d_vane}
        rows['Ypmin'] = [-Yp_vane, -Yp_blade]

        # Get the inlet stuff
        d = {
            "title": "Rows",
            "nv": 3,
            "ga": inlet.ga,
            "mdot_des": mdot,
            "rpm_des": rpm,
            "inlet": self.stator.to_tad(None, inlet),
            "row": rows
        }

        return d

def blade_section(cx, chi):
    """Makes a simple blade geometry from one-dimensional flow parameters."""
    # Copy defaults from MEANGEN (Denton)
    tkle     = 0.04 # LEADING EDGE THICKNESS/AXIAL CHORD.
    tkte     = 0.04 # TRAILING EDGE THICKNESS/AXIAL CHORD.
    tkmax   = 0.25 # MAXIMUM THICKNESS/AXIAL CHORD.
    xtkmax  = 0.40 # FRACTION OF AXIAL CHORD AT MAXIMUM THICKNESS 
    xmodle   = 0.02 # FRACTION OF AXIAL CHORD OVER WHICH THE LE IS MODIFIED.
    xmodte   = 0.01 # FRACTION OF AXIAL CHORD OVER WHICH THE TE IS MODIFIED.
    tk_typ   = 2.0  # FORM OF BLADE THICKNESS DISTRIBUTION.
    zweifel  = 0.85 # ZWEIFEL COEFFICIENT FOR TURBINES
    expo     = 1.0  # EXPONENT FOR TRANSFORMING THE AXIAL POSITION.
    xhat = np.linspace(0.,1.)
    x = 0.5*(1.-np.cos(np.pi * xhat))
    tanchi_lim = np.tan(np.radians(chi))
    tanchi = np.interp(x,(0.,1.),tanchi_lim)

    f,a = plt.subplots()
    a.plot(x,tanchi,'k-x')


if __name__ == "__main__":

    # Constants
    gamma = 1.4
    rgas = 287.14

    # Generate a stage in non-dimensional form
    nd_stage = nondim_stage_const_span(
            phi=0.8,
            psi=1.6,
            Lam=0.5,
            Alin=0.0,
            Ma=0.7,
            ga=gamma,
            eta=.9
            )

    # Choose dimensional conditions
    Omega = 50.0 * 2. * np.pi
    htr = 0.99
    Poin = 16e5
    Toin = 16e3
    Alin = 0.0
    inlet = State(gamma, rgas, Poin, Toin, Alin)

    # Get geometry
    rm, Dr, _, _ = scale_stage(nd_stage, inlet, Omega, htr)

	# Blade section
    blade_section(0.05, nd_stage.chi[:2])
    plt.show()