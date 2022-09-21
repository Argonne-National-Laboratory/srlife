# pylint: disable=wrong-import-position
"""
    Simple 1D heat balance thermohydraulics solver
"""

import numpy as np
import numpy.linalg as la
import scipy.interpolate as inter
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd


class StartLink:
    """Starting link in the chain, gives the first temperature"""

    def __init__(self, times, inlet_temperature, **kwargs):
        """
        Parameters:
            times (ntime,):                 discrete times
            inlet_temperature (ntime,):     inlet temperatures
        """
        self.times = times
        self.inlet_temperature = inlet_temperature

        self.inlet_ifn = inter.interp1d(self.times, self.inlet_temperature)

        # Fixed
        self.size = 1

    def T_inlet(self, t):
        """Calculate the inlet temperature at a given time

        Parameters:
            t:      time
        """
        return jnp.asarray(self.inlet_ifn(t))

    def residual(self, T_start, T_end, t):
        """Calculate the residual contribution

        Parameters:
            T_start:    inlet temperatures (null here)
            T_end:      outlet temperatures
            t:          time
        """
        return T_end - self.T_inlet(t)


class PanelLink:
    """Connection representing part of a panel"""

    def __init__(self, times, mass_flow, weights, **kwargs):
        """
        Parameters:
            times (ntime,):     discrete times
            mass_flow (ntime,): discrete flow rates
            weights (ntube,):   number of actual tube represented by each panel
        """
        self.times = times
        self.mass_flow = mass_flow
        self.weights = jnp.asarray(weights)

        self.mass_ifn = inter.interp1d(self.times, self.mass_flow)

    def mass_flow_rate(self, t):
        """
        Calculate the mass flow at a given time
        """
        return jnp.asarray(self.mass_ifn(t))

    @property
    def ntube(self):
        """Number of actual tubes in the panel"""
        return jnp.sum(self.weights)


class ManifoldLink(PanelLink):
    """Averages temperatures together at this node in the flow path"""

    def __init__(self, times, mass_flow, weights, **kwargs):
        """
        Args:
            times (ntime,):     times for input
            mass_flow (ntime,): mass flow rates
            weights (ntube,):   tube weights
        """
        super().__init__(times, mass_flow, weights, **kwargs)

        # Fixed again...
        self.size = 1

    def residual(self, T_in, T_out, t):
        """Residual equations

        Parameters:
            T_in:       inlet temperatures
            T_out:      outlet temperatures
            t:          time
        """
        return jnp.sum(self.weights * T_in) / self.ntube - T_out


class SimplePanelLink(PanelLink):
    """Very simple panel model for debugging"""

    def __init__(
        self, times, mass_flow, weights, ri, h, metal_temp, material, **kwargs
    ):
        """
        Parameters:
            times (ntime,):                 times for input
            mass_flow (ntime,):             panel mass flow rate
            weights (ntube,):               tube weights
            ri:                             tube inner radius
            h:                              tube height
            metal_temp (ntime,ntube,nt,nz): tube metal temperatures, fixed grid
            material:                       thermofluid model
        """
        super().__init__(times, mass_flow, weights, **kwargs)
        self.ri = ri
        self.h = h
        self.metal_temp = metal_temp
        self.material = material
        self.zs = np.linspace(0, self.h, self.metal_temp.shape[3])
        self.dtheta = 2.0 * np.pi / self.metal_temp.shape[2]
        self.dz = self.h / self.metal_temp.shape[3]

        self.metal_ifn = inter.interp1d(self.times, self.metal_temp, axis=0)

    @property
    def size(self):
        """Number of explicitly-represented tubes"""
        return len(self.weights)

    def residual(self, T_in, T_out, t):
        """Residual equations

        Args:
            T_in:       inlet temperatures
            T_out:      outlet temperatures
            t:          time
        """
        return self.Q_mass(T_in, T_out, t) - self.Q_conv(T_in, T_out, t)

    def mean_temperature(self, T_start, T_tube):
        """
        Returns the mean temperature of each tube
        """
        return (T_tube + T_start) / 2.0

    def metal_temperature(self, t):
        """
        Metal temperature field at a given time
        """
        return jnp.asarray(self.metal_ifn(t))

    def Q_mass(self, T_start, T_tube, t):
        """
        Heat transfered out of the panel due to mass flow

        Parameters:
            T_start (scalar):   temperature at start of panel
            T_tube (ntube,):    temperature at end of tubes
            t:                  time
        """
        mdot = self.mass_flow_rate(t)
        T_mean = self.mean_temperature(T_start, T_tube)
        cp = self.material.cp(T_mean)

        return self.weights * mdot / self.ntube * cp * (T_tube - T_start)

    def Q_conv(self, T_start, T_tube, t):
        """
        Heat transfered into the panel due to convection

        Parameters:
            T_start (scalar):   temperature at start of panel
            T_tube (ntube,):    temperature at end of tubes
            t:                  time
        """
        # Flow velocity
        u = self.flow_rates(T_start, T_tube, t)

        # Convective heat transfer coefficient
        # We could make this height dependent if we're using AD...
        T_mean = self.mean_temperature(T_start, T_tube)
        h = self.material.film_coefficient(T_mean, u, self.ri)

        # Integrate up the total contributions
        fluid_temps = self.fluid_temperatures(T_start, T_tube, t)
        T_metal = self.metal_temperature(t)

        flux = (
            self.weights[:, None, None]
            * h[:, None, None]
            * (T_metal - fluid_temps[:, None, :])
        )

        return self.ri * self.dz * self.dtheta * jnp.sum(flux, axis=(1, 2))

    def flow_rates(self, T_start, T_tube, t):
        """Recover the flow rates for a given state

        Args:
            T_start:    inlet temperatures
            T_tube:     outlet temperatures
            t:          time
        """
        mdot = self.mass_flow_rate(t)
        T_mean = self.mean_temperature(T_start, T_tube)
        rho = self.material.rho(T_mean)
        return mdot / (self.ntube * np.pi * rho * self.ri**2.0)

    def fluid_temperatures(self, T_start, T_tube, t):
        """Recover the fluid temperatures for a given state

        Args:
            T_start:    inlet temperatures
            T_tube:     outlet temperatures
            t:          time
        """
        return ((T_tube - T_start) / self.h * self.zs[:, None] + T_start).T


def deparameterize_flow_path(pset):
    """
    Convert parameter set to kwargs for FlowPath
    """
    return {
        "rtol": pset.get_default("rtol", 1e-6),
        "atol": pset.get_default("atol", 1e-8),
        "miter": pset.get_default("miter", 100),
        "verbose": pset.get_default("verbose", False),
    }


class FlowPath:
    """Flow path data structure

    A simplified description of a flow path as chain.
    """

    def __init__(
        self,
        times,
        mass_flow,
        inlet_temperature,
        rtol=1e-6,
        atol=1e-8,
        miter=50,
        verbose=False,
        **kwargs
    ):
        """
        Parameters:
            times (ntime,):             discrete times
            mass_flow (ntime,):         mass flow rate at each time
            inlet_temperature (ntime,): inlet temperature at each time
        """
        # Always start with the inlet node!
        self.times = times
        self.mass_flow = mass_flow
        self.inlet_temperature = inlet_temperature
        self.chain = [StartLink(times, inlet_temperature)]

        # Solver parameters
        self.rtol = 1e-6
        self.atol = 1e-8
        self.miter = 100
        self.verbose = verbose

    def add_panel(self, weights, ri, h, metal_temp, material):
        """
        Construct and add the standard panel -> manifold link

        Parameters:
            weights:        tube weights
            ri:             tube inner radius
            h:              tube height
            metal_temp:     metal temperatures
            material:       thermofluid material
        """
        self.chain.append(
            SimplePanelLink(
                self.times, self.mass_flow, weights, ri, h, metal_temp, material
            )
        )
        self.chain.append(ManifoldLink(self.times, self.mass_flow, weights))

    def add_panel_from_object(self, panel, material):
        """Same as add_panel, but use a panel object directly

        Args:
            panel (receiver.Panel): panel object
            material: fluid material properties
        """
        weights = np.array([float(tube.multiplier) for tube in panel.tubes.values()])
        # Fix this, we should take variable radii and heights
        ri = np.array([tube.r - tube.t for tube in panel.tubes.values()])[0]
        h = np.array([tube.h for tube in panel.tubes.values()])[0]

        metal_temps = []
        for tube in panel.tubes.values():
            if tube.abstraction == "3D":
                metal_temps.append(
                    tube.quadrature_results["ghost_temperature"][..., 1, 1:-1, 1:-1]
                )
            elif tube.abstraction == "2D":
                metal_temps.append(
                    np.broadcast_to(
                        tube.quadrature_results["ghost_temperature"][:, 1, 1:-1, None],
                        (tube.ntime, tube.nt, tube.nz),
                    )
                )
            elif tube.abstraction == "1D":
                metal_temps.append(
                    np.broadcast_to(
                        tube.quadrature_results["ghost_temperature"][:, 1, None, None],
                        (tube.ntime, tube.nt, tube.nz),
                    )
                )
            else:
                raise ValueError("Need to add")

        metal_temps = np.swapaxes(np.array(metal_temps), 0, 1)

        self.add_panel(weights, ri, h, metal_temps, material)

    def _setup(self):
        """
        Setup the data structures needed to solve the current chain
        """
        self.nvals = 0
        self.dof_map = []
        for obj in self.chain:
            self.dof_map.append(list(range(self.nvals, self.nvals + obj.size)))
            self.nvals += obj.size

    def solve(self, t):
        """
        Solve for the current fluid temperatures

        Args:
            t:      time
        """
        self._setup()
        self.t = t

        # Significant decision...
        T = np.zeros((self.nvals))

        # Initial residual
        R, J = self.RJ(T)
        nr0 = la.norm(R)

        if self.verbose:
            print("Iter.\t||R||\t\t||R||/||R0||")
            print("%i\t%e" % (0, nr0))

        for i in range(self.miter):
            dT = spla.spsolve(J, R)
            T -= dT
            R, J = self.RJ(T)
            nr = la.norm(R)
            if np.isnan(nr):
                raise RuntimeError("NaN detected!")
            if self.verbose:
                print("%i\t%e\t%e" % (i + 1, nr, nr / nr0))
            if (nr < self.atol) or (nr / nr0 < self.rtol):
                break
        else:
            raise RuntimeError("Too many iterations in newton solver!")

        return T

    def recover_tube_results(self, T, t):
        """Recover the tube flow rates and actual temperature fields

        Args:
            T (np.array):   temperatures
            t (float):      time
        """
        flow_rates = []
        tube_temperatures = []
        dofs_prev = []
        for i, (obj, dofs) in enumerate(zip(self.chain, self.dof_map)):
            # Panels are every other entry
            T_prev = T[dofs_prev]
            T_curr = T[dofs]

            if i % 2 == 1:
                flow_rates.append(obj.flow_rates(T_prev, T_curr, t))
                tube_temperatures.append(obj.fluid_temperatures(T_prev, T_curr, t))
            dofs_prev = dofs

        return flow_rates, tube_temperatures

    def RJ(self, T):
        """Formulate the residual and Jacobian for a given time

        Args:
            T (np.array):   nodal temperatures
        """
        R = np.zeros((self.nvals,))
        I = []
        J = []
        vals = []

        dofs_prev = []
        for obj, dofs in zip(self.chain, self.dof_map):
            T_prev = jnp.asarray(T[dofs_prev])
            T_curr = jnp.asarray(T[dofs])

            R[dofs] = obj.residual(T_prev, T_curr, self.t)
            Jprev = jacfwd(obj.residual, argnums=0)(T_prev, T_curr, self.t)

            i, j, v = coo_entries(dofs, dofs_prev, Jprev)
            I.extend(i)
            J.extend(j)
            vals.extend(v)

            Jcurr = jacfwd(obj.residual, argnums=1)(T_prev, T_curr, self.t)
            i, j, v = coo_entries(dofs, dofs, Jcurr)
            I.extend(i)
            J.extend(j)
            vals.extend(v)

            dofs_prev = dofs

        J = sp.coo_array((vals, (I, J)), shape=(self.nvals, self.nvals))
        return R, J.tocsr()


def coo_entries(row, col, jac):
    """
    Provide flat list for the row indices, column indices, and entries
    of a jacobian entry

    Args:
        row (np.array): row indices
        col (np.array): column indices
        jac (np.array): actual matrix entries
    """
    i, j = np.meshgrid(row, col)

    return list(i.flatten()), list(j.flatten()), list(jac.flatten())
