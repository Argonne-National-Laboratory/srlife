# pylint: disable=no-member, wrong-import-position
"""
    Thermal-fluid specific material classes
"""
import xml.etree.ElementTree as ET

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from srlife import materials


class ThermalFluidMaterial:
    """Thermal fluid material properties

    The goal of this class is to provide film coefficient as a function of
    fluid velocity, temperature, and tube inner radius.

    The current implementation works from the
    - Heat capacity (constant pressure)
    - Density
    - Dynamic viscosity
    - Conductivity

    as a function of temperature (in K).
    """

    def __init__(
        self,
        film_min=1e-8,
        T_max=2000.0,
        T_min=0.0,
        laminar_cutoff=2e3,
        laminar_value=4.01,
    ):
        self.film_min = film_min
        self.T_max = T_max
        self.T_min = T_min
        self.laminar_cutoff = laminar_cutoff
        self.laminar_value = laminar_value

    def T_effective(self, T):
        """
        Cutoff T to avoid issues with the correlations
        """
        return jnp.maximum(jnp.minimum(T, self.T_max), self.T_min)

    @classmethod
    def load(cls, fname, modelname):
        """Load model from an XML file

        Parameters:
            fname:      XML filename
            modelname:  model name in the file
        """
        tag, mtype = materials.find_name(fname, modelname)
        data = materials.load_node(tag)[modelname]

        if mtype == "PolynomialThermalFluidMaterial":
            return PolynomialThermalFluidMaterial.load(data)
        else:
            raise ValueError("Unknown ThermalFluidMaterial type %s" % mtype)

    def save(self, fname, modelname):
        """Save model to an XML file

        Parameters:
            fname:      XML filename
            modelname:  model name in the file
        """
        root = ET.Element("models")

        materials.save_node(
            modelname, self.data(), root, attrib={"type": self.model_type}
        )
        tree = ET.ElementTree(element=root)
        tree.write(fname)

    def film_coefficient(self, T, u, r):
        """Film coefficient

        Calculate the film coefficient as a function of
        temperature, flow velocity, and tube diameter

        Parameters:
            T:      temperature, in K
            u:      flow velocity, in mm/hr
            r:      tube inner radius, in mm
        """
        nu = self.nusselt(T, u, r)

        return jnp.maximum(nu * self.k(T) / (2.0 * r), self.film_min)

    def reynolds(self, T, u, r):
        """Reynolds number

        Parameters:
            T:      temperature, in K
            u:      flow velocity, in mm/hr
            r:      tube inner radius, in mm
        """
        return self.rho(T) * u * 2.0 * r / self.mu(T)

    def prandtl(self, T):
        """Prandtl number

        Parameters:
            T:      temperature, in K
        """
        return self.cp(T) * self.mu(T) / self.k(T)

    def nusselt(self, T, u, r):
        """Nusselt number

        This is really where the work gets done in calculating the film
        coefficient and different correlations are  possible.

        Right now we use the Gnielinski correlation for turbulent flow

        Parameters:
            T:      temperature, in K
            u:      flow velocity, in m/s
            r:      tube inner radius, in mm
        """
        re = self.reynolds(self.T_effective(T), u, r)
        pr = self.prandtl(self.T_effective(T))
        f = (0.79 * jnp.log(re) - 1.64) ** -2.0

        turbulent = ((f / 8.0) * (re - 1000.0) * pr) / (
            1.0 + 12.7 * (f / 8.0) ** 0.5 * (pr ** (2.0 / 3.0) - 1.0)
        )

        turbulent.at[re < self.laminar_cutoff].set(self.laminar_value)

        return turbulent


class PolynomialThermalFluidMaterial(ThermalFluidMaterial):
    """Thermal-fluid relations correlated with polynomials

    The polynomial definitions use C, the class handles converting K -> C

    Parameters:
        cp_poly (array): coefficients for the heat capacity, in numpy order
        rho_poly (array): coefficients for the density, in numpy order
        mu_poly (array): coefficients for the viscosity, in numpy order
        k_poly (array): coefficients for the conductivity, in numpy order
        film_min (Optional[float]): lower cutoff on the value of the film coefficient
        T_max (Optional[float]): upper cutoff on temperature used in finding values
        T_min (Optional[float]): lower cutoff on temperature used in finding values
        laminar_cutoff (Optional[float]): Reynolds number upper limit for laminar flow
        laminar_value (Optional[float]): Nusslet number to use in laminar flow
    """

    def __init__(self, cp_poly, rho_poly, mu_poly, k_poly, **kwargs):
        super().__init__(**kwargs)
        self.cp_poly = jnp.asarray(cp_poly)
        self.rho_poly = jnp.asarray(rho_poly)
        self.mu_poly = jnp.asarray(mu_poly)
        self.k_poly = jnp.asarray(k_poly)

        self.model_type = "PolynomialThermalFluidMaterial"

    def data(self):
        """
        Reduce the class down to a dictionary of strings for
        serialization
        """
        return {
            "cp_poly": materials.string_array(self.cp_poly),
            "rho_poly": materials.string_array(self.rho_poly),
            "mu_poly": materials.string_array(self.mu_poly),
            "k_poly": materials.string_array(self.k_poly),
            "film_min": str(self.film_min),
            "T_max": str(self.T_max),
            "T_min": str(self.T_min),
            "laminar_cutoff": str(self.laminar_cutoff),
            "laminar_value": str(self.laminar_value),
        }

    @classmethod
    def load(cls, values):
        """Create from a dictionary

        Parameters:
            values:     dictionary of values
        """
        # Assemble kwargs
        kwargs = {
            k: float(values[k])
            for k in ["film_min", "T_max", "T_min", "laminar_cutoff", "laminar_value"]
            if k in values
        }

        return cls(
            materials.destring_array(values["cp_poly"]),
            materials.destring_array(values["rho_poly"]),
            materials.destring_array(values["mu_poly"]),
            materials.destring_array(values["k_poly"]),
            **kwargs
        )

    def cp(self, T):
        """Heat capacity as a function of temperature in K

        Parameters:
            T:      temperature, in K
        """
        return jnp.polyval(self.cp_poly, T)

    def rho(self, T):
        """Density as a function of temperature in K

        Parameters:
            T:      temperature, in K
        """
        return jnp.polyval(self.rho_poly, self.T_effective(T))

    def mu(self, T):
        """Dynamic viscosity, as a function of temperature in K

        Parameters:
            T:  temperature, in K
        """
        return jnp.polyval(self.mu_poly, T)

    def k(self, T):
        """Conductivity, as a function of temperature in K

        Parameters:
            T:  temperature, in K
        """
        return jnp.polyval(self.k_poly, T)
