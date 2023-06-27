# pylint: disable=no-member
"""
  This module contains methods for calculating the reliability and
  creep-fatigue damage given completely-solved tube results and
  damage material properties
"""
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import multiprocess


class WeibullFailureModel:
    """Parent class for time independent Weibull failure models

    Determines principal stresses from mandel stress

    Determines tube reliability and overall reliability by taking input of
    element log reliabilities from respective Weibull failure model
    """

    tolerance = 1.0e-16

    def __init__(self, pset, *args, cares_cutoff=True):
        """Initialize the Weibull Failure Model

        Boolean:
        cares_cutoff:    condition for forcing reliability as unity in case of
                        high compressive stresses
        """
        self.cares_cutoff = cares_cutoff
        self.shear_sensitive = pset.get_default("shear_sensitive", True)

    def calculate_principal_stress(self, stress):
        """
        Calculate the principal stresses from Mandel vector and converts
        to conventional notation
        """
        if stress.ndim == 2:
            tensor = np.zeros(
                stress.shape[:1] + (3, 3)
            )  # [:1] when no time steps involved
        elif stress.ndim == 3:
            tensor = np.zeros(
                stress.shape[:2] + (3, 3)
            )  # [:2] when time steps involved
        # indices where (0,0) => (1,1)
        inds = [
            [(0, 0)],
            [(1, 1)],
            [(2, 2)],
            [(1, 2), (2, 1)],
            [(0, 2), (2, 0)],
            [(0, 1), (1, 0)],
        ]
        # multiplicative factors
        mults = [1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        # Converting mandel notation to conventional notation
        for i, (grp, m) in enumerate(zip(inds, mults)):
            for a, b in grp:
                tensor[..., a, b] = stress[..., i] / m

        # return la.eigvalsh(tensor)
        pstress = la.eigvalsh(tensor)

        if self.cares_cutoff:
            pmax = np.max(pstress, axis=2)
            pmin = np.min(pstress, axis=2)
            remove = np.abs(pmin / (pmax + self.tolerance)) > 3.0
            pstress[remove] = 0.0

        return pstress

    def determine_reliability(
        self, receiver, material, time, nthreads=1, decorator=lambda x, n: x
    ):
        """
        Determine the reliability of the tubes in the receiver by calculating individual
        material point reliabilities and finding the minimum of all points.

        Parameters:
          receiver        fully-solved receiver object
          material        material model to use
          time (float):   time in service

        Additional Parameters:
          nthreads        number of threads
          decorator       progress bar
        """
        with multiprocess.Pool(nthreads) as p:
            results = list(
                decorator(
                    p.imap(
                        lambda x: self.tube_log_reliability(
                            x, material, receiver, time
                        ),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )

        p_tube = np.array([res[0] for res in results])
        tube_fields = [res[1] for res in results]

        # Tube reliability is the minimum of all the time steps
        tube = np.min(p_tube, axis=1)

        # Panel reliability
        tube_multipliers = np.array(
            [
                [t.multiplier_val for (ti, t) in p.tubes.items()]
                for (pi, p) in receiver.panels.items()
            ]
        )
        panel = np.sum(tube.reshape(receiver.npanels, -1) * tube_multipliers, axis=1)

        # Overall reliability
        overall = np.sum(panel)

        # Add the field to the tubes
        for tubei, field in zip(receiver.tubes, tube_fields):
            tubei.add_quadrature_results("log_reliability", field)

        # Convert back from log-prob as we go
        return {
            "tube_reliability": np.exp(tube),
            "panel_reliability": np.exp(panel),
            "overall_reliability": np.exp(overall),
        }

    def tube_log_reliability(self, tube, material, receiver, time):
        """
        Calculate the log reliability of a single tube
        """
        volumes = tube.element_volumes()

        stresses = np.transpose(
            np.mean(
                np.stack(
                    (
                        tube.quadrature_results["stress_xx"],
                        tube.quadrature_results["stress_yy"],
                        tube.quadrature_results["stress_zz"],
                        tube.quadrature_results["stress_yz"],
                        tube.quadrature_results["stress_xz"],
                        tube.quadrature_results["stress_xy"],
                    )
                ),
                axis=-1,
            ),
            axes=(1, 2, 0),
        )

        temperatures = np.mean(tube.quadrature_results["temperature"], axis=-1)

        # Figure out the number of repetitions of the load cycle
        if receiver.days != 1:
            raise RuntimeError(
                "Time dependent reliability requires the load cycle"
                "be a single, representative cycle"
            )

        # Do it this way so we can vectorize
        inc_prob = self.calculate_element_log_reliability(
            tube.times, stresses, temperatures, volumes, material, time
        )

        # Return the sums as a function of time along with the field itself
        inc_prob = np.array(list(inc_prob) * len(tube.times)).reshape(
            len(tube.times), -1
        )

        return np.sum(inc_prob, axis=1), np.transpose(
            np.stack((inc_prob, inc_prob)), axes=(1, 2, 0)
        )


class CrackShapeIndependent(WeibullFailureModel):
    """
    Parent class for crack shape independent models
    which include only PIA and WNTSA models

    Determines normal stress acting on crack
    """

    def __init__(self, pset, *args, **kwargs):
        """
        Create a mesh grid of angles that can represent crack orientations
        Evaluate direction cosines using the angles

        Default values given to nalpha and nbeta which are the number of
        segments in the mesh grid
        """
        super().__init__(pset, *args, **kwargs)

        # limits and number of segments for angles
        self.nalpha = pset.get_default("nalpha", 21)
        self.nbeta = pset.get_default("nbeta", 31)

        # Mesh grid of the vectorized angle values
        self.A, self.B = np.meshgrid(
            np.linspace(0, np.pi, self.nalpha),
            np.linspace(0, 2 * np.pi, self.nbeta, endpoint=False),
            indexing="ij",
        )

        # Increment of angles to be used in evaluating integral
        self.dalpha = (self.A[-1, -1] - self.A[0, 0]) / (self.nalpha - 1)
        self.dbeta = (self.B[-1, -1] - self.B[0, 0]) / (self.nbeta - 1)

        # Direction cosines
        self.l = np.cos(self.A)
        self.m = np.sin(self.A) * np.cos(self.B)
        self.n = np.sin(self.A) * np.sin(self.B)

    def calculate_normal_stress(self, mandel_stress):
        """
        Use direction cosines to calculate the normal stress to
        a crack given the Mandel vector
        """
        # Principal stresses
        pstress = self.calculate_principal_stress(mandel_stress)

        # Normal stress
        return (
            pstress[..., 0, None, None] * (self.l**2)
            + pstress[..., 1, None, None] * (self.m**2)
            + pstress[..., 2, None, None] * (self.n**2)
        )


class CrackShapeDependent(WeibullFailureModel):
    """
    Parent class for crack shape dependent models

    Determines normal, shear, total and equivanlent stresses acting on cracks

    Calculates the element reliability using the equivalent stress from the
    crack shape dependent models
    """

    def __init__(self, pset, *args, **kwargs):
        """
        Create a mesh grid of angles that can represent crack orientations
        Evaluate direction cosines using the angles

        Default values given to nalpha and nbeta which are the number of
        segments in the mesh grid
        """
        super().__init__(pset, *args, **kwargs)

        # limits and number of segments for angles
        self.nalpha = pset.get_default("nalpha", 121)
        self.nbeta = pset.get_default("nbeta", 121)

        # Mesh grid of the vectorized angle values
        self.A, self.B = np.meshgrid(
            np.linspace(0, np.pi / 2, self.nalpha),
            np.linspace(0, np.pi / 2, self.nbeta),
            indexing="ij",
        )

        # Increment of angles to be used in evaluating integral
        self.dalpha = (self.A[-1, -1] - self.A[0, 0]) / (self.nalpha - 1)
        self.dbeta = (self.B[-1, -1] - self.B[0, 0]) / (self.nbeta - 1)

        # Direction cosines
        self.l = np.cos(self.A)
        self.m = np.sin(self.A) * np.cos(self.B)
        self.n = np.sin(self.A) * np.sin(self.B)

    def calculate_normal_stress(self, mandel_stress):
        """
        Use direction cosines to calculate the normal stress to
        a crack given the Mandel vector
        """
        # Principal stresses
        pstress = self.calculate_principal_stress(mandel_stress)

        # Normal stress
        return (
            pstress[..., 0, None, None] * (self.l**2)
            + pstress[..., 1, None, None] * (self.m**2)
            + pstress[..., 2, None, None] * (self.n**2)
        )

    def calculate_total_stress(self, mandel_stress):
        """
        Calculate the total stress given the Mandel vector
        """
        # Principal stresses
        pstress = self.calculate_principal_stress(mandel_stress)

        # Total stress
        return np.sqrt(
            ((pstress[..., 0, None, None] * self.l) ** 2)
            + ((pstress[..., 1, None, None] * self.m) ** 2)
            + ((pstress[..., 2, None, None] * self.n) ** 2)
        )

    def calculate_shear_stress(self, mandel_stress):
        """
        Calculate the shear stress given the normal and total stress
        """
        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Total stress
        sigma = self.calculate_total_stress(mandel_stress)

        # Shear stress
        return np.sqrt(sigma**2 - sigma_n**2)

    def calculate_flattened_eq_stress(
        self,
        time,
        mandel_stress,
        temperatures,
        material,
        tot_time,
        A,
        dalpha,
        dbeta,
    ):
        """
        Calculate the integral of equivalent stresses given the material
        properties and integration limits

        Parameters:
          time:           time instance variable for each stress/temperature
          mandel_stress:  element stresses in Mandel convention
          temperatures:   element temperatures
          material:       material model object with required data that includes
                          Weibull scale parameter (svals) and Weibull modulus (mvals)
                          and fatigue parameters Bv and Nv
          tot_time:       total service time used as input to calculate reliability

        Additional Parameters:
          A:              mesh grid of vectorized angle values
          dalpha:         increment of angle alpha to be used in evaluating integral
          dbeta:          increment of angle beta to be used in evaluating integral
        """

        # Material parameters
        self.temperatures = temperatures
        self.material = material
        self.mandel_stress = mandel_stress
        self.mvals = material.modulus(temperatures)
        Nv = material.Nv(temperatures)
        Bv = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(self.mvals, axis=0)
        Nvavg = np.mean(Nv, axis=0)
        Bvavg = np.mean(Bv, axis=0)

        # Projected equivalent stresses
        sigma_e = self.calculate_eq_stress(
            mandel_stress, self.temperatures, self.material
        )

        # Max principal stresss over all time steps for each element
        sigma_e_max = np.max(sigma_e, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        if np.all(time == 0):
            sigma_e_0 = sigma_e
        else:
            sigma_e[sigma_e < 0] = 0
            g = (
                np.trapz(
                    (sigma_e / (sigma_e_max + self.tolerance))
                    ** Nvavg[..., None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_e_0 = (
                (
                    ((sigma_e_max ** Nvavg[..., None, None]) * g * tot_time)
                    / Bvavg[..., None, None]
                )
                + (sigma_e_max ** (Nvavg[..., None, None] - 2))
            ) ** (1 / (Nvavg[..., None, None] - 2))

        # Defining area integral element wise
        integral = (
            (sigma_e_0 ** mavg[..., None, None])
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )

        # Flatten the last axis and calculate the mean of the positive values along that axis
        flat = integral.reshape(integral.shape[:-2] + (-1,))

        # Summing over area integral elements ignoring NaN values
        flat = np.nansum(flat, axis=-1)
        return flat

    def calculate_element_log_reliability(
        self, time, mandel_stress, temperatures, volumes, material, tot_time
    ):
        """
        Calculate the element log reliability given the equivalent stress

        Parameters:
          mandel_stress:    element stresses in Mandel convention
          temperatures:     element temperatures
          volumes:          element volumes
          material:         material model object with required data that includes
                            Weibull scale parameter (svals), Weibull modulus (mvals)
                            and uniaxial and polyaxial crack density coefficient (kvals,kpvals)
          shear_sensitive:  shear sensitivity condition for evaluating normalized
                            crack density coefficient (kbar) using a
                            numerically (when True) or using a fixed expression (when False)
        """
        self.temperatures = temperatures
        self.material = material
        self.mandel_stress = mandel_stress

        # Material parameters
        svals = material.strength(temperatures)
        mvals = material.modulus(temperatures)
        kvals = svals ** (-mvals)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)

        # shear_sensitive = True

        if self.shear_sensitive is True:
            kbar = self.calculate_kbar(
                self.temperatures, self.material, self.A, self.dalpha, self.dbeta
            )
        else:
            kbar = 2 * mavg + 1

        kpvals = kbar * kavg

        # Equivalent stress raied to exponent mv
        flat = (
            self.calculate_flattened_eq_stress(
                time,
                mandel_stress,
                self.temperatures,
                self.material,
                tot_time,
                self.A,
                self.dalpha,
                self.dbeta,
            )
        ) ** (1 / mavg)

        return -(2 * kpvals / np.pi) * (flat**mavg) * volumes


class PIAModel(CrackShapeIndependent):
    """
    Principal of independent action failure model

    Calculates reliability using only tensile stresses
    that are assumed to act independently on cracks
    """

    def calculate_element_log_reliability(
        self, time, mandel_stress, temperatures, volumes, material, tot_time
    ):
        """
        Calculate the element log reliability

        Parameters:
          time:           time instance variable for each stress/temperature
          mandel_stress:  element stresses in Mandel convention
          temperatures:   element temperatures
          volumes:        element volumes
          material:       material model object with required data that includes
                          Weibull scale parameter (svals), Weibull modulus (mvals)
                          and fatigue parameters (Bv,Nv)
          tot_time:       total service time used as input to calculate reliability
        """

        # Principal stresses
        pstress = self.calculate_principal_stress(mandel_stress)

        # Material parameters
        svals = material.strength(temperatures)
        mvals = material.modulus(temperatures)
        kvals = svals ** (-mvals)
        Nv = material.Nv(temperatures)
        Bv = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)
        Nvavg = np.mean(Nv, axis=0)
        Bvavg = np.mean(Bv, axis=0)

        # Only tension
        pstress[pstress < 0] = 0

        # Max principal stresss over all time steps for each element
        pstress_max = np.max(pstress, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        if np.all(time == 0):
            pstress_0 = pstress
        else:
            g = (
                np.trapz(
                    (pstress / (pstress_max + 1.0e-14)) ** Nvavg[..., None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent principal stress
            pstress_0 = (
                (pstress_max ** Nvavg[..., None] * g * tot_time) / Bvavg[..., None]
                + (pstress_max ** (Nvavg[..., None] - 2))
            ) ** (1 / (Nvavg[..., None] - 2))

        return -kavg * np.sum(pstress_0 ** mavg[..., None], axis=-1) * volumes


class WNTSAModel(CrackShapeIndependent):
    """
    Weibull normal tensile average failure model

    Evaluates an average normal tensile stress (acting on the cracks) integrated over the
    area of a sphere, and uses it to calculate the element reliability
    """

    def calculate_avg_normal_stress(
        self, time, mandel_stress, temperatures, material, tot_time
    ):
        """
        Calculate the average normal tensile stresses from the pricipal stresses

        Parameters:
          mandel_stress:  element stresses in Mandel convention
          temperatures:   element temperatures
          material:       material model object with required data that includes
                          Weibull scale parameter (svals), Weibull modulus (mvals)
                          and fatigue parameters (Bv,Nv)
          tot_time:       total service time used as input to calculate reliability
        """

        # Material parameters
        self.temperatures = temperatures
        self.material = material
        self.mandel_stress = mandel_stress

        mvals = material.modulus(temperatures)
        Nv = material.Nv(temperatures)
        Bv = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        Nvavg = np.mean(Nv, axis=0)
        Bvavg = np.mean(Bv, axis=0)

        # Time dependent Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Considering only tensile stresses
        sigma_n[sigma_n < 0] = 0

        # Max normal stresss over all time steps for each element
        sigma_n_max = np.max(sigma_n, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        if np.all(time == 0):
            sigma_n_0 = sigma_n
        else:
            g = (
                np.trapz(
                    (sigma_n / (sigma_n_max + self.tolerance))
                    ** Nvavg[..., None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_n_0 = (
                (
                    ((sigma_n_max ** Nvavg[..., None, None]) * g * tot_time)
                    / Bvavg[..., None, None]
                )
                + (sigma_n_max ** (Nvavg[..., None, None] - 2))
            ) ** (1 / (Nvavg[..., None, None] - 2))

        integral = (
            (sigma_n_0 ** mavg[..., None, None])
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        ) / (4 * np.pi)

        # Flatten the last axis and calculate the mean of the positive values along that axis
        flat = integral.reshape(integral.shape[:-2] + (-1,))

        # Average stress from summing while ignoring NaN values
        return np.nansum(np.where(flat >= 0.0, flat, np.nan), axis=-1)

    def calculate_element_log_reliability(
        self, time, mandel_stress, temperatures, volumes, material, tot_time
    ):
        """
        Calculate the element log reliability

        Parameters:
          mandel_stress:    element stresses in Mandel convention
          temperatures:     element temperatures
          volumes:          element volumes
          material:         material model object with required data that includes
                            Weibull scale parameter (svals), Weibull modulus (mvals)
                            and uniaxial and polyaxial crack density coefficient (kvals,kpvals)
          tot_time:       total time at which to calculate reliability
        """
        self.temperatures = temperatures
        self.material = material
        self.mandel_stress = mandel_stress

        # Material parameters
        svals = material.strength(temperatures)
        mvals = material.modulus(temperatures)
        kvals = svals ** (-mvals)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)

        # Polyaxial Batdorf crack density coefficient
        kpvals = (2 * mavg + 1) * kavg

        # Average normal tensile stress raied to exponent mv
        avg_nstress = (
            self.calculate_avg_normal_stress(
                time,
                mandel_stress,
                self.temperatures,
                self.material,
                tot_time,
            )
        ) ** (1 / mavg)

        return -kpvals * (avg_nstress**mavg) * volumes


class MTSModelGriffithFlaw(CrackShapeDependent):
    """
    Maximum tensile stess failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the maximum tensile stress fracture criterion for a Griffith flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses
        """
        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (sigma_n + np.sqrt((sigma_n**2) + (tau**2)))

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)
        using the Weibull scale parameter (svals), Weibull modulus (mvals),
        poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (((np.sin(self.A)) ** 2) * ((np.cos(self.A)) ** 2))
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class MTSModelPennyShapedFlaw(CrackShapeDependent):
    """
    Maximum tensile stess failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the maximum tensile stress fracture criterion for a penny shaped flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the average normal tensile stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt((sigma_n**2) + ((tau / (1 - (0.5 * nu[..., None, None]))) ** 2))
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                ((np.sin(2 * self.A)) ** 2)
                                / (2 - (nu[..., None, None] ** 2))
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class CSEModelGriffithFlaw(CrackShapeDependent):
    """
    Coplanar strain energy failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a Griffith flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses
        """
        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau**2))

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            ((np.cos(self.A)) ** mavg[..., None, None])
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class CSEModelPennyShapedFlaw(CrackShapeDependent):
    """
    Coplanar strain energy failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a penny shaped flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau / (1 - (0.5 * nu[..., None, None]))) ** 2)

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    np.sqrt(
                        ((np.cos(self.A)) ** 4)
                        + (
                            ((np.sin(2 * self.A)) ** 2)
                            / ((2 - nu[..., None, None]) ** 2)
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class SMMModelGriffithFlaw(CrackShapeDependent):
    """
    Shetty mixed-mod failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        cbar:   Shetty model empirical constant (cbar)
        """

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n + np.sqrt((sigma_n**2) + ((2 * tau / cbar[..., None, None]) ** 2))
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu),
                        Shetty model empirical constant (cbar)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                ((np.sin(2 * self.A)) ** 2)
                                / (cbar[..., None, None] ** 2)
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class SMMModelPennyShapedFlaw(CrackShapeDependent):
    """
    Shetty mixed mode failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        nu:     Poisson ratio
        cbar:   Shetty model empirical constant (cbar)
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt(
                (sigma_n**2)
                + ((4 * tau / (cbar[..., None, None] * (2 - nu[..., None, None]))) ** 2)
            )
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu),
                        Shetty model empirical constant (cbar)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                (4 * ((np.sin(2 * self.A)) ** 2))
                                / (
                                    (cbar[..., None, None] ** 2)
                                    * ((nu[..., None, None] - 2) ** 2)
                                )
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class DamageCalculator:
    """
    Parent class for all damage calculators, handling common iteration
    and scaling options
    """

    def __init__(self, pset):
        """
        Parameters:
          pset:       damage parameters
        """
        self.extrapolate = pset.get_default("extrapolate", "lump")
        self.order = pset.get_default("order", 1)

    def single_cycles(self, tube, material, receiver):
        """
        Calculate damage for a single tube

        Parameters:
          tube:       fully-populated tube object
          material:   damage material
          receiver:   receiver object (for metadata)
        """
        raise NotImplementedError("Superclass not implemented")

    def determine_life(self, receiver, material, nthreads=1, decorator=lambda x, n: x):
        """
        Determine the life of the receiver by calculating individual
        material point damage and finding the minimum of all points.

        Parameters:
          receiver        fully-solved receiver object
          material        material model ot use

        Additional Parameters:
          nthreads        number of threads
          decorator       progress bar
        """
        # pylint: disable=no-member
        with multiprocess.Pool(nthreads) as p:
            Ns = list(
                decorator(
                    p.imap(
                        lambda x: self.single_cycles(x, material, receiver),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )
        N = min(Ns)

        # Results come out as days
        return N

    def make_extrapolate(self, D):
        """
        Return a damage extrapolation function based on self.extrapolate
        giving the damage for the nth cycle

        Parameters:
          D:      raw, per cycle damage
        """
        if self.extrapolate == "lump":
            return lambda N, D=D: N * np.sum(D) / len(D)
        elif self.extrapolate == "last":

            def Dfn(N, D=D):
                N = int(N)
                if N < len(D) - 1:
                    return np.sum(D[:N])
                else:
                    return np.sum(D[:-1]) + D[-1] * N

            return Dfn
        elif self.extrapolate == "poly":
            p = np.polyfit(np.array(list(range(len(D)))) + 1, D, self.order)
            return lambda N, p=p: np.polyval(p, N)
        else:
            raise ValueError(
                "Unknown damage extrapolation approach %s!" % self.extrapolate
            )


class TimeFractionInteractionDamage(DamageCalculator):
    """
    Calculate life using the ASME time-fraction type approach
    """

    def single_cycles(self, tube, material, receiver):
        """
        Calculate the single-tube number of repetitions to failure

        Parameters:
          tube        single tube with full results
          material    damage material model
          receiver    receiver, for metadata
        """
        # Material point cycle creep damage
        Dc = self.creep_damage(tube, material, receiver)

        # Material point cycle fatigue damage
        Df = self.fatigue_damage(tube, material, receiver)

        nc = receiver.days

        # This is going to be expensive, but I don't see much way around it
        return min(
            self.calculate_max_cycles(
                self.make_extrapolate(c), self.make_extrapolate(f), material
            )
            for c, f in zip(Dc.reshape(nc, -1).T, Df.reshape(nc, -1).T)
        )

    def calculate_max_cycles(self, Dc, Df, material, rep_min=1, rep_max=1e6):
        """
        Actually calculate the maximum number of repetitions for a single point

        Parameters:
          Dc          creep damage per simulated cycle
          Df          fatigue damage per simulated cycle
          material    damaged material properties
        """
        if not material.inside_envelope("cfinteraction", Df(rep_min), Dc(rep_min)):
            return 0

        if material.inside_envelope("cfinteraction", Df(rep_max), Dc(rep_max)):
            return np.inf

        return opt.brentq(
            lambda N: material.inside_envelope("cfinteraction", Df(N), Dc(N)) - 0.5,
            rep_min,
            rep_max,
        )

    def creep_damage(self, tube, material, receiver):
        """
        Calculate creep damage at each material point

        Parameters:
          tube        single tube with full results
          material    damage material model
          receiver    receiver, for metadata
        """
        # For now just use the von Mises effective stress
        vm = np.sqrt(
            (
                (
                    tube.quadrature_results["stress_xx"]
                    - tube.quadrature_results["stress_yy"]
                )
                ** 2.0
                + (
                    tube.quadrature_results["stress_yy"]
                    - tube.quadrature_results["stress_zz"]
                )
                ** 2.0
                + (
                    tube.quadrature_results["stress_zz"]
                    - tube.quadrature_results["stress_xx"]
                )
                ** 2.0
                + 6.0
                * (
                    tube.quadrature_results["stress_xy"] ** 2.0
                    + tube.quadrature_results["stress_yz"] ** 2.0
                    + tube.quadrature_results["stress_xz"] ** 2.0
                )
            )
            / 2.0
        )

        tR = material.time_to_rupture(
            "averageRupture", tube.quadrature_results["temperature"], vm
        )
        dts = np.diff(tube.times)
        time_dmg = dts[:, np.newaxis, np.newaxis] / tR[1:]

        # Break out to cycle damage
        inds = self.id_cycles(tube, receiver)

        cycle_dmg = np.array(
            [
                np.sum(time_dmg[inds[i] : inds[i + 1]], axis=0)
                for i in range(receiver.days)
            ]
        )

        return cycle_dmg

    def fatigue_damage(self, tube, material, receiver):
        """
        Calculate fatigue damage at each material point

        Parameters:
          tube        single tube with full results
          material    damage material model
          receiver    receiver, for metadata
        """
        # Identify cycle boundaries
        inds = self.id_cycles(tube, receiver)

        # Run through each cycle and ID max strain range and fatigue damage
        strain_names = [
            "mechanical_strain_xx",
            "mechanical_strain_yy",
            "mechanical_strain_zz",
            "mechanical_strain_yz",
            "mechanical_strain_xz",
            "mechanical_strain_xy",
        ]
        strain_factors = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

        cycle_dmg = np.array(
            [
                self.cycle_fatigue(
                    np.array(
                        [
                            ef * tube.quadrature_results[en][inds[i] : inds[i + 1]]
                            for en, ef in zip(strain_names, strain_factors)
                        ]
                    ),
                    tube.quadrature_results["temperature"][inds[i] : inds[i + 1]],
                    material,
                )
                for i in range(receiver.days)
            ]
        )

        return cycle_dmg

    def id_cycles(self, tube, receiver):
        """
        Helper to separate out individual cycles by index

        Parameters:
          tube        single tube with results
          receiver    receiver, for metadata
        """
        tm = np.mod(tube.times, receiver.period)
        inds = list(np.where(tm == 0)[0])
        if len(inds) != (receiver.days + 1):
            raise ValueError(
                "Tube times not compatible with the receiver"
                " number of days and cycle period!"
            )

        return inds

    def cycle_fatigue(self, strains, temperatures, material, nu=0.5):
        """
        Calculate fatigue damage for a single cycle

        Parameters:
          strains         single cycle strains
          temperatures    single cycle temperatures
          material        damage model

        Additional parameters:
          nu              effective Poisson's ratio to use
        """
        pt_temps = np.max(temperatures, axis=0)

        pt_eranges = np.zeros(pt_temps.shape)

        nt = strains.shape[1]
        for i in range(nt):
            for j in range(nt):
                de = strains[:, j] - strains[:, i]
                eq = (
                    np.sqrt(2)
                    / (2 * (1 + nu))
                    * np.sqrt(
                        (de[0] - de[1]) ** 2
                        + (de[1] - de[2]) ** 2
                        + (de[2] - de[0]) ** 2.0
                        + 3.0 / 2.0 * (de[3] ** 2.0 + de[4] ** 2.0 + de[5] ** 2.0)
                    )
                )
                pt_eranges = np.maximum(pt_eranges, eq)

        dmg = np.zeros(pt_eranges.shape)
        # pylint: disable=not-an-iterable
        for ind in np.ndindex(*dmg.shape):
            dmg[ind] = 1.0 / material.cycles_to_fail(
                "nominalFatigue", pt_temps[ind], pt_eranges[ind]
            )

        return dmg
