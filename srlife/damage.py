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
from scipy.special import gamma


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

    def _full_stress(self, stress):
        """
        Calculate the full stress tensor for further processing
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

        return tensor

    def calculate_principal_stress(self, stress):
        """
        Calculate the principal stresses from Mandel vector and converts
        to conventional notation
        """
        tensor = self._full_stress(stress)

        pstress = la.eigvalsh(tensor)

        if self.cares_cutoff:
            pmax = np.max(pstress, axis=2)
            pmin = np.min(pstress, axis=2)
            remove = np.abs(pmin / (pmax + self.tolerance)) > 3.0
            pstress[remove] = 0.0

        return pstress

    def determine_reliability_volume(
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
            volume_results = list(
                decorator(
                    p.imap(
                        lambda x: self.tube_volume_log_reliability(
                            x, material, receiver, time
                        ),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )

        p_tube_volume = np.array([res[0] for res in volume_results])
        tube_fields_volume = [res[1] for res in volume_results]

        # Tube reliability is the minimum of all the time steps
        tube_volume = np.min(p_tube_volume, axis=1)

        # Panel reliability
        tube_multipliers = np.array(
            [
                [t.multiplier_val for (ti, t) in p.tubes.items()]
                for (pi, p) in receiver.panels.items()
            ]
        )
        panel_volume = np.sum(
            tube_volume.reshape(receiver.npanels, -1) * tube_multipliers, axis=1
        )

        # Overall reliability
        overall_volume = np.sum(panel_volume)

        # Add the field to the tubes
        for tubei, field in zip(receiver.tubes, tube_fields_volume):
            tubei.add_quadrature_results("log_reliability", field)

        # Convert back from log-prob as we go
        return {
            "tube_reliability_volume": np.exp(tube_volume),
            "panel_reliability_volume": np.exp(panel_volume),
            "overall_reliability_volume": np.exp(overall_volume),
        }

    def determine_reliability_surface(
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
            surface_results = list(
                decorator(
                    p.imap(
                        lambda x: self.tube_surface_log_reliability(
                            x, material, receiver, time
                        ),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )

        p_tube_surface = np.array([res[0] for res in surface_results])
        tube_fields_surface = [res[1] for res in surface_results]

        # Tube reliability is the minimum of all the time steps
        tube_surface = np.min(p_tube_surface, axis=1)

        # Panel reliability
        tube_multipliers = np.array(
            [
                [t.multiplier_val for (ti, t) in p.tubes.items()]
                for (pi, p) in receiver.panels.items()
            ]
        )
        panel_surface = np.sum(
            tube_surface.reshape(receiver.npanels, -1) * tube_multipliers, axis=1
        )

        # Overall reliability
        overall_surface = np.sum(panel_surface)

        # Add the field to the tubes
        for tubei, field in zip(receiver.tubes, tube_fields_surface):
            tubei.add_quadrature_results("log_reliability", field)

        # Convert back from log-prob as we go
        return {
            "tube_reliability_surface": np.exp(tube_surface),
            "panel_reliability_surface": np.exp(panel_surface),
            "overall_reliability_surface": np.exp(overall_surface),
        }

    def determine_reliability_combined(
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
            total_results = list(
                decorator(
                    p.imap(
                        lambda x: self.tube_combined_log_reliability(
                            x, material, receiver, time
                        ),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )

        p_tube_combined = np.array([res[0] for res in total_results])
        tube_fields_combined = [res[1] for res in total_results]

        # Tube reliability is the minimum of all the time steps
        tube_combined = np.min(p_tube_combined, axis=1)

        # Panel reliability
        tube_multipliers = np.array(
            [
                [t.multiplier_val for (ti, t) in p.tubes.items()]
                for (pi, p) in receiver.panels.items()
            ]
        )
        panel_combined = np.sum(
            tube_combined.reshape(receiver.npanels, -1) * tube_multipliers, axis=1
        )

        # Overall reliability
        overall_combined = np.sum(panel_combined)

        # Add the field to the tubes
        for tubei, field in zip(receiver.tubes, tube_fields_combined):
            tubei.add_quadrature_results("log_reliability", field)

        # Convert back from log-prob as we go
        return {
            "tube_reliability_combined": np.exp(tube_combined),
            "panel_reliability_combined": np.exp(panel_combined),
            "overall_reliability_combined": np.exp(overall_combined),
        }

    def tube_volume_log_reliability(self, tube, material, receiver, time):
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

        # Volume element log reliability
        inc_prob = self.calculate_volume_element_log_reliability(
            tube.times, stresses, temperatures, volumes, material, time
        )

        # Return the sums as a function of time along with the field itself
        inc_prob = np.array(list(inc_prob) * len(tube.times)).reshape(
            len(tube.times), -1
        )

        return np.sum(inc_prob, axis=1), np.transpose(
            np.stack((inc_prob, inc_prob)), axes=(1, 2, 0)
        )

    def tube_surface_log_reliability(self, tube, material, receiver, time):
        """
        Calculate the log reliability of a single tube
        """
        # volumes = tube.element_volumes()

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
        # Getting surface elements, surface normals and surface areas from surface_elements function
        surface_elements, surface_normals = tube.surface_elements()
        surface_areas = tube.element_surface_areas()

        temperatures = np.mean(tube.quadrature_results["temperature"], axis=-1)

        # Figure out the number of repetitions of the load cycle
        if receiver.days != 1:
            raise RuntimeError(
                "Time dependent reliability requires the load cycle"
                "be a single, representative cycle"
            )

        # Surface element log reliability
        inc_prob = self.calculate_surface_element_log_reliability(
            tube.times,
            stresses,
            surface_elements,
            surface_normals,
            temperatures,
            surface_areas,
            material,
            time,
        )

        # Return the sums as a function of time along with the field itself
        inc_prob = np.array(list(inc_prob) * len(tube.times)).reshape(
            len(tube.times), -1
        )

        return np.sum(inc_prob, axis=1), np.transpose(
            np.stack((inc_prob, inc_prob)), axes=(1, 2, 0)
        )

    def tube_combined_log_reliability(self, tube, material, receiver, time):
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
        # Getting surface elements, surface normals and surface areas from surface_elements function
        surface_elements, surface_normals = tube.surface_elements()
        surface_areas = tube.element_surface_areas()

        # Indices where surface elements is True
        surface_indices = np.where(surface_elements)[0]
        temperatures = np.mean(tube.quadrature_results["temperature"], axis=-1)

        # Figure out the number of repetitions of the load cycle
        if receiver.days != 1:
            raise RuntimeError(
                "Time dependent reliability requires the load cycle"
                "be a single, representative cycle"
            )

        # Total combined element log reliability
        inc_prob = self.calculate_volume_element_log_reliability(
            tube.times, stresses, temperatures, volumes, material, time
        )
        inc_prob[surface_indices] += self.calculate_surface_element_log_reliability(
            tube.times,
            stresses,
            surface_elements,
            surface_normals,
            temperatures,
            surface_areas,
            material,
            time,
        )
        # ensuring no None values in inc_prob
        inc_prob = [i for i in inc_prob if not None]

        # Return the sums as a function of time along with the field itself
        inc_prob = np.array(list(inc_prob) * len(tube.times)).reshape(
            len(tube.times), -1
        )

        return np.sum(inc_prob, axis=1), np.transpose(
            np.stack((inc_prob, inc_prob)), axes=(1, 2, 0)
        )

    def _calculate_surface_stresses(self, stresses, surface_elements, surface_normals):
        """
        Calculate the surface stresses (for those elements on the surface)

        Parameters:
            stresses (np.array): tube stress array in Mandel convention
            surface_elements (np.array): boolean indexing array giving
                which elements are on the surface
            surface_normals (np.array): surface normals, for those
                elements on the surface

        Returns:
            surface_stresses (np.array): (ntime, nelem, 6) array of surface stresses
        """
        full_stress = self._full_stress(stresses)

        surf_stress = np.zeros_like(full_stress)

        # If surface stresses do no have the number of surfaces add a new axis for them
        if surf_stress.ndim == 4:
            surf_stress = surf_stress[..., np.newaxis, :]

        # If surface normals do no have the number of surfaces add a new axis for them
        if surface_normals.ndim == 2:
            nx = surface_normals[..., np.newaxis, 0]
            ny = surface_normals[..., np.newaxis, 1]
            nz = surface_normals[..., np.newaxis, 2]
        elif surface_normals.ndim == 3:
            nx = surface_normals[..., 0]
            ny = surface_normals[..., 1]
            nz = surface_normals[..., 2]

        P = np.array(
            [
                [1 - nx**2, -nx * ny, -nx * nz],
                [-nx * ny, 1 - ny**2, -ny * nz],
                [-nx * nz, -ny * nz, 1 - nz**2],
            ]
        ).transpose(2, 0, 1, 3)

        surf_stress = np.einsum(
            "esik, tekl, esjl->tesij",
            P[surface_elements],
            full_stress[:, surface_elements],
            P[surface_elements],
        )

        return surf_stress

    def calculate_surface_principal_stress(
        self, stresses, surface_elements, surface_normals
    ):
        """
        Calculate the principal stresses from Mandel vector and converts
        to conventional notation
        """
        surf_tensor = self._calculate_surface_stresses(
            stresses, surface_elements, surface_normals
        )

        surf_pstress = la.eigvalsh(surf_tensor)

        if self.cares_cutoff:
            smax = np.max(surf_pstress, axis=-1)
            smin = np.min(surf_pstress, axis=-1)
            remove = np.abs(smin / (smax + self.tolerance)) > 3.0
            surf_pstress[remove] = 0.0

        return surf_pstress


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

        # Mesh grid of the vectorized angle values for volume flaws
        self.A, self.B = np.meshgrid(
            np.linspace(0, np.pi, self.nalpha),
            np.linspace(0, 2 * np.pi, self.nbeta, endpoint=False),
            indexing="ij",
        )

        # Vectorized angle values for surface flaws
        self.C = np.linspace(0, 2 * np.pi, self.nbeta)

        # Increment of angles to be used in evaluating integral for volume flaws
        self.dalpha = (self.A[-1, -1] - self.A[0, 0]) / (self.nalpha - 1)
        self.dbeta = (self.B[-1, -1] - self.B[0, 0]) / (self.nbeta - 1)

        # Increment of angles to be used in evaluating integral for surface flaws
        self.ddelta = (self.C[-1] - self.C[0]) / (self.nbeta - 1)

        # Direction cosines for volume flaws
        self.l = np.cos(self.A)
        self.m = np.sin(self.A) * np.cos(self.B)
        self.n = np.sin(self.A) * np.sin(self.B)

        # Direction cosines for surface flaws
        self.o = np.sin(self.C)
        self.p = np.cos(self.C)

    def calculate_volume_normal_stress(self, mandel_stress):
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

    def calculate_surface_normal_stress(self, mandel_stress, surface, normals):
        """
        Use direction cosines to calculate the normal stress to
        a crack in surface elements given the Mandel vector
        """
        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface

        # Principal stresses in surface elements
        surf_pstress = self.calculate_surface_principal_stress(
            mandel_stress, surface_elements, surface_normals
        )

        # Sorting
        surf_pstress = np.sort(surf_pstress, axis=-1)[..., ::-1]

        # Normal stress
        return surf_pstress[..., 0, None, None] * (self.p**2) + surf_pstress[
            ..., 1, None, None
        ] * (self.o**2)


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

        # Vectorized angle values for surface flaws
        self.C = np.linspace(0, np.pi / 2, self.nbeta)

        # Increment of angles to be used in evaluating integral for volume flaws
        self.dalpha = (self.A[-1, -1] - self.A[0, 0]) / (self.nalpha - 1)
        self.dbeta = (self.B[-1, -1] - self.B[0, 0]) / (self.nbeta - 1)

        # Increment of angles to be used in evaluating integral for surface flaws
        self.ddelta = (self.C[-1] - self.C[0]) / (self.nbeta - 1)

        # Direction cosines for volume flaws
        self.l = np.cos(self.A)
        self.m = np.sin(self.A) * np.cos(self.B)
        self.n = np.sin(self.A) * np.sin(self.B)

        # Direction cosines for surface flaws
        self.o = np.sin(self.C)
        self.p = np.cos(self.C)

    def calculate_volume_normal_stress(self, mandel_stress):
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

    def calculate_surface_normal_stress(self, mandel_stress, surface, normals):
        """
        Use direction cosines to calculate the normal stress to
        a crack in surface elements given the Mandel vector
        """
        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface

        # Principal stresses in surface elements
        surf_pstress = self.calculate_surface_principal_stress(
            mandel_stress, surface_elements, surface_normals
        )

        # Sorting
        surf_pstress = np.sort(surf_pstress, axis=-1)[..., ::-1]

        # Normal stress
        return surf_pstress[..., 0, None, None] * (self.p**2) + surf_pstress[
            ..., 1, None, None
        ] * (self.o**2)

    def calculate_volume_total_stress(self, mandel_stress):
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

    def calculate_surface_total_stress(self, mandel_stress, surface, normals):
        """
        Calculate the total stress in surface elements given the Mandel vector
        """
        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface

        # Principal stresses in surface elements
        surf_pstress = self.calculate_surface_principal_stress(
            mandel_stress, surface_elements, surface_normals
        )

        # Sorting
        surf_pstress = np.sort(surf_pstress, axis=-1)[..., ::-1]

        # Total stress
        return np.sqrt(
            ((surf_pstress[..., 0, None, None] * self.p) ** 2)
            + ((surf_pstress[..., 1, None, None] * self.o) ** 2)
        )

    def calculate_volume_shear_stress(self, mandel_stress):
        """
        Calculate the shear stress given the normal and total stress
        """
        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Total stress
        sigma = self.calculate_volume_total_stress(mandel_stress)

        # Shear stress
        return np.sqrt(sigma**2 - sigma_n**2)

    def calculate_surface_shear_stress(self, mandel_stress, surface, normals):
        """
        Calculate the shear stress in surface elements given the normal and total stress
        """
        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Total stress
        sigma = self.calculate_surface_total_stress(mandel_stress, surface, normals)

        # Shear stress
        return np.sqrt(sigma**2 - sigma_n**2)

    def calculate_volume_flattened_eq_stress(
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

        self.mvals = material.modulus_vol(temperatures)
        N = material.Nv(temperatures)
        B = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(self.mvals, axis=0)
        Navg = np.mean(N, axis=0)
        Bavg = np.mean(B, axis=0)

        # Projected equivalent stresses
        sigma_e = self.calculate_volume_eq_stress(
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
                    (sigma_e / (sigma_e_max + self.tolerance)) ** Navg[..., None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_e_0 = (
                (
                    ((sigma_e_max ** Navg[..., None, None]) * g * tot_time)
                    / Bavg[..., None, None]
                )
                + (sigma_e_max ** (Navg[..., None, None] - 2))
            ) ** (1 / (Navg[..., None, None] - 2))

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

    def calculate_surface_flattened_eq_stress(
        self,
        time,
        mandel_stress,
        surface_elements,
        surface_normals,
        temperatures,
        material,
        tot_time,
        ddelta,
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

        mvals = material.modulus_surf(temperatures)
        N = material.Ns(temperatures)
        B = material.Bs(temperatures)

        # Surface normals and surface elements
        count_surface_elements = np.count_nonzero(surface_elements)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)[:count_surface_elements]
        Navg = np.mean(N, axis=0)[:count_surface_elements]
        Bavg = np.mean(B, axis=0)[:count_surface_elements]

        # Projected equivalent stresses
        sigma_e = self.calculate_surface_eq_stress(
            mandel_stress,
            surface_elements,
            surface_normals,
            self.temperatures,
            self.material,
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
                    ** Navg[..., None, None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_e_0 = (
                (
                    ((sigma_e_max ** Navg[..., None, None, None]) * g * tot_time)
                    / Bavg[..., None, None, None]
                )
                + (sigma_e_max ** (Navg[..., None, None, None] - 2))
            ) ** (1 / (Navg[..., None, None, None] - 2))

        # Defining area integral element wise
        integral = (sigma_e_0 ** mavg[..., None, None, None]) * self.ddelta

        # Flatten the last axis and calculate the mean of the positive values along that axis
        flat = integral.reshape(integral.shape[:-2] + (-1,))

        # Summing over area integral elements ignoring NaN values
        flat = np.nansum(flat, axis=(-1, -2))

        return flat

    def calculate_volume_element_log_reliability(
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
        svals = material.strength_vol(temperatures)
        mvals = material.modulus_vol(temperatures)
        kvals = svals ** (-mvals)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)

        # shear_sensitive = True

        if self.shear_sensitive is True:
            try:
                kbar = self.calculate_volume_kbar(
                    self.temperatures, self.material, self.A, self.dalpha, self.dbeta
                )
            except AttributeError:
                kbar = 0.0
        else:
            kbar = 2 * mavg + 1

        kpvals = kbar * kavg

        # Equivalent stress raied to exponent mv
        try:
            flat = (
                self.calculate_volume_flattened_eq_stress(
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
        except AttributeError:
            flat = 0.0

        return -(2 * kpvals / np.pi) * (flat**mavg) * volumes

    def calculate_surface_element_log_reliability(
        self,
        time,
        mandel_stress,
        surface,
        normals,
        temperatures,
        surface_areas,
        material,
        tot_time,
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
        svals = material.strength_surf(temperatures)
        mvals = material.modulus_surf(temperatures)
        kvals = svals ** (-mvals)

        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)[:count_surface_elements]
        kavg = np.mean(kvals, axis=0)[:count_surface_elements]

        if self.shear_sensitive is True:
            try:
                kbar = self.calculate_surface_kbar(
                    self.temperatures, self.material, self.A, self.dalpha
                )
                kbar = kbar[:count_surface_elements]
            except AttributeError:
                kbar = 0.0
        else:
            # kbar = 2 * mavg + 1
            kbar = (mavg * gamma(mavg) * np.sqrt(np.pi)) / (gamma(mavg + 0.5))
            kbar = kbar[:count_surface_elements]

        kpvals = kbar * kavg

        # Equivalent stress raied to exponent mv
        try:
            flat = (
                self.calculate_surface_flattened_eq_stress(
                    time,
                    mandel_stress,
                    surface_elements,
                    surface_normals,
                    self.temperatures,
                    self.material,
                    tot_time,
                    self.ddelta,
                )
            ) ** (1 / mavg)
        except AttributeError:
            flat = 0.0

        return -(2 * kpvals / np.pi) * (flat**mavg) * surface_areas


class PIAModel(CrackShapeIndependent):
    """
    Principal of independent action failure model

    Calculates reliability using only tensile stresses
    that are assumed to act independently on cracks
    """

    def calculate_volume_element_log_reliability(
        self, time, mandel_stress, temperatures, volumes, material, tot_time
    ):
        """
        Calculate the element log reliability from volume elements only

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

        # Principal stresses in volume elements
        pstress = self.calculate_principal_stress(mandel_stress)

        # Material parameters
        svals = material.strength_vol(temperatures)
        mvals = material.modulus_vol(temperatures)
        kvals = svals ** (-mvals)
        N = material.Nv(temperatures)
        B = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)
        Navg = np.mean(N, axis=0)
        Bavg = np.mean(B, axis=0)

        # Only tension
        pstress[pstress < 0] = 0

        # Max principal stresss over all time steps for each element
        pstress_max = np.max(pstress, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        g = (
            np.trapz(
                (pstress / (pstress_max + 1.0e-14)) ** Navg[..., None],
                time,
                axis=0,
            )
            / time[-1]
        )

        # Time dependent principal stress
        pstress_0 = (
            (pstress_max ** Navg[..., None] * g * tot_time) / Bavg[..., None]
            + (pstress_max ** (Navg[..., None] - 2))
        ) ** (1 / (Navg[..., None] - 2))

        return -kavg * np.sum(pstress_0 ** mavg[..., None], axis=-1) * volumes

    def calculate_surface_element_log_reliability(
        self,
        time,
        mandel_stress,
        surface,
        normals,
        temperatures,
        surface_areas,
        material,
        tot_time,
    ):
        """
        Calculate the element log reliability from surface elements only

        Parameters:
          time:             time instance variable for each stress/temperature
          mandel_stress:    element stresses in Mandel convention
          temperatures:     element temperatures
          volumes:          element volumes
          material:         material model object with required data that includes
                            Weibull scale parameter (svals), Weibull modulus (mvals)
                            and fatigue parameters (Bv,Nv)
          tot_time:         total service time used as input to calculate reliability
          surface_normals:  np.array of surface normals (zeros for solids)
          surface_elements: logical indexing array, True if on surface
        """

        # Material parameters
        svals = material.strength_surf(temperatures)
        mvals = material.modulus_surf(temperatures)
        kvals = svals ** (-mvals)
        N = material.Ns(temperatures)
        B = material.Bs(temperatures)

        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface
        # surface_areas = areas
        count_surface_elements = np.count_nonzero(surface_elements)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)[:count_surface_elements]
        kavg = np.mean(kvals, axis=0)[:count_surface_elements]
        Navg = np.mean(N, axis=0)[:count_surface_elements]
        Bavg = np.mean(B, axis=0)[:count_surface_elements]

        # Principal stresses in surface elements
        surf_pstress = self.calculate_surface_principal_stress(
            mandel_stress, surface_elements, surface_normals
        )

        # Only tension
        surf_pstress[surf_pstress < 0] = 0

        # Sorting
        surf_pstress = np.sort(surf_pstress, axis=-1)[..., ::-1]

        # Max principal surface stresss over all time steps for each element
        surf_pstress_max = np.max(surf_pstress, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        g = (
            np.trapz(
                (surf_pstress / (surf_pstress_max + 1.0e-14)) ** Navg[..., None, None],
                time,
                axis=0,
            )
            / time[-1]
        )

        # Time dependent principal stress
        surf_pstress_0 = (
            (surf_pstress_max ** Navg[..., None, None] * g * tot_time)
            / Bavg[..., None, None]
            + (surf_pstress_max ** (Navg[..., None, None] - 2))
        ) ** (1 / (Navg[..., None, None] - 2))

        # Summing up over last two axes i.e. over the elements of surface stress and
        # surfaces for which normals are there
        return (
            -kavg
            * np.sum(surf_pstress_0 ** mavg[..., None, None], axis=(-1, -2))
            * surface_areas
        )


class WNTSAModel(CrackShapeIndependent):
    """
    Weibull normal tensile average failure model

    Evaluates an average normal tensile stress (acting on the cracks) integrated over the
    area of a sphere, and uses it to calculate the element reliability
    """

    def calculate_volume_avg_normal_stress(
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

        mvals = material.modulus_vol(temperatures)
        N = material.Nv(temperatures)
        B = material.Bv(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        Navg = np.mean(N, axis=0)
        Bavg = np.mean(B, axis=0)

        # Time dependent Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

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
                    (sigma_n / (sigma_n_max + self.tolerance)) ** Navg[..., None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_n_0 = (
                (
                    ((sigma_n_max ** Navg[..., None, None]) * g * tot_time)
                    / Bavg[..., None, None]
                )
                + (sigma_n_max ** (Navg[..., None, None] - 2))
            ) ** (1 / (Navg[..., None, None] - 2))

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

    def calculate_surface_avg_normal_stress(
        self,
        time,
        mandel_stress,
        surface_elements,
        surface_normals,
        temperatures,
        material,
        tot_time,
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

        mvals = material.modulus_surf(temperatures)
        N = material.Ns(temperatures)
        B = material.Bs(temperatures)

        # Surface normals and surface elements
        normals = surface_normals
        surface = surface_elements
        count_surface_elements = np.count_nonzero(surface_elements)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)[:count_surface_elements]
        Navg = np.mean(N, axis=0)[:count_surface_elements]
        Bavg = np.mean(B, axis=0)[:count_surface_elements]

        # Time dependent Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

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
                    ** Navg[..., None, None, None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent equivalent stress
            sigma_n_0 = (
                (
                    ((sigma_n_max ** Navg[..., None, None, None]) * g * tot_time)
                    / Bavg[..., None, None, None]
                )
                + (sigma_n_max ** (Navg[..., None, None, None] - 2))
            ) ** (1 / (Navg[..., None, None, None] - 2))

        integral = ((sigma_n_0 ** mavg[..., None, None, None]) * self.ddelta) / (
            2 * np.pi
        )

        # Flatten the last axis and calculate the mean of the positive values along that axis
        flat = integral.reshape(integral.shape[:-2] + (-1,))

        # Average stress from summing while ignoring NaN values
        return np.nansum(np.where(flat >= 0.0, flat, np.nan), axis=(-1, -2))

    def calculate_volume_element_log_reliability(
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
        svals = material.strength_vol(temperatures)
        mvals = material.modulus_vol(temperatures)
        kvals = svals ** (-mvals)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)

        # Polyaxial Batdorf crack density coefficient
        kpvals = (2 * mavg + 1) * kavg

        # Average normal tensile stress raied to exponent mv
        avg_nstress = (
            self.calculate_volume_avg_normal_stress(
                time,
                mandel_stress,
                self.temperatures,
                self.material,
                tot_time,
            )
        ) ** (1 / mavg)

        return -kpvals * (avg_nstress**mavg) * volumes

    def calculate_surface_element_log_reliability(
        self,
        time,
        mandel_stress,
        surface,
        normals,
        temperatures,
        surface_areas,
        material,
        tot_time,
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
        svals = material.strength_surf(temperatures)
        mvals = material.modulus_surf(temperatures)
        kvals = svals ** (-mvals)

        # Surface normals and surface elements
        surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)[:count_surface_elements]
        kavg = np.mean(kvals, axis=0)[:count_surface_elements]

        # Polyaxial Batdorf crack density coefficient
        kpvals = ((mavg * gamma(mavg) * np.sqrt(np.pi)) / (gamma(mavg + 0.5))) * kavg

        # Average normal tensile stress raied to exponent mv
        avg_nstress = (
            self.calculate_surface_avg_normal_stress(
                time,
                mandel_stress,
                surface_elements,
                surface_normals,
                self.temperatures,
                self.material,
                tot_time,
            )
        ) ** (1 / mavg)

        return -kpvals * (avg_nstress**mavg) * surface_areas


class MTSModelGriffithFlaw(CrackShapeDependent):
    """
    Maximum tensile stess failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the maximum tensile stress fracture criterion for a Griffith flaw
    """

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses
        """
        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (sigma_n + np.sqrt((sigma_n**2) + (tau**2)))

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)
        using the Weibull scale parameter (svals), Weibull modulus (mvals),
        poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus_vol(temperatures)

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

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the average normal tensile stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt((sigma_n**2) + ((tau / (1 - (0.5 * nu[..., None, None]))) ** 2))
        )

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
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
        mvals = self.material.modulus_vol(temperatures)

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

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses
        """
        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau**2))

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
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
        mvals = self.material.modulus_vol(temperatures)

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

    def calculate_surface_eq_stress(
        self, mandel_stress, surface, normals, temperatures, material
    ):
        """
        Calculate the equivalent stresses from the normal and shear stresses in surface elements
        """
        # Surface normals and surface elements
        # surface_normals = normals
        # surface_elements = surface

        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Shear stress
        tau = self.calculate_surface_shear_stress(mandel_stress, surface, normals)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau**2))

    def calculate_surface_kbar(self, temperatures, material, C, ddelta):
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
        mvals = self.material.modulus_surf(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Calculating kbar
        integral2 = 2 * (((np.cos(self.C)) ** mavg[..., None, None]) * self.ddelta)
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        # remove 1 None and take sum over axis=(1) only
        return kbar


class CSEModelPennyShapedFlaw(CrackShapeDependent):
    """
    Coplanar strain energy failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a penny shaped flaw
    """

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau / (1 - (0.5 * nu[..., None, None]))) ** 2)

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
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
        mvals = self.material.modulus_vol(temperatures)

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


class CSEModelGriffithNotch(CrackShapeDependent):
    """
    Coplanar strain energy failure model with a Griffith notch

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a penny shaped flaw
    """

    def calculate_surface_eq_stress(
        self, mandel_stress, surface, normals, temperatures, material
    ):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Surface normals and surface elements
        # surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)[:count_surface_elements]

        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Shear stress
        tau = self.calculate_surface_shear_stress(mandel_stress, surface, normals)

        # Projected equivalent stress
        return np.sqrt(
            (sigma_n**2) + (0.7951 / (1 - nu[..., None, None, None])) * tau**2
        )
        # remove 1 None

    def calculate_surface_kbar(self, temperatures, material, C, ddelta):
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
        mvals = self.material.modulus_surf(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                np.sqrt(
                    ((np.cos(self.C)) ** 4)
                    - (
                        (0.198775 * (np.sin(2 * self.C)) ** 2)
                        / (nu[..., None, None] - 1)
                    )
                )
                ** mavg[..., None, None]
            )
            * self.ddelta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))
        # remove 1 None and take sum over axis=(1) only
        return kbar


class SMMModelGriffithFlaw(CrackShapeDependent):
    """
    Shetty mixed-mod failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        cbar:   Shetty model empirical constant (cbar)
        """

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n + np.sqrt((sigma_n**2) + ((2 * tau / cbar[..., None, None]) ** 2))
        )

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
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
        mvals = self.material.modulus_vol(temperatures)

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

    def calculate_surface_eq_stress(
        self, mandel_stress, surface, normals, temperatures, material
    ):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        cbar:   Shetty model empirical constant (cbar)
        """

        # Surface normals and surface elements
        # surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)[:count_surface_elements]

        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Shear stress
        tau = self.calculate_surface_shear_stress(mandel_stress, surface, normals)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt((sigma_n**2) + (4 * (tau / cbar[..., None, None, None]) ** 2))
        )
        # removed 1 None

    def calculate_surface_kbar(self, temperatures, material, C, ddelta):
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
        mvals = self.material.modulus_vol(temperatures)

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
                        ((np.cos(self.C)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.C)) ** 4)
                            + (
                                ((np.sin(2 * self.C)) ** 2)
                                / (cbar[..., None, None] ** 2)
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * self.ddelta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))
        # remove 1 None and take sum over axis=(1) only
        return kbar


class SMMModelGriffithNotch(CrackShapeDependent):
    """
    Shetty mixed mode failure model with a Griffith Notch

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_surface_eq_stress(
        self, mandel_stress, surface, normals, temperatures, material
    ):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        nu:     Poisson ratio
        cbar:   Shetty model empirical constant (cbar)
        """

        # Surface normals and surface elements
        # surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Material parameters
        # nu = np.mean(material.nu(temperatures), axis=0)[:count_surface_elements]
        cbar = np.mean(material.c_bar(temperatures), axis=0)[:count_surface_elements]

        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Shear stress
        tau = self.calculate_surface_shear_stress(mandel_stress, surface, normals)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt(
                (sigma_n**2) + (3.1803 * (tau / cbar[..., None, None, None]) ** 2)
            )
        )

    def calculate_surface_kbar(self, temperatures, material, C, ddelta):
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
        mvals = self.material.modulus_surf(temperatures)

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
                        ((np.cos(self.C)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.C)) ** 4)
                            + (
                                0.795075
                                * ((np.sin(2 * self.C)) ** 2)
                                / (cbar[..., None, None] ** 2)
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * self.ddelta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))
        # remove 1 None and take sum over axis=(1) only
        return kbar


class SMMModelPennyShapedFlaw(CrackShapeDependent):
    """
    Shetty mixed mode failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_volume_eq_stress(self, mandel_stress, temperatures, material):
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
        sigma_n = self.calculate_volume_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_volume_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt(
                (sigma_n**2)
                + ((4 * tau / (cbar[..., None, None] * (2 - nu[..., None, None]))) ** 2)
            )
        )

    def calculate_volume_kbar(self, temperatures, material, A, dalpha, dbeta):
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
        mvals = self.material.modulus_vol(temperatures)

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


class SMMModelSemiCircularCrack(CrackShapeDependent):
    """
    Shetty mixed mode failure model with a semi-circular crack

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the Shetty mixed-mode fracture criterion
    for a Griffith flaw
    """

    def calculate_surface_eq_stress(
        self, mandel_stress, surface, normals, temperatures, material
    ):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        nu:     Poisson ratio
        cbar:   Shetty model empirical constant (cbar)
        """
        # Surface normals and surface elements
        # surface_normals = normals
        surface_elements = surface
        count_surface_elements = np.count_nonzero(surface_elements)

        # Material parameters
        # nu = np.mean(material.nu(temperatures), axis=0)[:count_surface_elements]
        cbar = np.mean(material.c_bar(temperatures), axis=0)[:count_surface_elements]

        # Normal stress
        sigma_n = self.calculate_surface_normal_stress(mandel_stress, surface, normals)

        # Shear stress
        tau = self.calculate_surface_shear_stress(mandel_stress, surface, normals)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt(
                (sigma_n**2) + (3.301 * (tau / cbar[..., None, None, None]) ** 2)
            )
        )

    def calculate_surface_kbar(self, temperatures, material, C, ddelta):
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
        mvals = self.material.modulus_surf(temperatures)

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
                        ((np.cos(self.C)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.C)) ** 4)
                            + (
                                0.82525
                                * ((np.sin(2 * self.C)) ** 2)
                                / (cbar[..., None, None] ** 2)
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * self.ddelta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))
        # remove 1 None and take sum over axis=(1) only
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
