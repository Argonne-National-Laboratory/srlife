"""
  Module with methods for calculating creep-fatigue damage given 
  completely-solved tube results and damage material properties
"""

import numpy as np
import scipy.optimize as opt

import multiprocess

class DamageCalculator:
  """
    Parent class for all damage calculators, handling common iteration
    and scaling options
  """
  def __init__(self):
    pass

  def single_cycles(self, tube, material, receiver):
    """
      Calculate damage for a single tube

      Parameters:
        tube:       fully-populated tube object
        material:   damage material
        receiver:   receiver object (for metadata)
    """ 
    raise NotImplementedError("Superclass not implemented")

  def determine_life(self, receiver, material, nthreads = 1, 
      decorator = lambda x, n: x):
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
      Ns = list(decorator(
        p.imap(lambda x: self.single_cycles(x, material, receiver), receiver.tubes),
        receiver.ntubes))
    N = min(Ns)

    # Translate to days
    return N * receiver.days

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
    # Material point creep damage
    Dc = self.creep_damage(tube, material)

    # Material point fatigue damage
    Df = self.fatigue_damage(tube, material, receiver)

    # This is going to be expensive, but I don't see much way around it
    return min(self.calculate_max_cycles(c, f, material) for c,f in 
        zip(Dc.flatten(), Df.flatten()))
  
  def calculate_max_cycles(self, Dc, Df, material, rep_min = 1, rep_max = 1e6):
    """
      Actually calculate the maximum number of repetitions for a single point

      Parameters:
        Dc          creep damage per simulated cycle
        Df          fatigue damage per simulated cycle
        material    damaged material properties
    """
    if np.isclose(Dc, 0) and np.isclose(Df, 0):
      return np.inf

    if not material.inside_envelope("cfinteraction", rep_min * Df, rep_min*Dc):
      return 0

    if material.inside_envelope("cfinteraction", rep_max*Df, rep_max * Dc):
      return np.inf

    return opt.brentq(lambda N: material.inside_envelope("cfinteraction", N* Df, N*Dc) - 0.5,
        rep_min, rep_max)

  def creep_damage(self, tube, material):
    """
      Calculate creep damage at each material point

      Parameters:
        tube        single tube with full results
        material    damage material model
    """
    # For now just use the von Mises effective stress
    vm = np.sqrt((
        (tube.quadrature_results['stress_xx'] - tube.quadrature_results['stress_yy'])**2.0 + 
        (tube.quadrature_results['stress_yy'] - tube.quadrature_results['stress_zz'])**2.0 + 
        (tube.quadrature_results['stress_zz'] - tube.quadrature_results['stress_xx'])**2.0 + 
        3.0 * (tube.quadrature_results['stress_xy']**2.0 + 
          tube.quadrature_results['stress_yz']**2.0 + 
          tube.quadrature_results['stress_xz']**2.0))/2.0)

    tR = material.time_to_rupture("averageRupture", tube.quadrature_results['temperature'], vm)
    dts = np.diff(tube.times)
    dmg = np.sum(dts[:,np.newaxis,np.newaxis]/tR[1:], axis = 0)

    return dmg

  def fatigue_damage(self, tube, material, receiver):
    """
      Calculate fatigue damage at each material point

      Parameters:
        tube        single tube with full results
        material    damage material model
        receiver    receiver, for metadata
    """
    # Identify cycle boundaries
    tm = np.mod(tube.times, receiver.period)
    inds = list(np.where(tm == 0)[0])
    if len(inds) != (receiver.days + 1):
      raise ValueError("Tube times not compatible with the receiver"
          " number of days and cycle period!")

    # Run through each cycle and ID max strain range and fatigue damage
    strain_names = ['mechanical_strain_xx', 'mechanical_strain_yy', 'mechanical_strain_zz',
        'mechanical_strain_yz', 'mechanical_strain_xz', 'mechanical_strain_xy']
    strain_factors = [1.0,1.0,1.0,np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    
    return sum(self.cycle_fatigue(np.array([ef*tube.quadrature_results[en][
      inds[i]:inds[i+1]] for 
      en,ef in zip(strain_names, strain_factors)]), 
      tube.quadrature_results['temperature'][inds[i]:inds[i+1]], material)
      for i in range(receiver.days))
  
  def cycle_fatigue(self, strains, temperatures, material, nu = 0.5):
    """
      Calculate fatigue damage for a single cycle

      Parameters:
        strains         single cycle strains
        temperatures    single cycle temperatures
        material        damage model

      Additional parameters:
        nu              effective Poisson's ratio to use
    """
    pt_temps = np.max(temperatures, axis = 0)

    pt_eranges = np.zeros(pt_temps.shape)
    nt = strains.shape[0]
    for i in range(nt):
      for j in range(nt):
        de = strains[:,j] - strains[:,i]
        eq = np.sqrt(2) / (2*(1+nu)) * np.sqrt(
            (de[0] - de[1])**2 + (de[1]-de[2])**2 + (de[2]-de[0])**2.0
            + 3.0/2.0 * (de[3]**2.0 + de[4]**2.0 + de[5]**6.0)
            )
        pt_eranges = np.maximum(pt_eranges, eq)
    
    dmg = np.zeros(pt_eranges.shape)
    # pylint: disable=not-an-iterable
    for ind in np.ndindex(*dmg.shape):
      dmg[ind] = 1.0 / material.cycles_to_fail("nominalFatigue", pt_temps[ind], pt_eranges[ind])

    return dmg
