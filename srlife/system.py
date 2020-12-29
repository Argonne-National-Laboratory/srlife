#pylint: disable=no-member

"""
  Complete receiver system structural solvers.  These operate
  under the assumption that you've already done the thermal problem.
"""

from abc import ABC, abstractmethod

import multiprocess

from srlife import spring, solverparams, receiver

class SystemSolver(ABC):
  """
    The system takes as input:
      1) A complete receiver, including temperatures
      2) The structural material

    It returns:
      1) The updated receiver, with all the structural results defined
  """
  @abstractmethod
  def solve(self, model, smat):
    """
      Solve the receiver system

      Parameters:
        model       receiver object
        smat        structural material
    """
    return

class SpringSystemSolver(SystemSolver):
  """
    Solve the receiver system using the network of springs model defined
    in the springs module
  """
  def __init__(self, pset = solverparams.ParameterSet(),
      rtol = 1.0e-6, atol = 1.0e-4, miter = 25, verbose = False):
    """
      Initialize the solver

      Additional parameters:
        pset        parameter dictionary overriding the default solver params
        rtol        relative tolerance
        atol        absolute tolerance
        miter       number of permissible nonlinear iterations
        verbose     print a lot of debug info
    """
    self.rtol = pset.get_default("rtol", rtol)
    self.atol = pset.get_default("atol", atol)
    self.miter = pset.get_default("miter", miter)
    self.verbose = pset.get_default("verbose", verbose)

  def solve(self, model, smat, ssolver, nthreads = 1,
      verbose = False, decorator = lambda fn, l: fn):
    """
      Solve a receiver model using a spring system

      Parameters:
        model           receiver model, with temperatures
        smat            structural material
        ssolver         structural solver to use

      Additional parameters:
        verbose         print stuff
        nthreads        number of threads to use
        decorator       progress decorator
    """
    network = self.make_network(model, smat, ssolver)
    subproblems = network.reduce_graph()

    # Simple heuristic for deciding who gets threads
    nprobs = len(subproblems)
    max_sub = max(sum(1 for i,j,data in sb.edges(data=True) 
      if isinstance(data['object'], spring.TubeSpring)) for sb in
      subproblems)

    if nprobs < max_sub:
      results = []
      if verbose:
        print("Solving substructures sequentially")
      for i,subproblem in enumerate(subproblems):
        if verbose:
          print("Solving substructure %i of %i" % (i+1, len(subproblems)))
        results.append(subproblem.solve_all(nthreads, decorator = decorator))
    else:
      sfn = lambda x: x.solve_all(1)
      if verbose: 
        print("Solving subproblems in parallel")
      with multiprocess.Pool(nthreads) as p:
        results = list(decorator(p.imap(sfn, subproblems), len(subproblems)))

    # Sigh, now need to copy the tube data into the correct location...
    for new, orig in zip(results, subproblems):
      for i,j,k in new.edges(keys=True):
        if isinstance(orig[i][j][k]['object'], spring.TubeSpring):
          orig[i][j][k]['object'].tube.copy_results(new[i][j][k]['object'].tube)

  def make_network(self, model, smat, ssolver):
    """
      Setup the complete spring network, given the receiver and
      material objects

      Parameters:
        model           fully-defined receiver object
        smat            structural material model
        ssolver         structural solver to use
    """
    network = spring.SpringNetwork(atol = self.atol, rtol = self.rtol,
        miter = self.miter, verbose = self.verbose)
    cn = 0
    network.add_node(cn)
    cn += 1
    for panel in model.panels.values():
      network.add_node(cn)
      cn += 1
      network.add_edge(0, cn-1, object = convert_to_spring(
        model.stiffness, smat, ssolver))
      top = cn-1
      for tube in panel.tubes.values():
        network.add_node(cn)
        cn += 1
        network.add_edge(top, cn-1, object = convert_to_spring(
          panel.stiffness, smat, ssolver))
        network.add_node(cn)
        cn += 1
        network.add_edge(cn-2, cn-1, object = convert_to_spring(tube,
          smat, ssolver))
        network.displacement_bc(cn-1, lambda t: 0.0)
    
    network.validate_setup()

    return network

def convert_to_spring(thing, smat, ssolver):
  """
    Translate an object into a spring

    Parameters:
      thing         object to convert
      smat          structural material
      ssolver       structural solver
  """
  if isinstance(thing, (float,int)):
    return spring.LinearSpring(thing)
  elif isinstance(thing, str):
    if thing not in ["disconnect", "rigid"]:
      raise ValueError("Special spring types are either 'disconnect' or 'rigid'!")
    return thing
  elif isinstance(thing, receiver.Tube):
    return spring.TubeSpring(thing, ssolver, smat)
  else:
    raise ValueError("Cannot convert object to spring!")
