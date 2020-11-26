import unittest

import numpy as np

from srlife import spring

class TestSimple(unittest.TestCase):
  """
    Simple test cases with elastic springs
  """
  def test_single(self):
    network = spring.SpringNetwork()
    network.add_node(0)
    network.add_node(1)
    network.add_edge(0,1, object = spring.LinearSpring(100))
    
    network.set_times([0,1])

    network.validate_setup()

    network.displacement_bc(0, lambda t: 0.0)
    network.force_bc(1, lambda t: 100.0*t)

    network.solve_all()

    self.assertTrue(np.allclose(network.displacements,
      [0,1]))

  def test_system(self):
    k1 = 100.0
    k2 = 200.0
    k3 = 150.0
    f = 100.0

    network = spring.SpringNetwork()
    network.add_node(0)
    network.add_node(1)
    network.add_node(2)
    network.add_edge(0,1, object = spring.LinearSpring(k1))
    network.add_edge(1,2, object = spring.LinearSpring(k2))
    network.add_edge(1,2, object = spring.LinearSpring(k3))
    
    network.set_times([0,1])

    network.validate_setup()

    network.displacement_bc(2, lambda t: 0.0)
    network.force_bc(0, lambda t: t*f)

    network.solve_all()

    kp = k2 + k3
    dxp = f / kp
    dx1 = f / k1
    
    exact = np.array([dx1+dxp,dxp,0])

    self.assertTrue(np.allclose(network.displacements,exact))

  def test_system_thrads(self):
    k1 = 100.0
    k2 = 200.0
    k3 = 150.0
    f = 100.0

    network = spring.SpringNetwork()
    network.add_node(0)
    network.add_node(1)
    network.add_node(2)
    network.add_edge(0,1, object = spring.LinearSpring(k1))
    network.add_edge(1,2, object = spring.LinearSpring(k2))
    network.add_edge(1,2, object = spring.LinearSpring(k3))
    
    network.set_times([0,1])

    network.validate_setup()

    network.displacement_bc(2, lambda t: 0.0)
    network.force_bc(0, lambda t: t*f)

    network.solve_all(nthreads=2)

    kp = k2 + k3
    dxp = f / kp
    dx1 = f / k1
    
    exact = np.array([dx1+dxp,dxp,0])

    self.assertTrue(np.allclose(network.displacements,exact))

  def test_remove_rigid(self):
    network = spring.SpringNetwork()
    k = 100.0
    
    for i in range(6):
      network.add_node(i)

    network.add_edge(0,1, object = "rigid")
    network.add_edge(1,3, object = spring.LinearSpring(k))
    network.add_edge(1,3, object = spring.LinearSpring(k))
    network.add_edge(2,4, object = "rigid")
    network.add_edge(3,5, object = spring.LinearSpring(k))
    network.add_edge(4,5, object = spring.LinearSpring(k))
    
    network.displacement_bc(4, lambda x: 0)

    network.set_times([0,1])
    network.validate_setup()
    
    network.remove_rigid()
    network.validate_solve()
    
    self.assertEqual(len(network.nodes), 4)
    self.assertEqual(len(network.edges), 4)

  def test_disconnect(self):
    network = spring.SpringNetwork()
    k = 100.0
    
    for i in range(8):
      network.add_node(i)

    network.add_edge(0, 1, object = spring.LinearSpring(k))
    network.add_edge(0, 2, object = spring.LinearSpring(k))

    network.add_edge(1, 3, object = spring.LinearSpring(k))
    network.add_edge(1, 3, object = spring.LinearSpring(k))
    network.add_edge(2, 4, object = "disconnect")

    network.add_edge(3, 5, object = spring.LinearSpring(k))
    network.add_edge(4, 6, object = spring.LinearSpring(k))
    network.add_edge(4,7, object = spring.LinearSpring(k))

    
    network.displacement_bc(0, lambda x: 0)
    network.displacement_bc(4, lambda x: 0)

    network.set_times([0,1])
    network.validate_setup()

    subs = network.reduce_graph()

    self.assertEqual(len(subs), 2)

    for sub in subs:
      sub.validate_solve()
