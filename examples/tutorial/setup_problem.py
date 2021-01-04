#!/usr/bin/env python3

import numpy as np

import sys
sys.path.append('../..')

from srlife import receiver

if __name__ == "__main__":
  # Setup the base receiver
  period = 24.0 # Loading cycle period, hours
  days = 1 # Number of cycles represented in the problem 
  panel_stiffness = "disconnect" # Panels are disconnected from one another

  model = receiver.Receiver(period, days, panel_stiffness)

  # Setup each of the two panels
  tube_stiffness = "rigid"
  panel_0 = receiver.Panel(tube_stiffness)
  panel_1 = receiver.Panel(tube_stiffness)

  # Basic receiver geometry
  r_outer = 12.7 # mm
  thickness = 1.0 # mm
  height = 5000.0 # mm

  # Tube discretization
  nr = 12
  nt = 20
  nz = 10

  # Mathematical definition of the tube boundary conditions
  # Function used to define daily operating cycle 
  onoff_base = lambda t: np.sin(np.pi*t/12.0)
  onoff = lambda t: (1+np.sign(onoff_base(t)))/2 * onoff_base(t)
  # Max flux
  h_max = 0.6 # W/mm^2 (which is also MW/m^2)
  # Flux circumferential component
  h_circ = lambda theta: np.cos(theta)
  # Flux axial component
  h_axial = lambda z: (1+np.sin(np.pi*z/height))/2
  # Total flux function
  h_flux = lambda time, theta, z: onoff(time) * h_max * h_circ(theta) * h_axial(z)
  
  # Flux multipliers for each tube
  h_tube_0 = 1.0
  h_tube_1 = 0.8
  h_tube_2 = 0.6
  h_tube_3 = 0.4

  # ID fluid temperature histories for each tube
  delta_T = 50 # K
  T_base = 550 + 273.15 # K
  fluid_temp = lambda t, z: delta_T * onoff(t) * z/height + T_base

  # ID pressure history
  p_max = 1.0 # MPa
  pressure = lambda t: p_max * onoff(t)

  # Time increments throughout the 24 hour day
  times = np.linspace(0,24,24*2+1)

  # Various meshes needed to define the boundary conditions
  # 1) A mesh over the times and height (for the fluid temperatures)
  time_h, z_h = np.meshgrid(times, np.linspace(0,height,nz), indexing='ij')
  # 2) A surface mesh over the outer surface (for the flux)
  time_s, theta_s, z_s = np.meshgrid(times, np.linspace(0,2*np.pi,nt+1)[:nt],
      np.linspace(0,height,nz), indexing = 'ij')

  # Setup each tube in turn and assign it to the correct panel
  # Tube 0
  tube_0 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
  tube_0.set_times(times)
  tube_0.set_bc(receiver.ConvectiveBC(r_outer-thickness,
    height, nz, times, fluid_temp(time_h,z_h)), "inner")
  tube_0.set_bc(receiver.HeatFluxBC(r_outer, height,
    nt, nz, times, h_flux(time_s, theta_s, z_s) * h_tube_0), "outer")
  tube_0.set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  # Tube 1
  tube_1 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
  tube_1.set_times(times)
  tube_1.set_bc(receiver.ConvectiveBC(r_outer-thickness,
    height, nz, times, fluid_temp(time_h,z_h)), "inner")
  tube_1.set_bc(receiver.HeatFluxBC(r_outer, height,
    nt, nz, times, h_flux(time_s, theta_s, z_s) * h_tube_1), "outer")
  tube_1.set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  # Tube 2
  tube_2 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
  tube_2.set_times(times)
  tube_2.set_bc(receiver.ConvectiveBC(r_outer-thickness,
    height, nz, times, fluid_temp(time_h,z_h)), "inner")
  tube_2.set_bc(receiver.HeatFluxBC(r_outer, height,
    nt, nz, times, h_flux(time_s, theta_s, z_s) * h_tube_2), "outer")
  tube_2.set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  # Tube 3
  tube_3 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
  tube_3.set_times(times)
  tube_3.set_bc(receiver.ConvectiveBC(r_outer-thickness,
    height, nz, times, fluid_temp(time_h,z_h)), "inner")
  tube_3.set_bc(receiver.HeatFluxBC(r_outer, height,
    nt, nz, times, h_flux(time_s, theta_s, z_s) * h_tube_3), "outer")
  tube_3.set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  # Assign to panel 0
  panel_0.add_tube(tube_0, "tube0")
  panel_0.add_tube(tube_1, "tube1")

  # Assign to panel 1
  panel_1.add_tube(tube_2, "tube2")
  panel_1.add_tube(tube_3, "tube3")

  # Assign the panels to the receiver
  model.add_panel(panel_0, "panel0")
  model.add_panel(panel_1, "panel1")

  # Save the receiver to an HDF5 file
  model.save("model.hdf5")
