# Weirdness with pylint and vtk (some dynamic import thing going on there)
# pylint: disable=no-member
"""
  Module for various ways to write receiver data structures to disk in a human-readable way
"""
import vtk

import numpy as np

class VTKWriter:
  """
    Write a single tube file to VTK
  """
  def __init__(self, tube, fname):
    """
      Parameters:
        tube        tube object
        fname       base filename to use
    """
    self.tube = tube
    self.fname = fname

  def write(self):
    """
      Actually write the tube object to a vtk file using a
      VTKStructuredGrid
    """
    grid = vtk.vtkUnstructuredGrid()

    R, T, Z = self.tube.mesh
    points = vtk.vtkPoints()

    X = R * np.cos(T)
    Y = R * np.sin(T)

    for x,y,z in zip(X.flatten(), Y.flatten(), Z.flatten()):
      points.InsertNextPoint(x,y,z)

    grid.SetPoints(points)
    
    self._set_grid(grid)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(self.fname + ".vtu")
    writer.SetInputData(grid)  
    writer.SetNumberOfTimeSteps(self.tube.ntime)

    writer.Start()
    for i in range(self.tube.ntime):
      self._dump_point_data(grid, i)
      self._dump_element_data(grid, i)
      writer.WriteNextTime(self.tube.times[i])
    writer.Stop()
  
  def _dump_point_data(self, grid, i):
    """
      Dump all our point data to the grid

      Parameters:
        grid        vtk unstructured grid
        i           time step
    """
    for field, data in self.tube.results.items():
      pdata = vtk.vtkFloatArray()
      pdata.SetNumberOfComponents(1)
      pdata.SetName(field)
      for d in data[i].flatten():
        pdata.InsertNextValue(d)
      grid.GetPointData().AddArray(pdata)

  def _dump_element_data(self, grid, i):
    for field, data in self.tube.quadrature_results.items():
      pdata = vtk.vtkFloatArray()
      pdata.SetNumberOfComponents(1)
      pdata.SetName(field)
      avg = np.mean(data[i], axis = 1)
      for d in avg:
        pdata.InsertNextValue(d)
      grid.GetCellData().AddArray(pdata)
    
  def _set_grid(self, grid):
    """
      Setup the VTK unstructured grid for the mesh

      Parameters:
        grid        vtk unstructured grid
    """
    if self.tube.ndim == 1:
      self._set_grid_1d(grid)
    elif self.tube.ndim == 2:
      self._set_grid_2d(grid)
    else:
      self._set_grid_3d(grid)

  def _set_grid_1d(self, grid):
    """
      Setup the VTK unstructured grid for a 1d mesh

      Parameters:
        grid        vtk unstructured grid
    """
    flat = lambda i: i
    for i in range(self.tube.nr-1):
      cell = vtk.vtkLine()
      for nk, a in enumerate((0,1)):
        cell.GetPointIds().SetId(nk, flat(i+a))
      grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

  def _set_grid_2d(self, grid):
    """
      Setup the VTK unstructured grid for a 1d mesh

      Parameters:
        grid        vtk unstructured grid
    """
    flat = lambda i, j: i * self.tube.nt + (j % self.tube.nt)
    for i in range(self.tube.nr-1):
      for j in range(self.tube.nt):
        cell = vtk.vtkQuad()
        for nk, (a,b) in enumerate(((0,0),(0,1),(1,1),(1,0))):
          cell.GetPointIds().SetId(nk, flat(i+a,j+b))
        grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

  def _set_grid_3d(self, grid):
    """
      Setup the VTK unstructured grid for a 1d mesh

      Parameters:
        grid        vtk unstructured grid
    """
    flat = lambda i, j, k: i * self.tube.nt * self.tube.nz + (j % self.tube.nt) * self.tube.nz + k
    for i in range(self.tube.nr-1):
      for j in range(self.tube.nt):
        for k in range(self.tube.nz-1):
          cell = vtk.vtkHexahedron()
          for nk, (a,b,c) in enumerate(
              ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))):
            cell.GetPointIds().SetId(nk,flat(i+a,j+b,k+c))
          grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
