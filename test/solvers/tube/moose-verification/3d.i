[Mesh]
  file = 'meshes/3d.e'
[../]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Variables]
      [./disp_x]
      [../]
      [./disp_y]
      [../]
      [./disp_z]
      [../]
[]


[Kernels]
      [./TensorMechanics]
      [../]
[]

[AuxVariables]
  [./temperature]
    initial_condition = 0
  [../]
  [./mechanical_strain_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./mechanical_strain_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./mechanical_strain_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./mechanical_strain_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./mechanical_strain_xz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./mechanical_strain_yz]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./thermal_strain_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./thermal_strain_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./thermal_strain_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./thermal_strain_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./thermal_strain_xz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./thermal_strain_yz]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./strain_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_xz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_yz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_yz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_xz]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./temperature]
    type = FunctionAux
    variable = temperature
    function = temperature
  [../]
  [./stress_xx]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xx
    index_i = 0
    index_j = 0
    execute_on = timestep_end
  [../]
  [./stress_yy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yy
    index_i = 1
    index_j = 1
    execute_on = timestep_end
  [../]
  [./stress_zz]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_zz
    index_i = 2
    index_j = 2
    execute_on = timestep_end
  [../]
  [./stress_xy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xy
    index_i = 0
    index_j = 1
    execute_on = timestep_end
  [../]
  [./stress_xz]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xz
    index_i = 0
    index_j = 2
    execute_on = timestep_end
  [../]
  [./stress_yz]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yz
    index_i = 1
    index_j = 2
    execute_on = timestep_end
  [../]

  [./strain_xx]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_xx
    index_i = 0
    index_j = 0
    execute_on = timestep_end
  [../]
  [./strain_yy]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_yy
    index_i = 1
    index_j = 1
    execute_on = timestep_end
  [../]
  [./strain_zz]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_zz
    index_i = 2
    index_j = 2
    execute_on = timestep_end
  [../]
  [./strain_xy]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_xy
    index_i = 0
    index_j = 1
    execute_on = timestep_end
  [../]
  [./strain_xz]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_xz
    index_i = 0
    index_j = 2
    execute_on = timestep_end
  [../]
  [./strain_yz]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_yz
    index_i = 1
    index_j = 2
    execute_on = timestep_end
  [../]

  [./mechanical_strain_xx]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_xx
    index_i = 0
    index_j = 0
    execute_on = timestep_end
  [../]
  [./mechanical_strain_yy]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_yy
    index_i = 1
    index_j = 1
    execute_on = timestep_end
  [../]
  [./mechanical_strain_zz]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_zz
    index_i = 2
    index_j = 2
    execute_on = timestep_end
  [../]
  [./mechanical_strain_xy]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_xy
    index_i = 0
    index_j = 1
    execute_on = timestep_end
  [../]
  [./mechanical_strain_xz]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_xz
    index_i = 0
    index_j = 2
    execute_on = timestep_end
  [../]
  [./mechanical_strain_yz]
    type = RankTwoAux
    rank_two_tensor = mechanical_strain
    variable = mechanical_strain_yz
    index_i = 1
    index_j = 2
    execute_on = timestep_end
  [../]

  [./thermal_strain_xx]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_xx
    index_i = 0
    index_j = 0
    execute_on = timestep_end
  [../]
  [./thermal_strain_yy]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_yy
    index_i = 1
    index_j = 1
    execute_on = timestep_end
  [../]
  [./thermal_strain_zz]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_zz
    index_i = 2
    index_j = 2
    execute_on = timestep_end
  [../]
  [./thermal_strain_xy]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_xy
    index_i = 0
    index_j = 1
    execute_on = timestep_end
  [../]
  [./thermal_strain_xz]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_xz
    index_i = 0
    index_j = 2
    execute_on = timestep_end
  [../]
  [./thermal_strain_yz]
    type = RankTwoAux
    rank_two_tensor = eigenstrain
    variable = thermal_strain_yz
    index_i = 1
    index_j = 2
    execute_on = timestep_end
  [../]

[]

[Functions]
  [./ramp]
    type = PiecewiseLinear
    x = '0 1 1001'
    y = '0 1 1'
  [../]
  [./unit_temperature]
    type = ParsedFunction
    value = '(sqrt(x^2+y^2)-9)/(10-9) * cos(atan(y/x))*(z/10 + 1)'
  [../]
  [./temperature]
    type = CompositeFunction
    functions = 'ramp unit_temperature'
    scale_factor = 100.0
  [../]
  [./pressure]
    type = CompositeFunction
    functions = 'ramp'
    scale_factor = 1.0
  [../]
[]

[BCs]
  [./bot]
    type = DirichletBC
    variable = disp_z
    boundary = bottom
    value = 0.0
  [../]

  [./top]
    type = DirichletBC
    variable = disp_z
    boundary = top
    value = 0.0
  [../]

  [./xy_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'three'
    value = 0.0
  [../]

  [./xy_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'three'
    value = 0.0
  [../]

  [./x]
    type = DirichletBC
    variable = disp_y
    boundary = 'nine'
    value = 0.0
  [../]

  [./Pressure]
    [./inner]
      boundary = pressure
      function = pressure
      displacements = 'disp_x disp_y disp_z'
    [../]
  [../]
[]

[Materials]
  [./strain]
    type = ComputeSmallStrain
    block = 'vessel'
    eigenstrain_names = eigenstrain
  [../]

  [./stress]
    type = ComputeNEMLStress
    database = 'model.xml'
    model = 'creeping'
    block = 'vessel'
    temperature = temperature
  [../]

  [./thermal_strain]
    type = ComputeThermalExpansionEigenstrainNEML
    database = 'model.xml'
    model = 'creeping'
    temperature = temperature
    eigenstrain_name = eigenstrain
    block = 'vessel'
    stress_free_temperature = 0
  [../]
[../]

[Preconditioning]
  [./SMP]
    type = SMP
    full = true
  [../]
[]

[Executioner]
  type = Transient
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'

  solve_type = NEWTON

  end_time = 1001

  nl_abs_tol = 1.0e-8

  [./TimeStepper]
    type = TimeSequenceStepper
    time_sequence  = '0 0.5 1 101 201 301 401 501 601 701 801 901 1001'
  [../]

  line_search = none
[]

[Outputs]
  [./out]
    type = Exodus
  [../]
[]
