output_filename = 'p1pt724_out'
p = 1.724

[Mesh]
      file ='model.e'
      uniform_refine = 1
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Modules]
  [TensorMechanics]
    [Master]
      [all]
        strain = SMALL
        add_variables = true
        new_system = true
        formulation = UPDATED
       # volumetric_locking_correction = true
        generate_output = 'cauchy_stress_xx cauchy_stress_yy cauchy_stress_zz cauchy_stress_xy '
                          'cauchy_stress_xz cauchy_stress_yz strain_xx strain_yy strain_zz strain_xy '
                          'strain_xz strain_yz'
      []
    []
  []
[]

[AuxVariables]
  [./volume]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_vm]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_maxPrincipal]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_midPrincipal]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_minPrincipal]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./volume]
    type = VolumeAux
    variable = volume
  [../]
  [./stress_vm]
      type = RankTwoScalarAux
      variable = stress_vm
      rank_two_tensor = cauchy_stress
      scalar_type = VonMisesStress
      execute_on = timestep_end
  [../]
  [./stress_max]
    type = RankTwoScalarAux
    rank_two_tensor = cauchy_stress
    variable = stress_maxPrincipal
    scalar_type = MaxPrincipal
  [../]
  [./stress_mid]
    type = RankTwoScalarAux
    rank_two_tensor = cauchy_stress
    variable = stress_midPrincipal
    scalar_type = MidPrincipal
  [../]
  [./stress_min]
    type = RankTwoScalarAux
    rank_two_tensor = cauchy_stress
    variable = stress_minPrincipal
    scalar_type = MinPrincipal
  [../]
[]


[Functions]
  [./P]
    type = ConstantFunction
    value = ${p}
  [../]
[]

[BCs]
    # [./support]
    #      type = DirichletBC
    #      preset = true
    #      variable = disp_y
    #      boundary = 'base_support'
    #      value = 0.0
    # [../]
     [./support]
           type = DirichletBC
           preset = true
           variable = disp_y
           boundary = 'point_support'
           value = 0.0
    [../]
    [./xfixed]
          type = DirichletBC
          preset = true
          variable = disp_x
          boundary = 'xfixed'
          value = 0.0
    [../]
    [./zfixed]
          type = DirichletBC
          preset = true
          variable = disp_z
          boundary = 'zfixed'
          value = 0.0
    [../]
    [./Pressure]
        [./Press]
           boundary = 'pres_surf'
           function = P
           displacements = 'disp_x disp_y disp_z'
        [../]
     [../]
[]

[Materials]
  [./stress]
    type = CauchyStressFromNEML
    database = 'ceramic.xml'
    model = 'elastic_model'
    block = '1 2 3'
  [../]
[]

[Preconditioning]
  [./pc]
    type = SMP
    full = True
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'

  l_max_its = 100
  l_tol = 1e-12
  nl_max_its = 15
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-6

  petsc_options = '-snes_converged_reason -ksp_converged_reason -snes_linesearch_monitor'
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu superlu_dist'
  line_search = 'none'

  end_time = 1.0
  dtmin = 0.001
  # dtmax = 1 #600
  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1
    cutback_factor = 0.4
    growth_factor = 2
    optimal_iterations = 10
    # force_step_every_function_point = true
    # timestep_limiting_function = ptime
  [../]
  # automatic_scaling = on
[]

[Outputs]
  exodus = true
  # console = true
  [./out]
    type = Exodus
    file_base = ${output_filename}
    # sync_times = '0 940 1880 2820 3760 4700 5640 6580 7520 8460 9400  10340  11280  12220  13160  14100 15040  15980  16920  17860  18800  19740  20680  21620 22560  23500'
    # sync_only = true
  [../]
[]

[Postprocessors]
  [./max_sig1]
    type = ElementExtremeValue
    variable = stress_maxPrincipal
    block = '1 2 3'
  [../]
  [./max_sig2]
    type = ElementExtremeValue
    variable = stress_midPrincipal
    block = '1 2 3'
  [../]
  [./max_sig3]
    type = ElementExtremeValue
    variable = stress_minPrincipal
    block = '1 2 3'
  [../]
  [./total_vol]
    type = ElementIntegralVariablePostprocessor
    variable = volume
    block = '1 2 3'
  [../]
[]
