import numpy as np
from srlife import receiver, thermal, materials

recfname_input="Receiver.hdf5"
recfname_output="Receiver_results.hdf5"
matname= "A740H"
thermalfname= matname + "_thermal.xml"
fluidname= "Fluid"
fluidfname= fluidname + ".xml"

## units:
#   conductivity: W/(mm-K)
#   specific heat, Cp: J/(kg-K)
#   density, rho: kg/mm^3
#   convective heat transfer coefficient, hc: W/(mm^2-K)
#   Heat flux: W/mm^2
#   Temperature: K
#   length: mm

## thermal properties
temp=np.array([296.15,373.15,473.15,573.15,673.15,773.15,873.15,973.15,1073.15,1173.15,1273.15])
cond=np.array([10.2e-3,11.7e-3,13.0e-3,14.5e-3,15.7e-3,17.1e-3,18.4e-3,20.2e-3,22.1e-3,23.8e-3,25.4e-3])
Cp=np.array([449,476,489,496,503,513,519,542,573,635,656])
rho=7940e-9
diff=cond/(rho*Cp)

## fluid property
hc_temp=np.array([298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 448.15, 473.15, 498.15, 523.15, 548.15, 573.15, 598.15, 623.15,
648.15, 673.15, 698.15, 723.15, 748.15, 773.15, 798.15, 823.15, 848.15, 873.15, 898.15, 923.15, 948.15, 973.15, 998.15, 1023.15000000000])
hc=np.array([4.68022082578762e-3, 4.79442533471872e-3, 4.91296677131151e-3, 5.03600654094643e-3, 5.16369914778528e-3, 5.29618817509435e-3,
 5.43360129074526e-3, 5.57604409502619e-3, 5.72359260687874e-3, 5.87628416739174e-3, 6.03410652977966e-3, 6.19698490887863e-3,
 6.36476678840525e-3, 6.53720434156319e-3, 6.71393442379131e-3, 6.89445626215072e-3, 7.07810721264081e-3, 7.26403730289167e-3,
 7.45118373685509e-3, 7.63824711231155e-3, 7.82367177144808e-3, 8.00563341514446e-3, 8.18203776091720e-3, 8.35053445405494e-3,
 8.50855043912016e-3, 8.65334632379800e-3, 8.78209770455564e-3, 8.89200087152820e-3, 8.98039888000336e-3, 9.04492007432691e-3])
hcdata={matname: (hc_temp, hc)}

## create material object (PiecewiseLinear or Constant)
# PiecewiseLinear
# materials.PiecewiseLinearThermalMaterial(matname,temp,cond,diff).save(thermalfname)
# materials.PiecewiseLinearFluidMaterial(hcdata).save(fluidfname)
# thermalmat=materials.ThermalMaterial.load(thermalfname)
# fluidmat=materials.FluidMaterial.load(fluidfname)
# Constant
thermalmat=materials.ConstantThermalMaterial(matname, 11.7e-3, 3.09569672)
fluidmat=materials.ConstantFluidMaterial({matname: 4.79442533471872e-3})

## Load receiver
rec=receiver.Receiver.load(recfname_input)

## Solve for Temperature
for npanel in rec.panels:
    for ntube in rec.panels[npanel].tubes:
        print("npanel=%s  ntube=%s" %(npanel, ntube))
        tube=rec.panels[npanel].tubes[ntube]
        temperatures=thermal.FiniteDifferenceImplicitThermalProblem(tube,thermalmat,fluidmat).solve()
        tube.add_results("temperature", temperatures)
        # Comment next 5 lines to detemine temperature for all tubes
        if npanel== "0" and ntube== "0":
            break
    else:
        continue
    break

## Save receiver with result
receiver.Receiver.save(rec, recfname_output)
