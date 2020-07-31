import sys
sys.path.append('../..')

from multiprocessing.pool import ThreadPool as Pool
import time
import numpy as np
from srlife import receiver, thermal, materials, writers

agents = 8
substeps = 2
recfname_input="Receiver.hdf5"
recfname_output="Receiver_results.hdf5"
paraview_filename="Receiver_paraview"
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

## thermal properties (A740H)
temp=np.array([-10000, 296.15,373.15,473.15,573.15,673.15,773.15,873.15,973.15,1073.15,1173.15,1273.15,10000])
cond=np.array([10.2e-3, 10.2e-3,11.7e-3,13.0e-3,14.5e-3,15.7e-3,17.1e-3,18.4e-3,20.2e-3,22.1e-3,23.8e-3,25.4e-3,25.4e-3])
Cp=np.array([449,449,476,489,496,503,513,519,542,573,635,656,656])
rho=8072e-9
diff=cond/(rho*Cp)

## fluid property
hc_temp=np.array([-10000, 823.15, 848.15, 873.15, 898.15, 923.15, 948.15, 973.15, 998.15, 1023.15, 10000])
hc=np.array([8.01e-3, 8.01e-3, 8.18e-3, 8.35e-3, 8.51e-3, 8.65e-3, 8.78e-3, 8.89e-3, 8.98e-3, 9.04e-3, 9.04e-3])
hcdata={matname: (hc_temp, hc)}

## create material object (PiecewiseLinear)
materials.PiecewiseLinearThermalMaterial(matname,temp,cond,diff).save(thermalfname)
materials.PiecewiseLinearFluidMaterial(hcdata).save(fluidfname)
thermalmat=materials.ThermalMaterial.load(thermalfname)
fluidmat=materials.FluidMaterial.load(fluidfname)

# ## Load receiver
rec=receiver.Receiver.load(recfname_input)

def solvetube (n):
    start_time = time.time()
    np = n[0]
    nt = n[1]
    print("start solving panel#%s tube#%s" %(np, nt))
    tube=rec.panels[np].tubes[nt]
    temperatures=thermal.FiniteDifferenceImplicitThermalProblem(tube,thermalmat,fluidmat, substep = substeps).solve()
    tube.add_results("temperature",temperatures)
    print("solved panel#%s tube#%s in %f seconds" %(np, nt, (time.time()-start_time)))
    return (np, nt, tube)

if __name__=='__main__':
    start_time = time.time()
    for npanel in rec.panels:
        tubes_to_solve=[]
        if int(npanel)>0:
            rec=receiver.Receiver.load(recfname_output)
        for ntube in rec.panels[npanel].tubes:
            # if (npanel=='0' or npanel=='8') and (ntube == '16'):
            tubes_to_solve.append([npanel,ntube])
        with Pool(processes=agents) as pool:
            tubes_solved = pool.map(solvetube, tubes_to_solve)
        for tube_solved in tubes_solved:
            rec.panels[tube_solved[0]].tubes[tube_solved[1]]=tube_solved[2]
        receiver.Receiver.save(rec, recfname_output)

    print("Total simulation time: %f" %(time.time()-start_time))
    rec.write_vtk(paraview_filename)
