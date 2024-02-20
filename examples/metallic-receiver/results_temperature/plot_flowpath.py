#!/usr/bin/env python3

import numpy as np

import sys
import matplotlib.pyplot as plt
import os.path

sys.path.append("../..")

from srlife import (
    receiver,
    solverparams,
    spring,
    structural,
    thermal,
    system,
    library,
    managers,
    damage,
)


if __name__ == "__main__":

    flow_type = sys.argv[1] #'constant' or 'variable'

    results_fname = os.path.join("example-with-results-"+str(flow_type)+"_flow.hdf5")
    model = receiver.Receiver.load(results_fname)

    for pi, panel in model.panels.items():
        for ti, tube in panel.tubes.items():
            print("panel:%s, tube:%s, T:%.2f"%(pi, ti, tube.results.get('temperature').max()-273.15))
            if int(ti)==0:
                if int(pi)==0:
                    Tfluid_flowpath_t0 = tube.axial_results.get('fluid_temperature')
                    Ttube_max_t0 = tube.results.get('temperature').max(axis=1).max(axis=1)
                else:
                    Tfluid_flowpath_t0 = np.hstack([Tfluid_flowpath_t0,tube.axial_results.get('fluid_temperature')])
                    Ttube_max_t0 = np.hstack([Ttube_max_t0, tube.results.get('temperature').max(axis=1).max(axis=1)])
            if int(ti)==1:
               if int(pi)==0:
                   Tfluid_flowpath_t1 = tube.axial_results.get('fluid_temperature')
                   Ttube_max_t1 = tube.results.get('temperature').max(axis=1).max(axis=1)
               else:
                   Tfluid_flowpath_t1 = np.hstack([Tfluid_flowpath_t1,tube.axial_results.get('fluid_temperature')])
                   Ttube_max_t1 = np.hstack([Ttube_max_t1, tube.results.get('temperature').max(axis=1).max(axis=1)])

    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(6,6))

    for i in np.arange(Tfluid_flowpath_t0.shape[0]/2).astype(int):
        color = np.random.rand(3)
        xp = np.arange(len(Tfluid_flowpath_t0[i,:]))/len(Tfluid_flowpath_t0[i,:])*int(pi)
        axs[0].plot(xp, Tfluid_flowpath_t0[i,:]-273.15,linestyle='dashed',linewidth=1.5,color=color)
        axs[1].plot(xp, Tfluid_flowpath_t1[i,:]-273.15,linestyle='dashed',linewidth=1.5,color=color)
        axs[0].plot(xp, Ttube_max_t0[i,:]-273.15,linestyle='solid',linewidth=1.5,color=color,label='%s hr'%tube.times[i])
        axs[1].plot(xp, Ttube_max_t1[i,:]-273.15,linestyle='solid',linewidth=1.5,color=color,label='%s hr'%tube.times[i])

    for ax in enumerate(axs.flat):
        ax[1].set(xlabel = 'Flowpath (panels)', ylabel = 'Temperature (C$\degree$)', title = 'Tube: %s'%(ax[0]))
        ax[1].legend(fontsize=8)
        ax[1].text(7,400,'solid: Tube crown', fontsize=10)
        ax[1].text(7,300,'dashed: Fluid', fontsize=10)

    fig.tight_layout()
    plt.show()
