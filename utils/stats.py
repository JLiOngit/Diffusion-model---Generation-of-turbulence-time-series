import numpy as np


def velocity_increments(velocities, tau):
    increments = (velocities[:,:,tau:] - velocities[:,:,:-tau]).reshape(-1)
    return increments

def structure_p(velocities, p, tau_values):
    structure_values = []
    for tau in tau_values :
        structure_p_tau = ((velocities[:,0,tau:] - velocities[:,0,:-tau])**p).mean().item()
        structure_values.append(structure_p_tau)
    return structure_values

def flatness_p(velocities, p, tau_values):
    structure_p_values = np.array(structure_p(velocities, p, tau_values))
    structure_2_values = np.array(structure_p(velocities, 2, tau_values))
    flatness_values = structure_p_values / (structure_2_values)**(p/2)
    return flatness_values