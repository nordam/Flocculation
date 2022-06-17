#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
import numpy as np
from numba import njit

@njit
def alpha(m1, m2):
    return 1e-6

@njit
def beta(m1, m2):
    return 1.0

@njit
def coagulate_dense(M, dt):
    Np = len(M)
    # Check all pairs of particles
    for i in range(Np):
        for j in range(i-1): # Do not react with self
            # Check that both particles are still valid
            if ((M[i] > 0) and (M[j] > 0)):
                # Coagulation rate
                rate = alpha(M[i], M[j]) * beta(M[i], M[j])
                # Probability of reacting
                p = 1 - np.exp(-dt*rate)
                # Uniform random number
                r = np.random.random()
                # If reaction occurs, calculate new particle size
                # Assign new size to one particle, and remove the other
                if r < p:
                    M[i] = M[i] + M[j]
                    M[j] = -999
    # Return only valid particles
    return M[M>0]

###########################################
#### Main function to run a simulation ####
###########################################

def experiment(M0, Tmax, dt, save_dt, args=None):
    '''
    Run the model.

    M0:    initial mass of particles
    Tmax:  Total duration of the simulation [s]
    dt:    Timestep [s]
    args:  Command line arguments from argparse (optional)
    '''

    # Number of particles
    Np = len(M0)
    # Number of timesteps
    Nt = int(Tmax / dt) + 1
    # Calculate size of output arrays
    N_skip = int(save_dt/dt)
    N_out = 1 + int(Nt / N_skip)
    # Array to store output
    M_out = np.zeros((N_out, Np)) - 999
    # Array to track mass
    M = M0.copy()

    # Use trange (progress bar) if instructed
    iterator = range
    if args is not None:
        if args.progress:
            iterator = trange

    # Time loop
    t = 0
    for n in iterator(Nt):

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            M_out[i,:len(M)] = M

        # Reaction
        M = coagulate_dense(M, dt)
        # Increment time
        t = dt*i

    return M_out


##############################
#### Numerical parameters ####
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = float, default = 1, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 10, help = 'Interval at which to save results')
parser.add_argument('--tmax', dest = 'Tmax', type = int, default = 100, help = 'Simulation time')
parser.add_argument('--Np', dest = 'Np', type = int, default = 1000, help = 'Number of particles')
parser.add_argument('--progress', dest = 'progress', action = 'store_true', help = 'Display progress bar?')
#parser.add_argument('--overwrite', dest = 'overwrite', action = 'store_true', help = 'Overwrite existing output?')
args = parser.parse_args()
args.overwrite=True

# Consistency check of arguments
if (args.save_dt / args.dt) != int(args.save_dt / args.dt):
    print('dt does not evenly divide save_dt, output times will not be as expected')
    sys.exit()


############################
#### Initial conditions ####
############################

M0 = np.zeros(args.Np)
# Initial mass distribution
M0[:] = 1

datafolder = '../results/'
if not os.path.exists(datafolder):
    print(f'Creating result folder: {datafolder}')
    os.mkdir(datafolder)

outputfilename_M = os.path.join(datafolder, f'flocculation_Np={args.Np}_dt={args.dt}_M.npy')

if (not os.path.exists(outputfilename_M)) or args.overwrite:
    tic = time.time()
    M_out = experiment(M0, args.Tmax, args.dt, args.save_dt, args=args)
    toc = time.time()
    np.save(outputfilename_M, M_out)
    print(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename_M}')
else:
    print(f'File exists, skipping: {outputfilename_M}')


