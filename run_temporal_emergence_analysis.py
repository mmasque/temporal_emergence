################## IMPORTS ##################
import multiprocessing as mp
import sys
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import pyphi # needs nonbinary install
pyphi.config.PARTITION_TYPE = 'ALL'
pyphi.config.MEASURE = 'AID'
pyphi.config.WELCOME_OFF = True
pyphi.config.PROGRESS_BARS = False

from temporal_emergence import TPMMaker, CoarseGrainer, PhiCalculator, DataGenerator, Helpers

################## CONFIG ##################

# Remove PyPhi parallelisation (as we have our own parallelisation)
pyphi.config.PARALLEL_CONCEPT_EVALUATION = False
pyphi.config.PARALLEL_CUT_EVALUATION = False

################## LOAD DATASET ##################

FOLDER  = "Cori_2016-12-14" # select the recording session

times = np.squeeze(np.load(FOLDER + "/spikes.times.npy"))
clusters = np.squeeze(np.load(FOLDER + "/spikes.clusters.npy"))
probe = np.squeeze(np.load(FOLDER + "/clusters.probes.npy"))

minindex = min(clusters)
maxindex = max(clusters)

# split data into individual neuron arrays
individual_times = []
for i in range(minindex, maxindex+1):
    indices = np.where(clusters==i)[0]
    individual_times.append(times[indices].astype(float))

# get only the good neurons
annotations = np.squeeze(np.load(FOLDER + "/clusters._phy_annotation.npy"))
good_indices = np.where(annotations >= 2)
good_neurons = np.array(individual_times)[good_indices].tolist()

probe1_indices = np.nonzero(probe)[0]
good_indices_probe1 = np.intersect1d(good_indices,probe1_indices)
good_neurons_probe1 = np.array(individual_times)[good_indices_probe1]

### select the required neurons ###
n_143 = good_neurons_probe1[143]
n_168 = good_neurons_probe1[168]

neurons = [143,168]
cluster_143_168 = good_neurons_probe1[neurons]
np.save("cluster_143_168", cluster_143_168)
################## RUN ANALYSIS ##################
"""
    - Runs macro emergence analysis on the pair of neurons loaded.
    - Repeats analysis for different parameters. 
    - Micro is always two bits / state. 
    - See temporal_emergence functions, specifically under PhiCalculator, for details 
    on coarse graining. 

"""

################## RUN ANALYSIS IN PARALLEL ##################

# Put every iteration in a function
def get_phis_macro_micro(k):
    cluster_143_168 = np.load("cluster_143_168.npy", allow_pickle=True)
    from temporal_emergence import TPMMaker, CoarseGrainer, PhiCalculator, DataGenerator, Helpers

    NUM_BITS = 2

    skips = list(range(2,11,2)) # Change 5 back to 2

    max_binsize = 0.02  # 50 ms bins
    min_binsize = 0.001 # 1ms bins  -   probably won't work
    num_binsizes = 10   # change back to 10
    binsizes = np.linspace(min_binsize, max_binsize, num_binsizes)

    num_transitions = 1000   # change back to 1000
    micro_phis = np.zeros((len(binsizes), len(skips)))
    macro_phis = np.zeros((len(binsizes), len(skips)))

 
    for i in range(len(binsizes)):  # paralelize here
        binsize = binsizes[i]
        for j in range(len(skips)):
            skip = skips[j]

            try:
                TPM,_ = TPMMaker.TPM_from_spiketrains(cluster_143_168,binsize,NUM_BITS,skip,num_transitions)
                tpmname = "micro_143_168_bin_"+str(binsize)+"_skip_"+str(skip)+"_iter_"+str(i)+".csv" 
                np.savetxt("TPMs/"+tpmname, TPM)
                success = True
            except:
                success = False
                print("Failed for binsize: " + str(binsize) + " and skip: " + str(skip)+" for iter: "+str(k))
            
            if success:
                micro_phis[i,j] = PhiCalculator.get_micro_average_phi(TPM, verbose=False)
                macro_phis[i,j] = PhiCalculator.get_macro_average_phi(TPM, verbose=False)
                print("Success for binsize: " + str(binsize) + " and skip: " + str(skip)+" for iter: "+str(k))
            
            else:
                micro_phis[i,j] = None
                macro_phis[i,j] = None

    #micro_name = "micro_phis_iter_" + str(iter) + ".csv"
    #np.savetxt("results/"+micro_name, micro_phis)

    #macro_name = "macro_phis_iter_" + str(iter) + ".csv"
    #np.savetxt("results/"+macro_name, macro_phis)
    return micro_phis, macro_phis

if __name__ == "__main__":
    # Create pool
    print(mp.cpu_count())
    
    with mp.Pool(mp.cpu_count()) as pool:
        ITERS = int(sys.argv[1])
        for i in range(ITERS):
            print(i)
            job = pool.apply_async(get_phis_macro_micro, [i])
            # save the arrays
            micro_phis, macro_phis = job.get()
            micro_name = "micro_phis_iter_" + str(i) + ".csv"
            np.savetxt("results/"+micro_name, micro_phis)

            macro_name = "macro_phis_iter_" + str(i) + ".csv"
            np.savetxt("results/"+macro_name, macro_phis)
    