#!/usr/bin/env python3
## GET THE PHIS FOR A PARTICULAR NEURON PAIR
from temporal_emergence import TPMMaker, PhiCalculator
import numpy as np
import pyphi
from mpi4py import MPI
pyphi.config.WELCOME_OFF = True
pyphi.config.PARTITION_TYPE = 'ALL'
pyphi.config.MEASURE = 'AID'
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = True
pyphi.config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = True
pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL_CUT_EVALUATION = False
pyphi.config.LOG_FILE_LEVEL = None
def get_phis(ref,tar, folder, outfolder, num_transitions=1000):

    ### LOAD DATASET ###

    i_msec = np.loadtxt(folder + "/cell" + str(ref) + ".txt")
    j_msec = np.loadtxt(folder + "/cell" + str(tar) + ".txt") 
    cluster = np.array([i_msec, j_msec])

    ### PARAMETERS ###

    NUM_BITS = 2

    skips = list(range(2,11,2))

    max_binsize = 0.02  # 20 ms bins
    min_binsize = 0.0029 # skip 1ms bins  -   never work and are very slow to compute
    num_binsizes = 9
    binsizes = np.linspace(min_binsize, max_binsize, num_binsizes)

    ### COMPUTE PHIS ###

    micro_phis = np.zeros((len(binsizes), len(skips)))
    NUM_COARSE_GRAININGS = 16
    macro_phis = np.zeros((len(binsizes), len(skips), NUM_COARSE_GRAININGS))

    for i in range(len(binsizes)):
        binsize = binsizes[i]
        for j in range(len(skips)):
            skip = skips[j]

            try:
                TPM,_ = TPMMaker.TPM_from_spiketrains(cluster,binsize,NUM_BITS,skip,num_transitions)
                tpmname = "micro_" + str(tar) + "_" + str(ref) + "_bin_"+str(binsize)+"_skip_"+str(skip)+".csv" 
                np.savetxt(outfolder+"/"+tpmname, TPM)
                success = True
            except ValueError:
                success = False
                #print("Failed for binsize: " + str(binsize) + " and skip: " + str(skip))
            
            if success:
                try:
                    micro_phis[i,j] = PhiCalculator.get_micro_average_phi(TPM, verbose=False)
                    macro_phis[i,j] = PhiCalculator.all_coarsegrains_get_macro_average_phi(TPM, verbose=False)
                except pyphi.exceptions.StateUnreachableError:
                    print(f"TPM for {i}, {j} raised a state unreachable error for binsize {binsize} and skip {skip}. ")
                    micro_phis[i,j] = None
                    macro_phis[i,j] = None
                #print("Success for binsize: " + str(binsize) + " and skip: " + str(skip))
            
            else:
                micro_phis[i,j] = None
                macro_phis[i,j] = None
    
    micro_phis_name = "micro_phis_"+str(ref)+"_"+str(tar)
    macro_phis_name = "macro_phis_"+str(ref)+"_"+str(tar)+"cg_"
    np.savetxt(outfolder+"/"+micro_phis_name, micro_phis)
    for k in range(NUM_COARSE_GRAININGS):
        np.savetxt(outfolder+"/"+macro_phis_name+str(k), macro_phis[:,:,k])
    #return (micro_phis, macro_phis)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    outfolder = "results"
    infolder = "data"
    num_transitions = 200
    if rank == 0:   # if we are the root node, case the data to other nodes
        bidirectionally = np.loadtxt("bidirectionally.txt")
        bidirectionally = comm.bcast(bidirectionally, root=0)
    else:
        bidirectionally = None
        bidirectionally = comm.bcast(bidirectionally, root=0)   # receive the cast
        # TODO this is inneficient, better to do this once in the root and use Send and Recv
        split_bidirectionally = np.array_split(bidirectionally, (size - 1))
        for i,j in split_bidirectionally[rank-1]:
            print(f"COMPUTING FOR {int(i), int(j)}")
            get_phis(int(i), int(j), infolder, outfolder, num_transitions)
