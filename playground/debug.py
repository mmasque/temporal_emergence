# Get the TPMs for 143-168. First import the data
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random

times = np.squeeze(np.load("Cori_2016-12-14/spikes.times.npy"))
clusters = np.squeeze(np.load("Cori_2016-12-14/spikes.clusters.npy"))
probe = np.squeeze(np.load("Cori_2016-12-14/clusters.probes.npy"))

minindex = min(clusters)
maxindex = max(clusters)

# split data into individual neuron arrays
individual_times = []
for i in range(minindex, maxindex+1):
    indices = np.where(clusters==i)[0]
    individual_times.append(times[indices].astype(float))

# get only the good neurons
annotations = np.squeeze(np.load("Cori_2016-12-14/clusters._phy_annotation.npy"))
good_indices = np.where(annotations >= 2)
good_neurons = np.array(individual_times)[good_indices].tolist()

probe1_indices = np.nonzero(probe)[0]
good_indices_probe1 = np.intersect1d(good_indices,probe1_indices)
print(good_indices_probe1.shape)
good_neurons_probe1 = np.array(individual_times)[good_indices_probe1]

n_143 = good_neurons_probe1[143]
n_168 = good_neurons_probe1[168]
import math
def binarise_spiketrain(a,S):
    """
    Binarises a spike-train with bins of size S. 
    - a is a list of times when the neuron fired. 
    """
    last = math.ceil(max(a)/S)
    a_states = np.zeros(math.ceil(max(a)/S))
    for spike_t in a:
        index = int(spike_t / S)
        a_states[index] = 1
    return a_states

def get_TPM_index(state):  # rename this
    """
    Given the state of a set of neurons, get the
    index into the TPM that the state corresponds to.

    Example: 
        state = [[1,0,0]
                 [0,1,1]]
        Neuron A is in state 100 and B is in state 011
        State A corresponds to the decimal 4
        State B corresponds to the decimal 3
        The index of A, the first listed state, varies 'fastest' in the TPM according to the PyPhi convention, 
        https://pyphi.readthedocs.io/en/latest/conventions.html.
        So, the indices, as tuples, will be (A0,B0),(A1,B0),(A2,B0),(A3,B0),(A4,B0),...,(A0,B1),(A1,B1)... 
        To convert these tuples to ints, we need A + B*8 = 4 + 3*8 = 28

        In general, where each node has n possible states, this becomes:
            - A*n^0 + B*n^1 + C*n^2 + ... where A, B, C are the states of each node,
            in order from first to last in the state input. 
            - Note that for the binary arrays accepted by this function, 
            n = 2^(m), where m is the size of the binary array that describes the state of a node. 
    """
    n = 2**(state.shape[1])
    index = 0
    for i in range(state.shape[0]):
        dec_node = int("".join(str(int(j)) for j in state[i,:]), 2) #slow?
        index += dec_node * n ** i
    return index

def get_TPM_nonbinary(binaryneurons, K, skipby):
    """Given an array of binarised neuron spike-trains
    and a K value for how many time-steps to include in a single state, 
    get the TPM of the system. 
        - Skipby controls where the future state starts: given that the current state
        starts at T, future state starts at T+skipby
    Example: 
        if K = 3, then find the TPM that describes 
        the transition probability of System[t-2,t-1,t] --> System[t+1,t+2,t+3]

    Returns:
        - A TPM of the system in state-state mode (TODO: conventions?)
        - A matrix A with the same dimensions of TPM, where A[i,j] is the 
        number of transitions that were used to calculate the value of TPM[i,j].
    
    """
    assert K >= 1
    assert binaryneurons.shape[0] >= 1

    size = (2**K**binaryneurons.shape[0])**2
    # initialise TPM and num_transitions arrays
    TPM = np.zeros((size, size))
    num_transitions = np.zeros((size, size))

    # start at K-1 because our state at time i looks BACK to i-1, i-2,.. to build the rest of state
    i = K - 1
    while i + skipby < binaryneurons.shape[1]:
        curr_state = binaryneurons[:,i-(K-1):(i+1)]
        i_c = get_TPM_index(curr_state)

        future_state = binaryneurons[:,i-(K-1) + skipby:(i+1) + skipby]   # Ugly indexing 
        i_f = get_TPM_index(future_state)

        num_transitions[i_c, i_f] += 1
        i += 1
    
    for j in range(num_transitions.shape[0]):
        total = sum(num_transitions[j,:])
        if total == 0:
            raise ValueError("State with index " + str(j) + " was not observed in the data")
        
        TPM[j,:] = num_transitions[j,:] / total
    
    return TPM, num_transitions

def TPM_from_spiketrains(spiketrains, S, K, skip):

    # get the binarised spike trains for each neuron
    # create a multidimensional array of the spiketrains by not considering further than the shortest train
    binarised_trains = [[] for _ in range(len(spiketrains))]
    min_length = math.inf
    for i in range(len(spiketrains)):
        binarised_trains[i] = binarise_spiketrain(spiketrains[i], S)
        min_length = min(min_length, len(binarised_trains[i]))
    
    binarised_trains = np.array([binarised_trains[i][0:min_length] for i in range(len(binarised_trains))])   # probably slow? 
    print(binarised_trains.shape)
    # compute the TPM 
    return get_TPM_nonbinary(binarised_trains, K, skip)
     
TPM = TPM_from_spiketrains([n_143,n_168],0.01,1,1)