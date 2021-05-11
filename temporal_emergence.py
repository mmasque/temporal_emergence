import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import pyphi # needs nonbinary install
pyphi.config.PARTITION_TYPE = 'ALL'
pyphi.config.MEASURE = 'AID'
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = True
pyphi.config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = True
pyphi.config.WELCOME_OFF = True

class Neuron:

    @staticmethod
    def binarise_spiketrain(a,S):
        """
        Binarises a spike-train with bins of size S. 
        - a is a list of times when the neuron fired. 
        """
        a_states = np.zeros(math.ceil(max(a)/S))
        for spike_t in a:
            index = int(spike_t / S)
            a_states[index] = 1
        return a_states

class TPMMaker:

    @staticmethod    
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
            dec_node = int("".join(str(int(j)) for j in state[i,:]), 2)
            index += dec_node * n ** i
        return index

    @staticmethod
    def get_TPM_nonbinary(binaryneurons, K, skipby, required_obs):
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

        size = (2**K)**binaryneurons.shape[0]
        
        # initialise TPM and num_transitions arrays
        TPM = np.zeros((size, size))
        num_transitions = np.zeros((size, size))

        # get a randomly ordered list of indices at which to look at transitions
        # start at K-1 because our state at time i looks BACK to i-1, i-2,.. to build the rest of state
        rand_indices = np.array(list(range(K-1, binaryneurons.shape[1] - skipby)))
        np.random.shuffle(rand_indices)

        for i in rand_indices:
            curr_state = binaryneurons[:,i-(K-1):(i+1)]
            i_c = TPMMaker.get_TPM_index(curr_state)
            total = sum(num_transitions[i_c,:])
            if total >= required_obs:   # don't add this observation if we already have enough
                continue

            future_state = binaryneurons[:,i-(K-1) + skipby:(i+1) + skipby]   # Ugly indexing 
            i_f = TPMMaker.get_TPM_index(future_state)

            num_transitions[i_c, i_f] += 1
        
        for j in range(num_transitions.shape[0]):
            total = sum(num_transitions[j,:])
            if total < required_obs:
                raise ValueError("State with index " + str(j) + \
                " was observed in the data fewer than " + str(required_obs) + " times, (" + str(total) + " times only).")
            
            TPM[j,:] = num_transitions[j,:] / total
        
        return TPM, num_transitions

    @staticmethod
    def TPM_from_spiketrains(spiketrains, S, K, skip, required_obs):

        # get the binarised spike trains for each neuron
        # create a multidimensional array of the spiketrains by not considering further than the shortest train
        binarised_trains = [[] for _ in range(len(spiketrains))]
        min_length = math.inf
        for i in range(len(spiketrains)):
            binarised_trains[i] = Neuron.binarise_spiketrain(spiketrains[i], S)
            min_length = min(min_length, len(binarised_trains[i]))
        
        binarised_trains = np.array([binarised_trains[i][0:min_length] for i in range(len(binarised_trains))])   # probably slow? 
        # compute the TPM 
        return TPMMaker.get_TPM_nonbinary(binarised_trains, K, skip, required_obs)


class CoarseGrainer:

    @staticmethod
    def coarse_grain_nonbinary_TPM(TPM, state_map, num_states_per_elem):
        """
        TODO: add checks to inputs

        Turns a nonbinary TPM into a binary TPM by coarse-graining each nonbinary element
        according to a grouping.
            - grouping only groups states of individual elements.
            - grouping example for a TPM of 2 elements each w 4 states:
            [[[0, 1], [2, 3]], [[0], [1, 2], [3]]]
                - The first element's 0 and 1 states are grouped into a state, and 2,3 into another.
                - The second element's 0 state stays the same, 1,2 are grouped and 3 stays the same.

        state_map:
            - a dictionary that takes in the index of a macro state, obtained from num_states_per_elem
            and returns a list of micro states to the TPM that make up the macro state. 

        For example, for a TPM of 2 elements, each with 4 states,
        the input TPM will have 4*4 = 16 states. We can coarse grain
        such that only the first three states of each element will be grouped into an OFF state,
        and the last element will be grouped into ON. 

        However, we don't have to coarsegrain to binary. We might say that the first state
        of each element will map to OFF, the second and third will map to FIRING, 
        fourth to BURSTING. 
        """
        num_states = 1
        for i in num_states_per_elem:
            num_states *= i

        new_TPM = np.zeros((num_states, num_states))
        for i in range(num_states):
            input_micro_indices = state_map[i]
            for j in range(num_states):
                output_micro_indices = state_map[j]
                
                total_prob = 0
                for output_state in output_micro_indices:
                    avg_for_output_state = 0
                    for input_state in input_micro_indices:
                        avg_for_output_state += TPM[input_state, output_state]

                    avg_for_output_state = avg_for_output_state / len(input_micro_indices)
                    total_prob += avg_for_output_state
        
                new_TPM[i,j] = total_prob
        return new_TPM

    @staticmethod
    def get_state_map(coarse_grain):
        # TODO: finish
        # get how many macro states we have
        num_micro_states_per_elem = []
        num_macro_states = 1
        for elem in coarse_grain:
            num_macro_states *= len(elem)
            num_micro_states_per_elem.append(sum(len(x) for x in elem))

        state_map = {i : [] for i in range(num_macro_states)}
        
        # for each micro state, get its index, and then get its macro index
        # to get macro state from a micro index, we need 

class PhiCalculator:
    @staticmethod
    def get_micro_average_phi(TPM,verbose=True):
        network = pyphi.Network(
        (TPM),
        num_states_per_node=[4,4]
        )
        phis = []
        states = Helpers.get_nary_states(2,4)   # not the TPM order! 
        for state in states:
            subsystem = pyphi.Subsystem(network, state)
            sia = pyphi.compute.sia(subsystem)
            if state == (0,1):
                if verbose:
                    print(sia.ces)
                    print(sia.partitioned_ces)
                    print(sia.cut)
            phis.append(sia.phi)
        if verbose:
            print(phis)
        return sum(phis) / len(phis)
    
    @staticmethod
    def get_macro_average_phi(micro_TPM, verbose=True):

        num_states_per_elem = [2,2]
        state_map = {0: [0,1,2, 4,5,6, 8,9,10], 1: [3,7,11], 2: [12,13,14], 3:[15]}
        macro_TPM = CoarseGrainer.coarse_grain_nonbinary_TPM(micro_TPM, state_map, num_states_per_elem)
        #np.savetxt("TPMs/macro_example_143_168.csv", macro_TPM)
        # Macro analysis where bursting is one state, and everything else is another
        network = pyphi.Network(
        macro_TPM,
        num_states_per_node=[2,2]
        )

        states = reversed(Helpers.get_nary_states(2,2)) # not the TPM order! 
        phis = []
        for state in states:
            subsystem = pyphi.Subsystem(network, state)
            sia = pyphi.compute.sia(subsystem)
            phis.append(sia.phi)
        
        if verbose:
            print(sia.ces)
            print(sia.partitioned_ces)
            print(sia.cut)
        return sum(phis) / len(phis)

class Helpers:
    @staticmethod
    def get_bin_states(l):
        states = []
        for i in range(2**l):
            state = list(bin(i)[2:])
            while len(state) < l:
                state.insert(0,'0')
            state = tuple([int(c) for c in state])
            states.append(state)
        return states

    # https://stackoverflow.com/a/28666223
    @staticmethod
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]

    @staticmethod
    def get_nary_states(n,b):
        states = []
        for i in range(b**n):
            state = Helpers.numberToBase(i,b)
            while len(state) < n:
                state.insert(0,0)
            state = tuple(state)
            states.append(state)
        return states

class DataGenerator:
    def __init__(self, TPM, base):
        self.TPM = TPM
        # base is the number of possible states in each state of the TPM
        self.base = base
        self.num_nodes = int(math.log(self.TPM.shape[0], self.base))

    def generate_timeseries(self, iters):
        bits_in_state = int(math.log2(self.base))
        data = np.zeros((self.num_nodes, iters*bits_in_state))
        curr_state = 0
        for i in range(0, iters*bits_in_state, bits_in_state):
            # see https://stackoverflow.com/a/41852266
            curr_state = np.random.choice(np.arange(len(self.TPM[curr_state])), p=self.TPM[curr_state])

            # get the system state in array form, with the state of each node 
            state_array = Helpers.numberToBase(curr_state, self.base)
            while len(state_array) < self.num_nodes:
                state_array.insert(0,0)
            state_array = state_array[::-1]

            for j in range(len(state_array)):   # for each node
                # we need to set multiple values of the timeseries, because each iteration in the nonbinary TPM 
                # is actually multiple binary steps, the nonbinary representation 'bunches up' binary data.
                binary_state = Helpers.numberToBase(state_array[j], 2)
                while len(binary_state) < bits_in_state:
                    binary_state.insert(0,0)
                for k in range(bits_in_state):
                    data[j, i+k] = binary_state[k]
        return data


if __name__ == "__main__":
    state_array = Helpers.numberToBase(2, 4)
    while len(state_array) < 2:
        state_array.insert(0,0)
    state_array = state_array[::-1]
    print(state_array)

    print(TPMMaker.get_TPM_index(np.array([[1,0],[0,0]])))

