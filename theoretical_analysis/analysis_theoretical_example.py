# %% 
import pyphi
import numpy as np
pyphi.config.PARTITION_TYPE = 'ALL'
pyphi.config.MEASURE = 'AID'

# %% define micro TPM

micro_TPM = np.array([
    # 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #00
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #10
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #20
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1/3,1/3,1/3,0], #30
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #01
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #11
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #21
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1/3,1/3,1/3,0], #31
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #02
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #12
    [1/9, 1/9,1/9,0,  1/9,1/9,1/9,0,  1/9,1/9,1/9,0,  0,  0,  0,  0], #22
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1/3,1/3,1/3,0], #32
    [ 0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  0], #03
    [ 0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  0], #13
    [ 0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  1/3,0,  0,  0,  0], #23
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1], #33

])

# %% Get phi for the micro TPM


def get_nary_states(n,b):
    states = []
    for i in range(b**n):
        state = numberToBase(i,b)
        while len(state) < n:
            state.insert(0,0)
        state = tuple(state)
        states.append(state)
    return states

# https://stackoverflow.com/a/28666223/6370263
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def get_micro_average_phi(TPM):
    network = pyphi.Network(
    (TPM),
    num_states_per_node=[4,4]
    )
    phis = []
    states = reversed(get_nary_states(2,4))
    for state in states:
        subsystem = pyphi.Subsystem(network, state)
        sia = pyphi.compute.sia(subsystem)
        phis.append(sia.phi)
    print(sia.ces)
    print(sia.partitioned_ces)
    return phis, sum(phis) / len(phis)

get_micro_average_phi(micro_TPM)
# %% GET MACRO AVERAGE PHI

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

    AB
    00 ....
    10
    20
    01
    11
    21
    02
    12
    22
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


def get_macro_average_phi(micro_TPM):

    num_states_per_elem = [2,2]
    state_map = {0: [0,1,2, 4,5,6, 8,9,10], 1: [3,7,11], 2: [12,13,14], 3:[15]}
    macro_TPM = coarse_grain_nonbinary_TPM(micro_TPM, state_map, num_states_per_elem)
    # Macro analysis where bursting is one state, and everything else is another
    print(macro_TPM)
    network = pyphi.Network(
    macro_TPM,
    num_states_per_node=[2,2]
    )

    states = reversed(get_nary_states(2,2))
    phis = []
    for state in states:
        subsystem = pyphi.Subsystem(network, state)
        sia = pyphi.compute.sia(subsystem)
        phis.append(sia.phi)
    print(sia.ces)
    print(sia.partitioned_ces)
    return phis, sum(phis) / len(phis)


get_macro_average_phi(micro_TPM)
# %%
macro_TPM = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])