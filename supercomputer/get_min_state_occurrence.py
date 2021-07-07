### COMPUTE MIN NUMBER OF STATE OCCURRENCES in each BIDIRECTIONALLY CONNECTED pair ### 
from temporal_emergence import TPMMaker 
import numpy as np
import multiprocessing as mp

folder = "DATA/Cori_2016-12-14_probe1"
out_results = "RESULTS/min_occurrences"

### OBTAIN BIDIRECTIONALLY CONNECTED NEURONS ### 

probe1_371 = np.loadtxt("DATA/W_py_5400.csv", delimiter=",")

# get tuples of connected neurons
ai,bi = np.where(abs(probe1_371) != 0)
index_pairs = list(zip(ai,bi))

bidirectionally = []
for (r,t) in index_pairs:
    if (t,r) in index_pairs and (t,r) not in bidirectionally:
        bidirectionally.append((r,t))

small_bidirectionally = bidirectionally[0:12]

### COMPUTE MIN OCCURRENCES ### 

def min_occurrences(i,j):
    ### PARAMETERS ###
    NUM_BITS = 2

    max_binsize = 0.02  # 20 ms bins
    min_binsize = 0.0029 # skip 1ms bins  -   never work and are very slow to compute
    num_binsizes = 9
    binsizes = np.linspace(min_binsize, max_binsize, num_binsizes)

    min_num_occurrences = np.zeros(len(binsizes))
    ### LOAD DATASET ###

    i_sec = np.loadtxt(folder + "/cell" + str(i) + ".txt") / 1000   # divide through as they are loaded in seconds
    j_sec = np.loadtxt(folder + "/cell" + str(j) + ".txt") / 1000
    cluster = np.array([i_sec, j_sec])
    ### LOOP THROUGH PARAMETERS, computing min of state occurrences ###
    for k in range(len(binsizes)):
        binsize = binsizes[k]
        min_occ = np.min(TPMMaker.get_num_state_occurrences(cluster,binsize, NUM_BITS, 0))   # keep skip at 0, it's not actually used
        min_num_occurrences[k] = min_occ

    #with open(out_results, 'a') as f:
    #    f.write(str(i) + "," + str(j) + "," + str(min_num_occurrences) + "\n")
    print(min_num_occurrences)
    return (i,j,min_num_occurrences)

def chunk_min_occurrences(chunk):
    print("CHUNK_MIN_OCCURRENCES")
    print(chunk)
    rs = []
    for (i,j) in chunk:
        r = min_occurrences(i,j)
        rs.append(r)
    print(rs)
    return rs

def resulter(result):
    global results
    results += result

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


### PARALLEL PROCESSING SETUP ###
print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
results = []
#for (i,j) in small_bidirectionally:
#    pool.apply_async(min_occurrences, args=(i,j), callback=resulter)

cs = chunks(bidirectionally, len(bidirectionally)//24 + 1)
print(cs)
for chunk in cs:
    print(chunk)
    pool.apply_async(chunk_min_occurrences, args=(chunk,), callback=resulter)

#sorted(results, key=lambda element: (element[1], element[2]))
pool.close()
pool.join()

np.save(out_results, results)