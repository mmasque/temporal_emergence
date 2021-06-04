import numpy as np
import sys
import os

class SteinmetzLoader:

    @staticmethod
    def loader(folder, file):
        return np.squeeze(np.load(folder + "/" + file))

    @staticmethod
    def get_times(folder):
        return SteinmetzLoader.loader(folder, "spikes.times.npy")

    @staticmethod
    def get_clusters(folder):
        return SteinmetzLoader.loader(folder, "spikes.clusters.npy")

    @staticmethod
    def get_probes(folder):
        return SteinmetzLoader.loader(folder, "clusters.probes.npy")

    @staticmethod
    def get_individual_neuron_arrays(folder):
        times = SteinmetzLoader.get_times(folder)
        clusters = SteinmetzLoader.get_clusters(folder)
        probes = SteinmetzLoader.get_probes(folder)
        minindex, maxindex = min(clusters), max(clusters)

        # split data into individual neuron arrays
        individual_times = []
        for i in range(minindex, maxindex+1):
            indices = np.where(clusters==i)[0]
            individual_times.append(times[indices].astype(float))

        return individual_times

    @staticmethod
    def get_good_neurons(folder, individual_times):
        # get only the good neurons
        annotations = np.squeeze(np.load(folder+"/clusters._phy_annotation.npy"))
        good_indices = np.where(annotations >= 2)
        good_neurons = np.array(individual_times)[good_indices].tolist()
        return good_indices, good_neurons
    
    @staticmethod
    def get_good_neurons_for_probe(folder, individual_times, probe_index):

        good_indices, good_neurons = SteinmetzLoader.get_good_neurons(folder, individual_times)
        probe = SteinmetzLoader.get_probes(folder)

        probe1_indices = np.nonzero(probe)[probe_index]
        good_indices_probe = np.intersect1d(good_indices,probe1_indices)
        good_neurons_probe = np.array(individual_times)[good_indices_probe]

        return good_indices_probe, good_neurons_probe

    @staticmethod
    def save_single_folder_some_neurons(neurons, outfolder):
        os.mkdir(outfolder)
        for i, neuron in enumerate(neurons[0::50]): ## change
            fname = "cell" + str(i) + ".txt"
            with open(outfolder+"/"+fname, 'wb') as t:
                np.savetxt(t, neuron)

    @staticmethod
    def save_single_folder_neurons(infolder, outfname, probe_index=None):
        individual_times = SteinmetzLoader.get_individual_neuron_arrays(infolder)
        if probe_index is None:
            _, neurons = SteinmetzLoader.get_good_neurons(infolder, individual_times)
        else:
            _, neurons = SteinmetzLoader.get_good_neurons_for_probe(infolder, individual_times, probe_index)

        SteinmetzLoader.save_single_folder_some_neurons(neurons, outfname)
    
    

if __name__ == "__main__":
    folder, out = sys.argv[1], sys.argv[2]
    SteinmetzLoader.save_single_folder_neurons(folder, out, 0)
    #SteinmetzLoader.get_good_neurons_for_probe(folder, out, 0)

    
