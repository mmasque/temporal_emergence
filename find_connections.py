import numpy as np
import sys
from pathlib import Path

class Converter:
    @staticmethod
    def sec_to_ms(timeseries):
        return timeseries * 1000
    @staticmethod
    def sec_to_ms_neurons(neurons):
        return np.array([Converter.sec_to_ms(neuron) for neuron in neurons])

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
    def get_annotations(folder):
        return SteinmetzLoader.loader(folder, "clusters._phy_annotation.npy")

    @staticmethod
    def get_individual_neuron_arrays(folder, probe, indices=None):
        """Load the spike times into individual arrays, one for each neuron
            - probe: The probe index to select for the recording
            - indices: If None, loads all neurons in the probe, else 
            loads the indices (relative to the good neurons in the probe) specified.
        """
        times = SteinmetzLoader.get_times(folder)   # times of the spikes
        clusters = SteinmetzLoader.get_clusters(folder) # the cluster number of each spike, matches with times
        probes = SteinmetzLoader.get_probes(folder) # the probe number of each spike
        annotations = SteinmetzLoader.get_annotations(folder) # get the quality of each cluster

        good_indices = np.where(annotations >= 2)   # take only the clusters (neurons) with good quality
        probe_indices = np.where(probes == probe)   # select the probe indices
        good_indices_probe = np.intersect1d(good_indices, probe_indices) # take the indices of good neurons in the probe
        
        if indices is not None:
            good_indices_probe = good_indices_probe[indices]
        
        # split data into individual neuron arrays
        individual_times = []
        for i in good_indices_probe:
            indices = np.where(clusters==i)[0]  # find all the times of spikes for each cluster index
            individual_times.append(times[indices].astype(float))

        return individual_times

    @staticmethod
    def save_to_files(neurons, outfolder):
        Path(outfolder).mkdir(parents=True, exist_ok=True)  # https://stackoverflow.com/a/273227
        for i, neuron in enumerate(neurons):
            fname = "cell" + str(i) + ".txt"
            with open(outfolder+"/"+fname, 'wb') as t:
                np.savetxt(t, neuron)

    @staticmethod
    def save_to_single_file(neurons, outfile):
        with open(outfile + ".txt", 'wb') as f:
            for neuron in neurons:
                np.savetxt(f, neuron)
                f.write(b';\n')

    @staticmethod
    def save_single_folder_neurons(infolder, outfname, probe_index, neuron_indices=None):
        """Load steinmetz neuron time series and save as individual files, converted
        to ms (from sec)
            - infolder:         the folder with the Steinmetz dataset
            - probe_index:      the probe to load neurons from
            - neuron_indices:   the neuron indices to load, relative to the probe's good neurons.
                                If none, loads all the neurons in the selected probe.  
        """
         # load the timeseries of each neuron in separated arrays
        individual_times = SteinmetzLoader.get_individual_neuron_arrays(infolder, probe_index, neuron_indices)
        
        # convert them to ms (original data is in seconds)
        individual_times_ms = Converter.sec_to_ms_neurons(individual_times)

        # save the loaded neurons to individual files
        SteinmetzLoader.save_to_files(individual_times_ms, outfname)

        # save to single file
        #SteinmetzLoader.save_to_single_file(individual_times_ms, outfname)
        

if __name__ == "__main__":
    folder, out, probe = sys.argv[1:4]
    indices = [35,26,143,168]

    SteinmetzLoader.save_single_folder_neurons(folder, out, 1)

    
