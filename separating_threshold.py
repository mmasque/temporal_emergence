#%%
""" PSEUDOCODE FROM SUPPLEMENTARY MATERIAL, PASQUALE ET AL. 2010
xx  bins of the current logISIH
    yy  logISIH
    Smooth yy by operating a local regression using weighted linear least squares and a 1st degree polynomial model (e.g. using smooth function in MATLAB Curve Fitting Toolbox, method ‘lowess’)
    Identify local maxima in the logISIH (e.g. using findpeaks function in MATLAB Signal Processing Toolbox, parameter ‘minpeakdistance’ set at 2)
    if there is at least one peak
        Look for peaks within 10^2 ms
        If there is more than one peak within 10^2 ms
            Consider the biggest one and save the first peak's x- and y-coordinates
        else if there is no peak identified below 10^2 ms 
                return; // the channel is not analyzed
            end
        end
        if there is only one peak
            return; // no ISIth can be determined if there is only one peak
        else	// there is more than one peak
            for each pair of peaks constituted by the first and one of the following
                Compute the void parameter: the void parameter is a measure of the degree of separation between the two peaks through the minimum
        end
            Look for local minima whose void parameter satisfies a threshold (e.g. 0.7)
            if there is no minimum that satisfies the threshold
                return;	// no ISIth can be determined 
            else
                Save the corresponding ISI value as ISIth
            end
        end

"""
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

def separating_threshold(bins,ys):
    F = 0.1    # Different to review paper, see line 120: https://github.com/ellesec/burstanalysis/blob/master/Burst_detection_methods/logisi_pasq_method.R
    s_xs_ys = lowess(ys, bins, F, it=0, delta=0.0, is_sorted=True)
    xs, ys = s_xs_ys[:, 0], s_xs_ys[:, 1]
    peaks = find_peaks(ys,distance=2)[0]  # Distance set according to supplementary information, Pasquale et al. 2010
    x_peaks = xs[peaks]

    if peaks.size > 1:
        # indices of peaks in first 100ms
        peaks_100ms = peaks[np.where(x_peaks < 100)[0]]

        if peaks_100ms.size == 0: # require a peak in first 100ms
            return False
        
        # find index of max peak in the early peaks
        max_peak_100ms_ind = np.argmax(ys[peaks_100ms]) # index of peak in array of peaks
        max_peak_100ms = peaks_100ms[max_peak_100ms_ind] # index of max early peak in xs,ys

        VOID_THRESH = 0.7
        best_void, best_2nd_ind, local_min = 0, None, None
        for i in range(max_peak_100ms_ind+1, len(peaks)):
            p2 = peaks[i]
            local_min_ind = max_peak_100ms + np.argmin(ys[max_peak_100ms:p2+1])    # local min between peaks
            local_min_i = ys[local_min_ind]
            void = 1 - local_min_i/np.sqrt(ys[max_peak_100ms] * ys[p2])   # void param
            print(void)
            if void > VOID_THRESH:  # require void larger than thresh
                if void > best_void:    # update best void if needed
                    best_void, best_2nd_ind, local_min = max_peak_100ms, p2, local_min_ind

        if best_2nd_ind == None:   # if no second peak had good enough void 
            raise ValueError("Didn't find a burst")

        return max_peak_100ms, best_2nd_ind, local_min, void, s_xs_ys
    else:
        raise ValueError("Didn't find a burst")
    
def is_bursting(cellfile):
    spikes = np.loadtxt(cellfile)
    diff_spikes = np.diff(spikes)
    
    logbins = np.logspace(0,5,100)
    centered_bins = np.array([(logbins[i] + logbins[i+1])/2 for i in range(len(logbins)-1)])
    ys, _ = np.histogram(diff_spikes, bins=logbins)
    try:
        peak1, peak2, min_ind, void, _ = separating_threshold(centered_bins,ys)
        return True, (peak1, peak2,min_ind, void)

    except:
        return False, None
# %%
if __name__ == "__main__":
    i,j = 143,187

    # for neuron i
    spikes = np.loadtxt("GLMCC/Cori_2016-12-14_probe1/cell" + str(i) + ".txt")
    diff_spikes = np.diff(spikes)
    
    logbins = np.logspace(0,5,100)
    centered_bins = np.array([(logbins[i] + logbins[i+1])/2 for i in range(len(logbins)-1)])
    ys, _ = np.histogram(diff_spikes, bins=logbins)
    try:
        peak1, peak2, min_ind, void, fit = separating_threshold(centered_bins,ys)
        plt.hist(diff_spikes, bins=logbins)
        plt.scatter([centered_bins[(peak1, peak2)]], [ys[(peak1, peak2)]], color='red')
        plt.scatter([centered_bins[min_ind]], [ys[min_ind]], color='green')
        plt.xscale("log")
        plt.plot(fit[:,0], fit[:, 1])

        print(ys[min_ind], ys[peak1], ys[peak2])
        print("Void: ", 1-ys[min_ind]/np.sqrt(ys[peak1] * ys[peak2]))

    except:
        pass
    ## PLOT
# %%
