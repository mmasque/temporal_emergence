""" get_location_plot is doing the work here, plots the physical location of neurons given 
    a matrix of connectivity, and a list of locations of the neurons in the connectivity matrix
"""
import json
from urllib.request import urlopen
import requests
import urllib.request
import igraph as ig

import chart_studio.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go

import numpy as np
from find_connections import SteinmetzLoader

def get_connections(connection_matrix):
    connections = []
    for i in range(connection_matrix.shape[0]):
        tos = np.nonzero(connection_matrix[i])[0]
        for t in tos:
            connections.append((i, t, connection_matrix[i,t]))
    return connections

def get_good_locations(folder, probe):
    # get only the good neurons
    annotations = SteinmetzLoader.get_annotations(folder)
    good_indices = np.where(annotations >= 2)

    probes = SteinmetzLoader.get_probes(folder)
    probe_indices = np.where(probes == probe)
    good_indices_probe1 = np.intersect1d(good_indices,probe_indices)

    ### LOAD DEPTH DATA ### 

    ## Get the peak channels of the clusters: the most prominent recording of each neuron (neuron=cluster)
    peak_channel = np.squeeze(np.load("Cori_2016-12-14/clusters.peakChannel.npy"))
    good_indices_probe1_channels = np.ndarray.astype(peak_channel[good_indices_probe1], int)

    brain_location = np.genfromtxt("Cori_2016-12-14/channels.brainLocation.tsv", dtype=None, delimiter='\t', names=True)
    
    return brain_location[good_indices_probe1_channels]


def get_location_plot(positions, connections, savefile=None):
    """Plots a location of neurons in Allen common framework, colorcoded by location, with connection edges
        neurons: list of indices of neurons, corresponding to the indices in the connection tuples
        positions: [(x,y,z,brain_location)]
            x,y,z are coordinates in the Allen common framework
            brain_location is a string describing the brain location of a neuron
            - position(neuron[i]) = positions[i]
        connections: [(source,target)]
            source/target is the index of the source/target neuron in the connection. 
            The source/target index does not access the neurons array, rather, it corresponds 
            to an index that is in the neurons array.  
    """
    
    ### GET EDGES
    xe = []
    ye = []
    ze = []
    ws = []
    for i,j,w in connections:
        xe += [positions[i][0],positions[j][0],None]
        ye += [positions[i][1],positions[j][1],None]
        ze += [positions[i][2],positions[j][2],None]
        ws += w
    ### GET POSITIONS, leaving out brain_location string
    positions_just3d = np.array([(x,y,z) for (x,y,z,_) in positions])
    xyz = [np.array(i) for i in zip(*positions_just3d)]

    ### GET BRAIN LOCATION labels
    locations = np.array([l for (_,_,_,l) in positions])

    ## GET COLORS TO MATCH BRAIN LOCATION
    ind_locations = list(set(locations))

    colors = []
    for loc in locations:
        colors.append((ind_locations.index(loc)+1) * 1/len(ind_locations))

    ## PLOTTING
    layt=ig.Layout(coords=list(positions_just3d))

    trace1=go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color='rgb(125,125,125)', width=1),text=ws, hoverinfo='text', name='functional connections')

    trace2=go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode='markers',
                    marker=dict(symbol='circle', size=6, color=colors, colorscale='Viridis', 
                        line=dict(color='rgb(50,50,50)', width=0.5)), text=locations, hoverinfo='text', name='neurons')

    data=[trace1, trace2]
    fig=go.Figure(data=data)
    iplot(fig, filename='Bi-directionally connected bursting neurons')

    if savefile is not None:

        fig.write_html(savefile)

    

if __name__ == "__main__":
    probe1_371 = np.loadtxt("results/connectivity_Cori_2016-12-14_probe1/W_py_5400.csv", delimiter=",")
    connections = get_connections(probe1_371)

    # get only the good neurons
    annotations = SteinmetzLoader.get_annotations("Cori_2016-12-14")
    good_indices = np.where(annotations >= 2)

    probe = SteinmetzLoader.get_probes("Cori_2016-12-14")
    probe1_indices = np.nonzero(probe)[0]
    good_indices_probe1 = np.intersect1d(good_indices,probe1_indices)

    ### LOAD DEPTH DATA ### 

    ## Get the peak channels of the clusters: the most prominent recording of each neuron (neuron=cluster)
    peak_channel = np.squeeze(np.load("Cori_2016-12-14/clusters.peakChannel.npy"))
    good_indices_probe1_channels = np.ndarray.astype(peak_channel[good_indices_probe1], int)

    brain_location = np.genfromtxt("Cori_2016-12-14/channels.brainLocation.tsv", dtype=None, delimiter='\t', names=True)
    good_indices_probe1_channel_locations = get_good_locations("Cori_2016-12-14", 1)
    
    get_location_plot(good_indices_probe1_channel_locations, connections, "networks/cori_2016-12-14_probe1.html")