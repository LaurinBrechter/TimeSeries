import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from sklearn.decomposition import PCA
import math
from scipy.spatial import KDTree


def dist(x, query):
    """
    calculates the euclidean distance between two time series or a database of time series and a query
    """
    return np.sqrt(np.sum((x - query)**2, axis=1))

def BuildIndex(data_reduced):
    """
    builds the index.
    data: the data from which to build the index, should have dimensionality of < 20, optimally < 12
    """
    return KDTree(data_reduced, leafsize=data_reduced.shape[1])


def PAA(data, output_size, is_query=False):
    """
    project the data into a lower dimension via PAA
    """
    paa=PiecewiseAggregateApproximation(output_size=output_size,window_size=None)

    # if data has only one dimension we can conclude it's a query and not a database
    if is_query:
        transformed_data = paa.fit_transform(data.reshape(1,-1))
    else:
        transformed_data = paa.fit_transform(data)
    return transformed_data

def RangeQuery(data, query, eps, returnindex=True):
    res = dist(data, query) < eps
    if returnindex:
        return np.arange(0,len(data))[res], len(data) * data.shape[1]
    else:
        return res, len(data) * data.shape[1]




def RangeQueryTransform(data_reduced, data_original, query_transformed, query_original, eps):
    
    n = query_original.shape[0] # len(query_original)
    N = query_transformed.shape[1] # len(query_transformed)

    # anzahl distanzberechnungen im reduzierten Raum
    n_actions = N * len(data_reduced)


    eps_adj = eps / np.sqrt(n/N)


    candidates = RangeQuery(data_reduced, query_transformed, eps_adj)[0]


    n_candidates = len(candidates)

    # anzahl Distanzberechnungen im nicht reduzierten Raum
    n_actions += n_candidates * data_original.shape[1]

    res = RangeQuery(data_original[candidates], query_original, eps, returnindex=False)


    actual_seq = candidates[res[0]]

    if len(candidates) > 0:
        false_alarms_perc = 1 - (len(actual_seq)/len(candidates))
    else:
        false_alarms_perc = np.NAN

    return actual_seq, n_actions, false_alarms_perc


def K_NearestNeighborTransform(data_reduced, data_original, query_transformed, query_original, k):

    n = query_original.shape[0] # reducesd time series dimensionality
    N = query_transformed.shape[1] # original time series dimensionality

    # anzahl distanzberechnungen im reduzierten Raum
    n_actions = N * len(data_reduced)

    dists = dist(data_reduced, query_transformed)
    idx = np.arange(0,len(dists))

    kn = np.argsort(dists)[:k]

    actual = data_original[kn]

    emax = dist(actual, query_original).max()

    kn_actual = RangeQuery(data_original, query_original, emax)

    return np.argsort(dists)[:k], emax, kn_actual
    return np.sort(np.vstack((dists, idx)).transpose())


def RangeQueryIndex(data_reduced, data_original, query_transformed, query_original, eps):
    """
    Range query, falls mit k-D Trees gearbeitet wird
    """
    n = len(query_original)
    N = len(query_transformed)

    # find all candidate objects
    candidates_index = np.array(data_reduced.query_ball_point(query_transformed, eps / np.sqrt(n/N))[0])

    

    n_operations = len(data_reduced) * data_reduced.shape[1]
    n_operations += len(candidates_index) * data_original.shape[1]
    
    # retrieve the original sequences
    seq_act = data_original[candidates_index]

    # compute the actual distances
    real_dist = np.sqrt(np.sum((seq_act - query_original)**2, axis=1))

    # discard the false alarms
    res = candidates_index[real_dist <= eps]
    return  res 
    

def K_NearestNeighborIndex(data_original, data_reduced, query_original,query_transformed, k):
    """
    k-neighbor query, falls mit k-D Trees gearbeitet wird
    """
    

    # find the k nearest objects in the index
    kn_candiates_index = data_reduced.query(x=query_transformed, k=k)[1][0]

    # retrieve the actual seq from disk
    seq_act = data_original[kn_candiates_index]


    # compute the actual distances and get emax
    real_dist = np.sqrt(np.sum((seq_act - query_original)**2, axis=1))
    emax = np.max(real_dist)

    rq = RangeQueryIndex(data_reduced, data_original, query_original,query_transformed, emax)

    return rq



def PLA_segment(data):
    n = len(data)

    t = np.arange(1,len(data)+1)

    # Konstanten
    a_denominator = n * (n+1) * (n-1)
    b_denominator = n * (n-1)

    c_a = (n+1)/2
    c_b = ((2*n+1)/3)

    # print("diff",t-c_a*data)

    # print(np.sum((t - c_a) * data))
    a = 12 * np.sum((t - c_a) * data) / a_denominator
    b = (6 * np.sum((t - c_b) * data)) / b_denominator

    return a,b,t