# Imports (copied directly from mpnn notebook)

import sys, os, time, glob, random, shutil, subprocess, math

import numpy as np
import pandas as pd

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans

from scipy.optimize import curve_fit

# Functions (copied directly from mpnn notebook)

def prob_of_1term(value, zdf, pilot_term, terms_and_cuts):
    cut, good_high, term, is_int = terms_and_cuts[pilot_term]

    cut_div = cut + 0.01
    if (is_int):
        cut_div = 1
#     print(value)
    representatives = zdf[abs( (zdf[term] - value) / cut_div ) < 0.02]

    if ( good_high ):
        fail = (representatives[pilot_term] < cut).sum()
        ok = (representatives[pilot_term] >= cut).sum()
    else:
        fail = (representatives[pilot_term] > cut).sum()
        ok = (representatives[pilot_term] <= cut).sum()

    if (fail + ok < 5):
        return np.nan, 1
    return ok / (fail + ok), fail + ok


def get_low_high_bin_size_low_bin_num_bins(dfz, pilot_term, terms_and_cuts):
    cut, good_high, term, is_int = terms_and_cuts[pilot_term]
    cut_div = cut + 0.01

    low = dfz[term].min()
    high = dfz[term].max()
    if (abs( (low - cut) / cut_div ) > 1000 ):
        print("Crazy value!!!", pilot_term, low, high)
        assert(False)

    bin_size = abs(cut_div * 0.02)

    if ( is_int ):
        bin_size = 1
    low_bin = math.floor( low / bin_size )

    num_bins = math.floor( high / bin_size ) - low_bin + 1

    return low, high, bin_size, low_bin, num_bins

# Find the index of xs that maximumizes the following equation
# np.sum( xs[:i] * flip < divisor ) + np.sum( xs[i:] * flip > divisor )
def find_ind_that_divides( divisor, array, flip ):
    best_sum = 0
    best_i = None
    for i in range(len(array)):
        value = np.sum( array[:i] * flip < flip * divisor ) + np.sum( array[i:] * flip > flip * divisor )
        if ( value > best_sum ):
            best_sum = value
            best_i = i
    return best_i

def sigmoid(x, a, b):
    return 1 / ( 1 + np.exp( -( x - a) * b ) )

def smooth_empty_prob_array(eqs, arr, good_high, counts, graphs=False, low=0, bin_size=0, gd=None, term=""):
    counts = list(counts)
    x = list(range(0, len(arr)))
    to_remove = []
    for i in range(len(arr)):
        if (math.isnan(arr[i])):
            to_remove.append(i)
    arr_copy = list(arr)
    for i in reversed(to_remove):
        x.pop(i)
        arr_copy.pop(i)
        counts.pop(i)
    arr_copy = np.array(arr_copy)

#     print(good_high)

    # We're trying to fit a sigmoid here. I've found that while the
    # function will often converge with a nonsense starting point
    # if you want it to be robust, you need to calculate the parameters
    # by hand first

    # The xguess is where the sigmoid crosses 0.5


    flip = 1 if good_high else -1

    never_high = max(arr_copy) < 0.5
    never_low = min(arr_copy) > 0.5

    # Your data is totally garbage
    if (never_high and never_low):
        xguess = x[int(len(x)/2)]

    # Your data is all below 0.5, assign xguess to edge
    elif ( never_high ):
        if ( good_high ):
            xguess = x[-1]
        else:
            xguess = x[0]

    # Your data is all above 0.5, assign xguess to edge
    elif (never_low):
        if ( good_high ):
            xguess = x[0]
        else:
            xguess = x[-1]

    else:
        # here we have full range data
        # pick x that maximizes the following function
        # np.sum( arr_copy[:x] < 0.5 ) + np.sum( arr_copy[x:] > 0.5 )

        best_ix = find_ind_that_divides(0.5, arr_copy, flip)
        xguess = x[best_ix]


    # Ok, now let's work on the slope guess
    # Formula is: guess = ln( 1 / y - 1) / ( xdist from 0.5 to y)
    # We'll use y = 0.2 and 0.8

    never_high = max(arr_copy) < 0.8
    never_low = min(arr_copy) > 0.2

    # Data never goes above 0.8, assign xvalue to edge
    if ( never_high ):
        if ( good_high ):
            ub = x[-1]
        else:
            lb = x[0]
    else:
    # Find xvalue that corresponds to graph crossing 0.8
        best_ix = find_ind_that_divides(0.8, arr_copy, flip)
        if ( good_high ):
            ub = x[best_ix]
        else:
            lb = x[best_ix]

    # Data never goes below 0.2, assign xvalue to edge
    if ( never_low ):
        if ( good_high ):
            lb = x[0]
        else:
            ub = x[-1]
    else:
    # Find xvalue that corresponds to graph crossing 0.2
        best_ix = find_ind_that_divides(0.2, arr_copy, flip)
        if ( good_high ):
            lb = x[best_ix]
        else:
            ub = x[best_ix]

    # One side of the data is bad, just use the other side
    if ( ub <= xguess ):
        ub = xguess - lb + xguess
    if ( lb >= xguess ):
        lb = xguess - ( ub - xguess )

    # The data is really bad here, just assign the ub and lb to the edges
    if ( ub == lb ):
        lb = x[0]
        ub = x[-1]

    # Average our two distances
    critical_dist = (( ub - xguess ) + (xguess - lb )) / 2

    # Find slope guess
    slope_guess = np.abs( np.log( 1 / 0.2 - 1) / critical_dist ) * flip

    # Curve fit
    popt, pcov = curve_fit( sigmoid, x, arr_copy, p0=(xguess, slope_guess), maxfev=100000, sigma=1/np.sqrt(counts) )

    # Uncomment this if you're debugging the guesses (They do really well tbh)
#     popt = (xguess, slope_guess)

    # Our new fitted data
    arr2 = sigmoid(np.arange(0, len(arr), 1), popt[0], popt[1])

    a_prime = popt[0]*bin_size+low
    b_prime = popt[1]/bin_size
    eqs.append( " 1 / ( 1 + EXP( -( %s - %.5f ) * %.5f ) ) "%(term[:-5], a_prime, b_prime))


    if (graphs):
        plt.figure(figsize=(5,3))
        sns.set(font_scale=1)
        plt.plot(np.arange(0, len(arr), 1)*bin_size+low, arr)
        plt.plot(np.arange(0, len(arr), 1)*bin_size+low, arr2)
        if (gd):
            plt.xlim([gd[0], gd[1]])
            plt.xlabel(gd[2])
            plt.axvline(gd[3], color='r')
        plt.ylabel("P( passing filter )")
        sns.set(font_scale=1.8)
        plt.show()

    for i in range(len(arr2)):
        arr[i] = arr2[i]

    return arr, eqs

def create_prob_array(eqs, low, high, low_bin, num_bins, bin_size, pilot_term, dfz, terms_and_cuts, graphs=False):
    cut, good_high, term, is_int = terms_and_cuts[pilot_term]
    arr = np.zeros(num_bins)
    for i in range(len(arr)):
        arr[i] = np.nan

    counts = np.zeros(num_bins)
    counts.fill(1)

    print("%s from %.3f to %.3f"%(pilot_term, low, high))
    for val in np.arange(low, high + bin_size, bin_size/2):
        binn = math.floor( val / bin_size ) - low_bin
        if (binn >= len(arr)):
            continue
        if (is_int):
            val = round(val, 1)
#         print(val)
        prob, count = prob_of_1term(val, dfz, pilot_term, terms_and_cuts)

        counts[binn] = count + 1
        if ( math.isnan(prob)):
            pass
        else:
            arr[binn] = prob

    gd = None
    try:
        gd = graph_data[pilot_term]
    except:
        pass
    arr, eqs = smooth_empty_prob_array(eqs, arr, good_high, counts, graphs, low, bin_size, gd, term)

    return arr, eqs

def apply_prob_arrays(dfz, prob_arrays, prob_name):
    prob_terms = []
    for term in prob_arrays:
        print(term)
        arr, bin_size, low_bin = prob_arrays[term]
        prob_term = term + "_prob"
        idx = (np.floor(dfz[term] / bin_size) - low_bin).astype("int")
        is_low = (idx < 0)
        is_high = (idx >= len(arr) ).sum()
        low = np.min(idx)
        high = np.max(idx)

#         if (is_low.sum() > 0 or is_high.sum() > 0):
#             print("Warning: bounds overshoot on %s [%.3f, %.3f]"%
#                   (term, low_bin * bin_size, (low_bin + len(arr)) * bin_size))
#             print("Below: %i Below_median: %.3f Below_max: %.3f Above: %i Above_median: %.3f Above_max: %.3f"%(
#                 is_low.sum(), (0 - np.median(idx[is_low]))*bin_size, (0 - low)*bin_size,
#                 is_high.sum(), (np.median(idx[is_high]) - len(arr))*bin_size, (high-len(arr))*bin_size
#             ))

        idx = np.clip(idx, 0, len(arr)-1)
        dfz[prob_term] = arr[ idx ]
        prob_terms.append(prob_term)
    dfz[prob_name] = 1
    for prob_term in prob_terms:
        dfz[prob_name] *= dfz[prob_term]




def train_and_predict_mle(df, all_indices, test_indices, terms_and_cuts, prob_name, whole_df, graphs=False):

    use_indices = list(set(all_indices) - set(test_indices))

    test_df = df.iloc[test_indices].copy(True)
    use_df = df.iloc[use_indices].copy(True)


    prob_arrays = {}
    eqs = []
    for pilot_term in terms_and_cuts:
        cut, good_high, term, is_int = terms_and_cuts[pilot_term]
        low, high, bin_size, low_bin, num_bins = get_low_high_bin_size_low_bin_num_bins(whole_df, pilot_term, terms_and_cuts)
        prob_array, eqs = create_prob_array(eqs, low, high, low_bin, num_bins, bin_size, pilot_term, use_df, terms_and_cuts, graphs)
        prob_arrays[term] = (prob_array, bin_size, low_bin)

    apply_prob_arrays(test_df, prob_arrays, prob_name)

    return test_df[[prob_name, 'tag']], prob_arrays, eqs


