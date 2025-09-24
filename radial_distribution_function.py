import scipy
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from numpy import zeros, array, pi, sum, sqrt
import pandas as pd
import os
from scipy.optimize import curve_fit


def dist(pos1, pos2, box_size_2d):
    """Compute periodic distance between two points in 2D (y, x)."""
    d = np.abs(pos1 - pos2)
    #for dim in range(1,2):
     #   if d[dim] > box_size_2d[dim] / 2:
      #      print('pbc applied')
       #     d[dim] = box_size_2d[dim] - d[dim]
    return np.linalg.norm(d)

def rdf_2D(positions, box_size, dr):
    """
    Compute 2D radial distribution function (g(r)) for a 3D stack,
    projecting into the (y,x) plane.
    positions: array of shape (N, 3) in order [z, y, x]
    box_size: [y, x]
    """
    positions = np.asarray(positions)
    Np = len(positions)
    
    box_size_2d = box_size[1:]  # y, x
    rho_2d = Np / (box_size_2d[0] * box_size_2d[1])
    r_max = min(box_size_2d) / 2
    Nbins = int(r_max / dr)
    count = np.zeros(Nbins-1)
    r_edges = np.linspace(0, r_max, Nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for i in range(Np):
        for j in range(i + 1, Np):
            # Use only y and x components
            pos_i = positions[i][1:]  # [y, x]
            pos_j = positions[j][1:]
            d = dist(pos_i, pos_j, box_size_2d)
            if d < r_max:
                bin_idx = int(d / dr)
                if bin_idx < len(count):
                    count[bin_idx] += 2  # i-j and j-i

    g_r = np.zeros(Nbins-1)
    for i in range(Nbins-1):
        r_outer = r_edges[i+1]
        r_inner = r_edges[i]
        shell_area = np.pi * (r_outer**2 - r_inner**2)
        g_r[i] = count[i] / (rho_2d * Np * shell_area)

    return r_centers, g_r


def rdf_2D_sliced(positions, box_size, dr,z_gap, slice_tolerance):
    '''Computes the 2D g(r) by slicing the z stack according to the z resolution
    Takes: positions: array of shape (N, 3) in order [z, y, x]
           box_size: [z, y, x]
           dr: radial bin width
           z_gap: gap between z slices
           slice_tolerance: minimum number of particles in a slice to be considered valid
    Returns: r_centers, g_r_averaged average g(r) over all valid slices
    '''
    positions = np.asarray(positions)
    Np = len(positions)
    box_size_2d = box_size[1:]  # y, x
    r_max = min(box_size_2d) / 2
    Nbins = int(r_max / dr)
    r_edges = np.linspace(0, r_max, Nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    # bin in z with spacing z_gap
    z_vals = positions[:,0] # take the z positions
    z_max = np.max(z_vals)
    z_min = np.min(z_vals)
    rdf = []
    num_slices = int((z_max - z_min)/z_gap)

    for i in range(num_slices):
        z_lo = z_min + i*z_gap
        z_hi = z_lo + z_gap

        mask = (z_vals >= z_lo) & (z_vals < z_hi) # this is boolean, gives False or True depending on the condition
        slices = positions[mask][:, 1:] # the first : is to slice colum-wise, then 1: takes the y,x coord
        if len(slices) > slice_tolerance: # if there are enough particles in the slice
            rho_2d = len(slices) / (box_size_2d[0] * box_size_2d[1])
            count = np.zeros(Nbins)
            for k in range(len(slices)):
                for l in range(k + 1, len(slices)):
                    pos_k = slices[k] 
                    pos_l = slices[l]
                    d = dist(pos_k, pos_l, box_size_2d)
                    if d < r_max:
                        bin_idx = int(d / dr)
                        if bin_idx < len(count):
                            count[bin_idx] += 2  # i-j and j-i

            g_r = np.zeros(Nbins)
            for i in range(Nbins):
                r_outer = r_edges[i+1]
                r_inner = r_edges[i]
                shell_area = np.pi * (r_outer**2 - r_inner**2)
                g_r[i] = count[i] / (rho_2d * len(slices) * shell_area)

            rdf.append(g_r)
   
    print(f'rdf for every plane{rdf}')
    # average g(r) over all slices
    if len(rdf) == 0:
        raise ValueError("No valid slices with at least 2 particles were found.")           

    g_r_averaged = np.mean(rdf, axis = 0)
    #g_r_final = np.array(g_r)
 
    return r_centers, g_r_averaged
 
    
def rdf_3D(positions, box_size, dr):
    """
    Compute 3D radial distribution function (g(r)).
    positions: array of shape (N, 3) in order [z, y, x]
    box_size: [z, y, x]
    """
    positions = np.asarray(positions)
    Np = len(positions)
    
    rho = Np / (box_size[0] * box_size[1] * box_size[2])
    r_max = min(box_size) / 2
    Nbins = int(r_max / dr)
    count = np.zeros(Nbins-1)
    r_edges = np.linspace(0, r_max, Nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for i in range(Np):
        for j in range(i + 1, Np):
            #d = dist_2D_periodic(positions[i], positions[j], box_size)
            d = np.linalg.norm(positions[i] - positions[j])
            if d < r_max:
                bin_idx = int(d / dr)
                if bin_idx < len(count):
                    count[bin_idx] += 2  # i-j and j-i

    g_r = np.zeros(Nbins-1)
    for i in range(Nbins-1):
        r_outer = r_edges[i+1]
        r_inner = r_edges[i]
        shell_volume = (4/3) * pi * (r_outer**3 - r_inner**3)
        g_r[i] = count[i] / (rho * Np * shell_volume)

    return r_centers, g_r


def rdf_3D_restricted(positions, box_size, dr, thr):
    """
    Computes the 3D g(r) excluding edge particles as RDF centers.
    positions: array (N, 3) in order [z, y, x]
    box_size: [z, y, x]
    thr: tuple with (thr_z, thr_y, thr_x) defining excluded edge thickness
    """
    positions = np.asarray(positions)
    Np = len(positions)
    r_max = min(box_size) / 2
    Nbins = int(r_max / dr)
    count = np.zeros(Nbins)
    r_edges = np.linspace(0, r_max, Nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    thr_z, thr_y, thr_x = thr
    
    # Identify indices of "good" (non-edge) points
    good_indices = []
    for i in range(Np):
        if (positions[i, 0] > thr_z and positions[i, 0] < (box_size[0] - thr_z) and
            positions[i, 1] > thr_y and positions[i, 1] < (box_size[1] - thr_y) and
            positions[i, 2] > thr_x and positions[i, 2] < (box_size[2] - thr_x)):
            good_indices.append(i)
    N_ref = len(good_indices)

    # Number density
    rho = Np / np.prod(box_size)

    # Count pairs (only if i is a "good" center)
    for i in good_indices:
        for j in range(Np):
            if i == j:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            if d < r_max:
                bin_idx = int(d / dr)
                if bin_idx < len(count):
                    count[bin_idx] += 1

    # Normalize
    g_r = np.zeros(Nbins)
    for i in range(Nbins):
        r_outer = r_edges[i+1]
        r_inner = r_edges[i]
        shell_volume = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        g_r[i] = count[i] / (rho * N_ref * shell_volume)

    return r_centers, g_r


def critical_func(x, A, l, eta):
    """Critical function for fitting RDF peaks.
    g(r) = A * (exp(-r/l))/r^(1+eta) + 1
        A=amplitude l=correlation length eta=critical exponent
    """
    return A * (np.exp(-x/l))/np.power(x, 1+eta) + 1


def fit_rdf_critical(start, rdf, dists):
    """Fit the RDF using the critical function."""
    popt, pcov = curve_fit(critical_func, dists[start:], rdf[start:], p0=[1, 1, 1])
    return popt, pcov

