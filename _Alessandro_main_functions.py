# Coded by Alessandro Perrone (perrone.1900516@studenti.uniroma1.it)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import os
import imageio
import math
import ExtraFunctions as ef
import itertools
import trackpy as tp
import datetime
import matplotlib.animation as animation
import collections
from scipy.spatial.distance import pdist, squareform, cdist
from skimage import io
import scipy.ndimage as ndi
from scipy.ndimage._ni_support import _normalize_sequence
import tifffile as tiff
import ExtraFunctions as ef
import One_channel as oc
import xml.etree.ElementTree as ET
from collections import Counter

def make_network(patch_coo,dist_treshold,tolerance):
    '''
    This function makes a web: it connects patches that are close enough to be in the same particle. You can also set a minimum distance
    Takes:
        :patch_coo: location vector of every patch (dict)
        :dist_treshold: Max distance for patches still to be considered in the same particle (float)
        :dist_min: Min distance for patches still to be considered in the same particle (float) - of course if this filters stuff, it means there is a mistrack somewhere, but lets just ignore that fact for now.
    Returns:
        :curdict: A dict object containing ONLY the patches with a list of other patches connected to it.
    '''
    curdict = {}
    distance_dict = {}
    distance_vector_dict = {}
    for identifier1, co1 in patch_coo.items():
        # Every loop another patch
        #curdict[identifier1] = []
        #distance_dict[identifier1] = []
        #distance_vector_dict[identifier1] = []
        neighbors = []
        distances = []
        distance_vectors = []
        for identifier2, co2 in patch_coo.items(): # find distances to all other patches
            if identifier1 == identifier2:
                continue
            distancevector = [(co1[0]-co2[0]), (co1[1]-co2[1]), (co1[2]-co2[2])] # z,y,x
            distance = math.sqrt(distancevector[0]**2 + distancevector[1]**2 + distancevector[2]**2)
            if dist_treshold - tolerance < distance < dist_treshold + tolerance:
                neighbors.append(identifier2)
                distances.append(distance)
                distance_vectors.append(distancevector)
        if neighbors:
            curdict[identifier1] = neighbors
            distance_dict[identifier1] = distances
            distance_vector_dict[identifier1] = distance_vectors
        #try:
         #   curdict[identifier1].remove(identifier1) # if dist_min == 0, it will find itself, so make sure that does not happen.
        #except KeyError:
         #   pass
    return curdict, distance_dict, distance_vector_dict

def compute_angles_between_patches(vectors):
    ''' 
    This function computes the angle of the identifier patch and all of its neighbors.
    Takes: 
        :vectors: the dict of all the vectors connected to the identifier patch
    Returns: 
        :angles_dict: the angles of the identifier patch and its neighbors
    '''
    angles_dict = {}
    for identifier, vector in vectors.items():
        angles_dict[identifier] = []

        for i in range(len(vector)):
            for j in range(i+1,len(vector)):
                v1 = np.array(vector[i])
                v2 = np.array(vector[j])

                cos = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)) 
                #cos = np.clip(cos, -1.0,1.0)
                angle = math.acos(cos)

                angle_deg = math.degrees(angle)
                angles_dict[identifier].append(angle_deg)

    return  angles_dict               

def compute_all_distancesandangles(vectors):
    '''
    This function computes the distance and angles between all the vectors using itertools combinations.
    Takes: 
        :vectors: list of vectors (in z,y,x but it should not matter for the distances). 
    Returns:
        :vects: list of the distances vectors in the network. 
        :distances: list of the distances (scalars) between the vectors in the network. 
        :angles: list of the angles (in degrees) between the vectors in the network.
    '''
    vects = []
    distances = []
    angles = []
    #for identifier, vectors in vectors_in_network.items():
    for i, j in itertools.combinations(vectors,2):
        i = np.array(i)
        j = np.array(j)
        #print(f"i {i} j {j} for identifier {identifier}")
        vij = i - j
        vij_norm = np.sqrt((i[2]-j[2])**2 +(i[1]-j[1])**2+(i[0]-j[0])**2 ) # norm computed as sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)   #np.linalg.norm(vij)
        #print(f"||vij|| {vij_norm} for identifier {identifier}")
        if np.linalg.norm(i) > 0 and np.linalg.norm(j) > 0:
            cos = np.dot(i,j)/(np.linalg.norm(i) * np.linalg.norm(j)) # clip ensure that the value gets rounded to a good one
            #cos = np.clip(cos, -1.0,1.0)
            angle = math.acos(cos)
            angle_deg = math.degrees(angle)
        else:
            angle_deg = 0

        vects.append(vij)    
        distances.append(vij_norm)
        angles.append(angle_deg)

    return vects, distances, angles

def find_shape(network,vectors_in_network,dist_treshold,dist_tolerance,target_angle,ang_tolerance):
    ''' 
        This function reconstruct the shape we can find in the network based on the neighbors:
            2 neighbors --> look for regular triangles
            3 neighbors --> look for a regular tethraedron 
        Takes:  
            :network: dict of the patches in the network. Give the patch identifier and I give you its neighbors
            :vectors_in_network: dict of the vectors. Give the patch identifier and I give you the distance between its neighbors
            :target_angle: float angle value (60° in our case)
            :ang_tolerance: float of the tolerance on the target_angle
        Returns:
        :triangle_neighbors: dict of patches id in the triangles  
        :triangle_vectors: dict of distance vectors between patches in the triangle  
    '''
    triangle_neighbors = {}
    triangle_vectors = {}
    tetra_neighbors = {}
    tetra_vectors = {}
    #angles = compute_angles_between_patches(vectors_in_network) # compute the angles between the vectors in the network
    for identifier, neighbors in network.items():
        if len(neighbors) < 2:
            continue
        if len(neighbors) == 2:
            #d1 = distances_in_network[identifier][0]
            #d2 = distances_in_network[identifier][1]
            tr_vects, tr_dists, tr_angles = compute_all_distancesandangles(vectors_in_network[identifier]) # passing the function a list which contains only the vectors of the id
            #v1 = vectors_in_network[identifier][0] # vector distance between the identifier and the first neighbor
            #v2 = vectors_in_network[identifier][1] # vector distance between the identifier and the second neighbor
            #for angle in angles[identifier]: # angle between the identifier and the two neighbors
            for tr_d, tr_a in zip(tr_dists, tr_angles):
                if (target_angle - ang_tolerance < tr_a < target_angle + ang_tolerance): # to go over the angles
                    #d12 = np.array(v1) - np.array(v2)
                    #d12_norm = np.linalg.norm(d12)
                    if (dist_treshold - dist_tolerance < tr_d < dist_treshold + dist_tolerance): 
                        triangle_neighbors[identifier] = []
                        triangle_vectors[identifier] = []
                        triangle_neighbors[identifier].append(neighbors)
                        triangle_vectors[identifier].append(tr_dists) 

        if len(neighbors) == 3:
            vects, dists, angs = compute_all_distancesandangles(vectors_in_network[identifier]) # passing the function a list which contains only the vectors of the id
            # print(f"for id {identifier} distances between vertices {dists} and angles between vertices {angs}")
            for d,a in zip(dists, angs):
                if (target_angle - ang_tolerance < a < target_angle + ang_tolerance): # to go over the angles
                    if (dist_treshold - dist_tolerance < d < dist_treshold + dist_tolerance): # to go over the distances
                        tetra_neighbors[identifier] = []
                        tetra_vectors[identifier] = []
                        tetra_neighbors[identifier].append(neighbors) 
                        tetra_vectors[identifier].append(dists)

    return triangle_neighbors, triangle_vectors, tetra_neighbors, tetra_vectors      

def is_triangle_in_tetrahedron(cleaned_triangles_neighbors, cleaned_tetras_neighbors):
    """
    This function checks if any triangle is part of a tetrahedron.
    Takes: cleaned_triangles_neighbors: dict of the triangles without repetitions.
    cleaned_tetras_neighbors: dict of the tetrahedrons without repetitions.
    Returns: a list of triangles ids that are part of a tetrahedron
    """

    fake_triangles = []
    # Build a set of all tetrahedrons as frozensets of 4 patch IDs
    tetra_set = set()
    tetra_members = set()
    for tid, neighbors_list in cleaned_tetras_neighbors.items():
        for neighbors in neighbors_list:
            tetra = frozenset([tid] + neighbors)
            tetra_set.add(tetra)
            tetra_members.update([tid]+neighbors)

    # Check each triangle
    for id, neighbors_list in cleaned_triangles_neighbors.items():
        for neighbors in neighbors_list:
            triangle = frozenset([id] + neighbors)
            for tetra in tetra_set:
                if triangle.issubset(tetra):
                    fake_triangles.append(id)
                    break

    return fake_triangles                

def clusters(network, patch_coo):
    '''
    This function groups patches with more than 3 neighbors into clusters.
    Takes:
        :network: dict of the patches in the network. Give the patch identifier and I give you its neighbors
        :patch_coo: location vector of every patch (dict)
        Returns:
        :cleaned_cluster_members: dict of the clusters with the patch identifier and its neighbors withou repetitions
        :clusters_coordinates: dict of the clusters with THE PATCH IDENTIFIER and its neighbors coordinates without repetitions
        :'''
    #cluster_lengths = [len(item) for item in network.values() if len(item) > 3] # this is a list of the lengths of the lists in the network dict
    cluster_members = {}
    clusters_coordinates = {}
    for identifier, neighbors in network.items():
        if len(neighbors) > 3:
            # Initialize cluster members and coordinates
            cluster_members[identifier] = []
            clusters_coordinates[identifier] = []
            #for neighbor in neighbors:
            # Append neighbors to the cluster
            cluster_members[identifier].append(neighbors)

    cleaned_cluster_members = remove_repeated(cluster_members) # Remove duplicates

    for id, cleaned_neighbors in cleaned_cluster_members.items():
            flat_neighbors = [item for sublist in cleaned_neighbors for item in sublist] # Flatten the list of neighbors
            # Append the coordinates of the patch itself
            clusters_coordinates[id].append(patch_coo[id])

            # Append the coordinates of the neighbors
            for cleaned_neighbor in flat_neighbors:
                clusters_coordinates[id].append(patch_coo[cleaned_neighbor])

    return cleaned_cluster_members, clusters_coordinates            

def derive_patch_4(Avec,Bvec,Cvec):
    '''
    This uses an INTERWEBZZZ method: https://stackoverflow.com/questions/4372556/given-three-points-on-a-tetrahedron-find-the-4th
    Takes:
     :Avec,Bvec,Cvec: 3 vertices of a triangle (A,B,C) in numpy array vectors. Returns coordinates of 4th potential patch position, both the 'up' and 'down' version.

    '''
    d,co_all = [],[]
    for i,j in itertools.combinations([Avec,Bvec,Cvec], 2):
        #Gives lines AB, AC and BC in (a,b)
        co = i - j
        dist = np.linalg.norm(co) # Calculate distance between each patch
        d += [dist] 
        co_all += [co]
    av_d = np.mean(d) 
    center = ( Avec + Bvec + Cvec ) / 3 # Average your three points to get the center of the triangle
    normal = np.cross((Cvec - Avec) , (Bvec - Avec)) # Calculate the normal vector by taking the cross product of two of the sides
    unit_normal = normal / np.linalg.norm(normal) # Normalize the normal vector (make it of unit length)
    scaled_normal = unit_normal * (math.sqrt(2/3)*av_d) # Scale the normal by the height of regular tetrahedron
    top = center + scaled_normal
    bottom = center - scaled_normal
    return top, bottom

def reconstructed_tetras(patch_coo, triangles_neighbors):
    '''
    This uses an INTERWEBZZZ method: https://stackoverflow.com/questions/4372556/given-three-points-on-a-tetrahedron-find-the-4th
    Takes:
      :patch_coo: dict of all the patches in the network. 
      :triangles_neighbors: dict of the patches that form the triangles. Give the function the cleaned dict, so without duplicates.
    Returns:
        :tetras: a dict of the two possible tetrahedron patches for each triangle (up, down).  
    '''
    recostructed_tetras = {} # Create empty dict for the tetrahedrons

    # Loop through each patch and its triangles
    for identifier, neighbors_list in triangles_neighbors.items():
        for neighbors in neighbors_list:
            # Get the adjusted coordinates of the patch and its neighbors
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # COORDINATES of First neighbor
            p3 = patch_coo[neighbors[1]]  # COORDINATES of Second neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z

            top, bottom = derive_patch_4(p1,p2,p3) # this computes the possible tetraedron (4th patch up, 4th patch down)

            # Append the top and bottom vertices
            recostructed_tetras[identifier] = []
            #recostructed_tetras[identifier].append(( np.array(top), np.array(bottom) )) 
            recostructed_tetras[identifier] = (np.array(top),np.array(bottom))

    return recostructed_tetras

def check_angles(angles_dict, vectors_in_network, target_angle,ang_tolerance):
    ''' Useless'''
    good_angles_dict = {} # this dict contains ONLY the good angles
    good_vectors_dict = {} # this dict contains ONLY the vectors corresponding to the good angles

    for identifier, angles in angles_dict.items():
        # Only process patches with more than 3 neighbors
        if len(vectors_in_network[identifier]) < 2:
            continue

        good_angles = []
        good_vectors = []
        vector_list = vectors_in_network[identifier] # Get the vectors for this patch
        angle_index = 0 # create a counter to change angle
        for i in range(len(vector_list)):
            for j in range(i+1, len(vector_list)):
                angle = angles[angle_index] # Get the current angle
                if target_angle- ang_tolerance < angle < target_angle + ang_tolerance:
                    good_angles.append(angle)
                    good_vectors.append((vector_list[i],vector_list[j])) # store the 2 vectors that make the right angle
                angle_index += 1    
        if good_angles > 3: # if the list is not empty
            for k in good_angles:
                # compute the difference between the good_angles and the target angles and just keep the closest to target angles
                good_angles_dict[identifier] = good_angles
                good_vectors_dict[identifier] = good_vectors

        # pop out only the identifier of the good angles

    return good_angles_dict, good_vectors_dict    

def longest_key(network):
    '''
    This function finds the key with the longest list in the network and also how many times a certain length occurs in the dictionary. Then it prints it out.
    Takes:
        :network: dict of the patches in the network.
    '''
    # Find the key with the longest list in the network dictionary
    max_key = max(network, key=lambda k: len(network[k])) #key = lambda k: len(network[k])) determines the value to compare for each key. 
    max_length = len(network[max_key])

    print(f"The key with the longest list is: {max_key}")
    print(f"The length of the longest list is: {max_length}")
    print(f"The longest list is: {network[max_key]}")

    lengths = [len(item) for item in network.values()]
    length_counts = Counter(lengths) # this is a dict

    sorted_length_counts = dict(sorted(length_counts.items())) # sort the dict by key (length of the list)

    for length, count in sorted_length_counts.items():
        print(f"Length {length} occurs {count} times")

def is_shared(cleaned_tetras_neighbors):
    ''' 
    This function looks for shared ids between the tetraedrons. If ONLY ONE patch is contained in two NN lists then it appends it to a list.
     :Takes: cleaned_tetras_neighbors: dict of the tetrahedrons without repetitions.
     :Return: shared_patches_ids: ids that respect the condition
    '''
    shared_patches_ids = []
    
    for id, neighbors in cleaned_tetras_neighbors.items():
        current_neighbors = set([n for sublist in neighbors for n in sublist]) # Flat the list and convert it to a set
        for other_id, other_neighbors in cleaned_tetras_neighbors.items():
            if id == other_id:
                continue
            other_neighbors_flat = set([n for sublist in other_neighbors for n in sublist]) # Flat the neighbors list for the other ID

            # Check if exactly one patch is shared
            shared_patches = list(current_neighbors & other_neighbors_flat)
            if len(shared_patches) == 1:
                print(f"Tetrahedron {id} shares exactly one patch with {other_id}: {shared_patches}")
                shared_patches_ids.append([id, other_id, shared_patches[0]])
    
    return shared_patches_ids 

def remove_repeated(neighbors):
    """
    Removes repeated objects from the a dictionary.

    Parameters:
        neighbors : dict
            Dictionary where keys are patch identifiers and values are lists of neighbors of the patch.

    Returns:
        dict : A new dictionary with unique objects.
    """
    unique = set()  # Use a set to store unique tetrahedrons
    cleaned = {}    # Dictionary to store the cleaned tetras_neighbors

    for identifier, neighbors_list in neighbors.items():
        cleaned_neighbors = []
        for neighbors in neighbors_list:
            # Sort the list (including the identifier) to ensure uniqueness
            element = tuple(sorted([identifier] + neighbors))
            if element not in unique:
                unique.add(element)  # Add the tetrahedron to the set
                cleaned_neighbors.append(neighbors)  # Keep the neighbors in the cleaned list
        if cleaned_neighbors:
            cleaned[identifier] = cleaned_neighbors

    return cleaned

def find_centers(cleaned_tetras_neighbors, cleaned_triangles_neighbors, from_triangles_to_tetras, patch_coo):
    '''
    This function finds the centers of the tetrahedrons.
    Takes:
        :cleaned_tetras_neighbors: dict of the tetrahedrons WITHOUT repetitions.
        :cleaned_triangles_neighbors: dict of the triangles WITHOUT repetitions.
        :from_triangles_to_tetras: dict of the top/bottom vertex of the reconstructed tetrahedrons.
        :patch_coo: dict of the patches.
    Returns:
        :centers: dict of the centers of the tetrahedrons and triangles.
    '''
    real_centers = {}
    fake_centers = {}
    for identifier, triangles_neighbors_list in cleaned_triangles_neighbors.items():
        for neighbors in triangles_neighbors_list:
            # Ensure from_triangles_to_tetras[identifier] exists and is a tuple
            if identifier not in from_triangles_to_tetras or not isinstance(from_triangles_to_tetras[identifier], tuple):
                print(f"Skipping identifier {identifier} due to invalid data in from_triangles_to_tetras.")
                continue

            # Get the coordinates of the tetrahedron vertices
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # COORDINATES of First neighbor
            p3 = patch_coo[neighbors[1]]  # COORDINATES of Second neighbor
            top, bottom = from_triangles_to_tetras[identifier]  # Get the top and bottom vertices of the tetrahedron

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z

            # Calculate the center of the tetrahedron
            center_top = (p1 + p2 + p3 + top) / 4.0
            center_bottom = (p1 + p2 + p3 + bottom) / 4.0

            if identifier not in fake_centers:
                fake_centers[identifier] = []
            fake_centers[identifier].append((center_top, center_bottom))  # Append the centers as a tuple

    for identifier, tetras_neighbors_list in cleaned_tetras_neighbors.items():
        for tetra_neighbors in tetras_neighbors_list:
            # Get the coordinates of the tetrahedron vertices
            c1 = patch_coo[identifier]
            c2 = patch_coo[tetra_neighbors[0]]
            c3 = patch_coo[tetra_neighbors[1]]
            c4 = patch_coo[tetra_neighbors[2]]

            # Reorder the coordinates from z, y, x to x, y, z
            c1 = np.array([c1[2], c1[1], c1[0]])  # x, y, z
            c2 = np.array([c2[2], c2[1], c2[0]])  # x, y, z
            c3 = np.array([c3[2], c3[1], c3[0]])  # x, y, z
            c4 = np.array([c4[2], c4[1], c4[0]])  # x, y, z

            # Calculate the center of the tetrahedron
            center = (c1 + c2 + c3 + c4) / 4.0 

            if identifier not in real_centers:
                real_centers[identifier] = []
            real_centers[identifier].append(center)

    return fake_centers, real_centers

def check_particles_overlap(diameter, centers_df, tolerance):
    """
    This function selects which patches are good ones based on the criterion of overlapping circumscribed spheres.
    If the radii overlap, then the patch is removed from the patch coordinates.

    Parameters:
        diameter : float
            The diameter i.e. the center-to-center distance.
        centers_df : pd.DataFrame
            DataFrame containing the centers with columns ['ID', 'x', 'y', 'z'].

    Returns:
        updated_patch_coo : df
            DataFrame of the non-overlapping particles.
    """

    non_overlapping = pd.DataFrame(columns=['ID', 'x', 'y', 'z'])
    # Extract coordinates as a NumPy array
    coordinates = centers_df[['x', 'y', 'z']].to_numpy()
    ids = centers_df['ID'].to_numpy()


    # Compute pairwise distances using broadcasting
    pairwise_distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)

    pairwise_distances = np.round(pairwise_distances,2)

    diameter = round(diameter,2)

    # Identify overlapping IDs
    overlapping_ids = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if pairwise_distances[i, j] < diameter - tolerance:
                overlapping_ids.add(ids[i])
                overlapping_ids.add(ids[j])
                print(f"Overlap detected between {ids[i]} and {ids[j]} with distance radius {pairwise_distances[i, j]:.2f} < radius {diameter:.2f}")

    print(f"there are {len(overlapping_ids)} overlapping ids {overlapping_ids} ")            

    # Filter out overlapping particles
    non_overlapping = centers_df[~centers_df['ID'].isin(overlapping_ids)].reset_index(drop=True) # ~: boolean NOT operator (flips isin() from True to False and viceversa), drop=True: do not add the old index as a column

    return non_overlapping, list(overlapping_ids)









