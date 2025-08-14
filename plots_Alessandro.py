# Coded by Alessandro Perrone (perrone.1900516@studenti.uniroma1.it)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import matplotlib.animation as animation
import plotly.graph_objects as go



'''

This is a code to plot patches/particles in a box in 3D using matplotlib and plotly. In particular you have the possibility to also plot triangles and tetrahedrons in 3D 
in an intercative fashion.
The box edges are given by the user as parameters of the functions. 
The coordinates of the patches/particles have to be given in a dictionary, where the keys are the patch identifiers and the values are the coordinates of the patches/particles.
To modify the markers, colors, ... in the plotly plotter functions, you can modify the parameters of the go.Scatter3d function, it's just a dictionary.
PLEASE NOTE THAT THE COORDINATES OF THE PATCHES/PARTICLES HAVE TO BE IN THE FORMAT (z, y, x) AND NOT (x, y, z) AS USUAL a part from the code that plots the box,
where the coordinates are in the format (x, y, z), and the draw_tetrahedrons function, where the coordinates are in the format (x, y, z). In the other functions the 
coordinates are reordered to (x, y, z) for correct plotting.
Sorry about that, but you can always modify the lines of code that reorders the coordinates if your data are already in (x, y, z).

Code by Alessandro Perrone, 2025.
'''


def draw_box(ax, box_length):
        # Draw the box
    box_lines = [
        [[0, 0], [0, 0], [0, box_length[0]]],
        [[0, 0], [0, box_length[1]], [0, 0]],
        [[0, 0], [box_length[1], box_length[1]], [0, box_length[0]]],
        [[0, 0], [0, box_length[1]], [box_length[0], box_length[0]]],
        [[box_length[2], box_length[2]], [0, 0], [0, box_length[0]]],
        [[box_length[2], box_length[2]], [0, box_length[1]], [0, 0]],
        [[box_length[2], box_length[2]], [box_length[1], box_length[1]], [0, box_length[0]]],
        [[box_length[2], box_length[2]], [0, box_length[1]], [box_length[0], box_length[0]]],
        [[0, box_length[2]], [0, 0], [0, 0]],
        [[0, box_length[2]], [box_length[1], box_length[1]], [0, 0]],
        [[0, box_length[2]], [0, 0], [box_length[0], box_length[0]]],
        [[0, box_length[2]], [box_length[1], box_length[1]], [box_length[0], box_length[0]]]
    ]

    for line in box_lines:
        ax.plot(line[0], line[1], line[2], color="black")

    # Set the limits of the box
    ax.set_xlim([0, box_length[2]])
    ax.set_ylim([0, box_length[1]])
    ax.set_zlim([0, box_length[0]])


def draw_triangles_matplotlib(triangles_neighbors, patch_coo, network, box_length):
    """
    Visualize the triangles in 3D within a box after shifting the coordinates by box_length.

    Parameters:
        triangles_neighbors : dict where keys are patch identifiers and values are lists of tuples of neighbors forming triangles.
        patch_coo : dict where keys are patch identifiers and values are their 3D coordinates.
        box_length : tuple a 3D array with the dimensions of the box in the format (z, y, x).
    """
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    box_length = np.array(box_length)
    # Draw the box
    draw_box(ax, box_length)
    
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]]) # in x,y,z
        if patch_id in network.keys():
            ax.scatter(*coord, color='green', marker='.', alpha = 0.5, label='In network' if patch_id == list(network.keys())[0] else "")
        else:
            ax.scatter(*coord, color='black', marker='.', alpha = 0.5, label='Patch' if patch_id == list(patch_coo.keys())[0] else "")


    # Loop through each patch and its triangles
    for identifier, neighbors_list in triangles_neighbors.items():
        for neighbors in neighbors_list:
            # Get the adjusted coordinates of the patch and its neighbors
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # First neighbor
            p3 = patch_coo[neighbors[1]]  # Second neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z
            
            # Create a triangle
            triangle = [p1, p2, p3]

            # Plot the triangle as a filled polygon
            ax.add_collection3d(Poly3DCollection([triangle], alpha=0.5, edgecolor='k'))

            # Plot the vertices of the triangle
            ax.scatter(*p1, color='darkgreen', marker = '1', label='Patch' if identifier == list(triangles_neighbors.keys())[0] else "")
            ax.scatter(*p2, color='darkorange', marker = '.', label='Triangle Neighbor' if identifier == list(triangles_neighbors.keys())[0] else "")
            ax.scatter(*p3, color='darkorange', marker = '.', label='Triangle Neighbor'if identifier == list(triangles_neighbors.keys())[0] else "")

            # Add the identifier as text labels
            ax.text(*p1, f'{identifier}', color='black', fontsize=8)
            ax.text(*p2, f'{neighbors[0]}', color='black', fontsize=8)
            ax.text(*p3, f'{neighbors[1]}', color='black', fontsize=8)

    # Set labels and title
    ax.set_xlabel('X in $\mu$m')
    ax.set_ylabel('Y in $\mu$m')
    ax.set_zlabel('Z in $\mu$m')
    ax.set_title('3D Visualization of Particles and Triangles')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def draw_tetrahedrons_matplotlib(triangles_neighbors, patch_coo, network, reconstructed_tetras, box_length):
    """
    Visualize the tetrahedrons in 3D within a box using matplotlib.

    Parameters:
        triangles_neighbors : dict
            Dictionary where keys are patch identifiers and values are lists of tuples of neighbors forming triangles.
        patch_coo : dict
            Dictionary where keys are patch identifiers and values are their 3D coordinates.
        reconstructed_tetras : dict
            Dictionary containing the top and bottom vertices of the tetrahedrons.
        box_length : tuple
            A 3D array with the dimensions of the box in the format (z, y, x).
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    box_length = np.array(box_length)

    # Draw the box
    draw_box(ax, box_length)

    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]]) # in x,y,z
        if patch_id in network.keys():
            ax.scatter(*coord, color='green', marker='.', alpha = 0.5, label='In network' if patch_id == list(network.keys())[0] else "")
        else:
            ax.scatter(*coord, color='black', marker='.', alpha = 0.5, label='Patch' if patch_id == list(patch_coo.keys())[0] else "")

    # Loop through each triangle and its reconstructed tetrahedron
    for identifier, neighbors_list in triangles_neighbors.items():
        for neighbors in neighbors_list:
            # Get the coordinates of the triangle vertices
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # First neighbor
            p3 = patch_coo[neighbors[1]]  # Second neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z

            # Plot the triangle as a filled polygon
            triangle = [p1, p2, p3]
            ax.add_collection3d(Poly3DCollection([triangle], alpha=0.5, edgecolor='k'))

            # Plot the vertices of the triangle
            ax.scatter(*p1, color='darkgreen', marker='1', label='Patch' if identifier == list(triangles_neighbors.keys())[0] else "")
            ax.scatter(*p2, color='darkorange', marker='.', label='Triangle Neighbor' if identifier == list(triangles_neighbors.keys())[0] else "")
            ax.scatter(*p3, color='darkorange', marker='.')

            # Add the tetrahedron (top and bottom vertices)
            if identifier in reconstructed_tetras:
                for top, bottom in reconstructed_tetras[identifier]:
                    # Plot the top vertex
                    ax.scatter(*top, color='blue', marker='.', label='Top Vertex' if identifier == list(reconstructed_tetras.keys())[0] else "")
                    # Plot the bottom vertex
                    ax.scatter(*bottom, color='red', marker='.', label='Bottom Vertex' if identifier == list(reconstructed_tetras.keys())[0] else "")

                    # Add triangular faces connecting the top vertex to the triangle
                    ax.add_collection3d(Poly3DCollection([[p1, p2, top]], alpha=0.5, edgecolor='blue'))
                    ax.add_collection3d(Poly3DCollection([[p2, p3, top]], alpha=0.5, edgecolor='blue'))
                    ax.add_collection3d(Poly3DCollection([[p3, p1, top]], alpha=0.5, edgecolor='blue'))

                    # Add triangular faces connecting the bottom vertex to the triangle
                    ax.add_collection3d(Poly3DCollection([[p1, p2, bottom]], alpha=0.5, edgecolor='red'))
                    ax.add_collection3d(Poly3DCollection([[p2, p3, bottom]], alpha=0.5, edgecolor='red'))
                    ax.add_collection3d(Poly3DCollection([[p3, p1, bottom]], alpha=0.5, edgecolor='red'))

    # Set labels and title
    ax.set_xlabel('X in $\mu$m')
    ax.set_ylabel('Y in $\mu$m')
    ax.set_zlabel('Z in $\mu$m')
    ax.set_title('3D Visualization of Tetrahedrons')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()




def draw_patches_in_box(patch_coo, box_length):
    """
    Visualize the patches in 3D within a box using Plotly.

    Parameters:
        patch_coo : dict
            Dictionary where keys are patch identifiers and values are their 3D coordinates.
        box_length : tuple
            Dimensions of the box in the format (z, y, x).
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Plot all patches
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]])  # Reorder to x, y, z
        fig.add_trace(go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.8),
            #text=[f'{patch_id}'],  # Add patch ID as text
            #textposition='top center',
            #name='Patch'
        ))

    # Add the box
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Visualization of Patches in the Box',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            aspectmode='data',  # Ensures the aspect ratio is correct

        ),
        width=800,
        height=800,
        showlegend=False
    )

    # Show the plot
    fig.show()


def draw_triangles(triangles_neighbors, patch_coo, network, box_length):
    """
    Visualize the triangles in 3D within a box using Plotly.

    Parameters:
        triangles_neighbors : dict
            Dictionary where keys are patch identifiers and values are lists of tuples of neighbors forming triangles.
        patch_coo : dict
            Dictionary where keys are patch identifiers and values are their 3D coordinates.
        network : dict
            Dictionary containing the patches that are part of the network.
        box_length : tuple
            A 3D array with the dimensions of the box in the format (z, y, x).
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Plot all patches
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]])  # Reorder to x, y, z
        if patch_id in network.keys():
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers', # +text',
                marker=dict(size=2, color='green', opacity=0.8),
                text=[f'{patch_id}'],
                textposition='top center',
                name='In Network'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers', # +text',
                marker=dict(size=2, color='black', opacity=0.5),
                text=[f'{patch_id}'],
                textposition='top center',
                name='Patch'
            ))

    # Plot triangles
    for identifier, neighbors_list in triangles_neighbors.items():
        for neighbors in neighbors_list:
            # Get the adjusted coordinates of the patch and its neighbors
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # First neighbor
            p3 = patch_coo[neighbors[1]]  # Second neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z

            # Add the triangle as a mesh
            fig.add_trace(go.Mesh3d(
                x=[p1[0], p2[0], p3[0]],
                y=[p1[1], p2[1], p3[1]],
                z=[p1[2], p2[2], p3[2]],
                color='purple',
                opacity=0.5,
                name='Triangle'
            ))

    # Add the box
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Visualization of Particles and Triangles',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            aspectmode='data',  # Ensures the aspect ratio is correct

        ),
        width=800,
        height=800,
        showlegend=False
    )

    # Show the plot
    fig.show()


def draw_tetrahedrons(triangles_neighbors, patch_coo, network, reconstructed_tetras, box_length):
    """
    Visualize the tetrahedrons in 3D within a box using Plotly.

    Parameters:
        triangles_neighbors : dict
            Dictionary where keys are patch identifiers and values are lists of tuples of neighbors forming triangles.
        patch_coo : dict
            Dictionary where keys are patch identifiers and values are their 3D coordinates.
        network : dict
            Dictionary containing the patches that are part of the network.
        reconstructed_tetras : dict
            Dictionary containing the top and bottom vertices of the tetrahedrons.
        box_length : tuple
            A 3D array with the dimensions of the box in the format (z, y, x).
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Plot all patches
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]])  # Reorder to x, y, z
        if patch_id in network.keys():
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                text=[f'{patch_id}'],
                textposition='top center',
                name='In Network'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=2, color='black', opacity=0.5),
                text=[f'{patch_id}'],
                textposition='top center',
                name='Patch'
            ))

    # Plot triangles and tetrahedrons
    for identifier, neighbors_list in triangles_neighbors.items():
        for neighbors in neighbors_list:
            # Get the coordinates of the triangle vertices
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # First neighbor
            p3 = patch_coo[neighbors[1]]  # Second neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z

            # Add the triangle as a mesh
            fig.add_trace(go.Mesh3d(
                x=[p1[0], p2[0], p3[0]],
                y=[p1[1], p2[1], p3[1]],
                z=[p1[2], p2[2], p3[2]],
                color='orange',
                opacity=0.5,
                name='Triangle'
            ))

            # Add the tetrahedron (top and bottom vertices)
            if identifier in reconstructed_tetras:
                #for top_bottom in reconstructed_tetras[identifier]:
                    #top, bottom = top_bottom  # Unpack the tuple
                    top, bottom = reconstructed_tetras[identifier]
                    # Plot the top and bottom vertices
                    fig.add_trace(go.Scatter3d(
                        x=[top[0]],
                        y=[top[1]],
                        z=[top[2]],
                        mode='markers',
                        marker=dict(size=2, color='blue', symbol = 'x', opacity=0.8),
                        name='Top Vertex'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[bottom[0]],
                        y=[bottom[1]],
                        z=[bottom[2]],
                        mode='markers',
                        marker=dict(size=2, color='red', symbol = 'x', opacity=0.8),
                        name='Bottom Vertex'
                    ))

                    # Add lines connecting the triangle vertices to the bottom vertex
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], bottom[0]],
                        y=[p1[1], bottom[1]],
                        z=[p1[2], bottom[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Bottom Edge'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[p2[0], bottom[0]],
                        y=[p2[1], bottom[1]],
                        z=[p2[2], bottom[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Bottom Edge'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[p3[0], bottom[0]],
                        y=[p3[1], bottom[1]],
                        z=[p3[2], bottom[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Bottom Edge'
                    ))

                    # Add triangular faces connecting the top vertex to the triangle
                    fig.add_trace(go.Mesh3d(
                        x=[p1[0], p2[0], top[0]],
                        y=[p1[1], p2[1], top[1]],
                        z=[p1[2], p2[2], top[2]],
                        color='blue',
                        opacity=0.2,
                        name='Top Face'
                    ))
                    fig.add_trace(go.Mesh3d(
                        x=[p2[0], p3[0], top[0]],
                        y=[p2[1], p3[1], top[1]],
                        z=[p2[2], p3[2], top[2]],
                        color='blue',
                        opacity=0.2,
                        name='Top Face'
                    ))
                    fig.add_trace(go.Mesh3d(
                        x=[p3[0], p1[0], top[0]],
                        y=[p3[1], p1[1], top[1]],
                        z=[p3[2], p1[2], top[2]],
                        color='blue',
                        opacity=0.2,
                        name='Top Face'
                    ))

                    # Add triangular faces connecting the bottom vertex to the triangle
                    fig.add_trace(go.Mesh3d(
                        x=[p1[0], p2[0], bottom[0]],
                        y=[p1[1], p2[1], bottom[1]],
                        z=[p1[2], p2[2], bottom[2]],
                        color='red',
                        opacity=0.2,
                        name='Bottom Face'
                    ))
                    fig.add_trace(go.Mesh3d(
                        x=[p2[0], p3[0], bottom[0]],
                        y=[p2[1], p3[1], bottom[1]],
                        z=[p2[2], p3[2], bottom[2]],
                        color='red',
                        opacity=0.2,
                        name='Bottom Face'
                    ))
                    fig.add_trace(go.Mesh3d(
                        x=[p3[0], p1[0], bottom[0]],
                        y=[p3[1], p1[1], bottom[1]],
                        z=[p3[2], p1[2], bottom[2]],
                        color='red',
                        opacity=0.2,
                        name='Bottom Face'
                    ))

    # Add the box
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Visualization of Tetrahedrons',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            aspectmode='data',  # Ensures the aspect ratio is correct

        ),
        width=800,
        height=800,
        showlegend=False
    )

    # Show the plot
    fig.show()


def draw_real_tetraedrons(tetras_neighbors, patch_coo, network, box_length, triangles_neighbors=None):
    """
    Visualize the tetrahedrons in 3D within a box using Plotly.

    Parameters:
        tetras_neighbors : dict
            Dictionary where keys are patch identifiers and values are lists of neighbors forming tetrahedrons.
        patch_coo : dict
            Dictionary where keys are patch identifiers and values are their 3D coordinates.
        network : dict
            Dictionary containing the patches that are part of the network.
        box_length : tuple
            A 3D array with the dimensions of the box in the format (z, y, x).
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Plot all patches
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]])  # Reorder to x, y, z
        if patch_id in network.keys():
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                #name = [f'{patch_id}'],
                text=[f'{patch_id}'],
                textposition ='top center',
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=1, color='black', opacity=0.5),
                text=[f'{patch_id}'],
                textposition='top center',
                name='Patch',
                hoverinfo= 'text'
            ))

    # Plot tetrahedrons
    for identifier, neighbors_list in tetras_neighbors.items():
        for neighbors in neighbors_list:
            # Get the coordinates of the tetrahedron vertices
            p1 = patch_coo[identifier]  # The patch itself
            p2 = patch_coo[neighbors[0]]  # First neighbor
            p3 = patch_coo[neighbors[1]]  # Second neighbor
            p4 = patch_coo[neighbors[2]]  # Third neighbor

            # Reorder the coordinates from z, y, x to x, y, z
            p1 = np.array([p1[2], p1[1], p1[0]])  # x, y, z
            p2 = np.array([p2[2], p2[1], p2[0]])  # x, y, z
            p3 = np.array([p3[2], p3[1], p3[0]])  # x, y, z
            p4 = np.array([p4[2], p4[1], p4[0]])  # x, y, z

            # Add triangular faces of the tetrahedron
            fig.add_trace(go.Mesh3d(
                x=[p1[0], p2[0], p3[0]],
                y=[p1[1], p2[1], p3[1]],
                z=[p1[2], p2[2], p3[2]],
                color='orange',
                opacity=0.5,
                name='Tetrahedron Face',
                hoverinfo='none'
            ))
            fig.add_trace(go.Mesh3d(
                x=[p1[0], p2[0], p4[0]],
                y=[p1[1], p2[1], p4[1]],
                z=[p1[2], p2[2], p4[2]],
                color='orange',
                opacity=0.5,
                name='Tetrahedron Face',
                hoverinfo='none'
            ))
            fig.add_trace(go.Mesh3d(
                x=[p1[0], p3[0], p4[0]],
                y=[p1[1], p3[1], p4[1]],
                z=[p1[2], p3[2], p4[2]],
                color='orange',
                opacity=0.5,
                name='Tetrahedron Face',
                hoverinfo='none'
            ))
            fig.add_trace(go.Mesh3d(
                x=[p2[0], p3[0], p4[0]],
                y=[p2[1], p3[1], p4[1]],
                z=[p2[2], p3[2], p4[2]],
                color='orange',
                opacity=0.5,
                name='Tetrahedron Face',
                hoverinfo='none'
            ))

            # Add lines connecting the vertices
            edges = [
                (p1, p2), (p1, p3), (p1, p4),
                (p2, p3), (p2, p4), (p3, p4)
            ]
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[edge[0][0], edge[1][0]],
                    y=[edge[0][1], edge[1][1]],
                    z=[edge[0][2], edge[1][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Tetrahedron Edge',
                    hoverinfo='none'
                ))

    
    if  triangles_neighbors:
        draw_triangles(triangles_neighbors, patch_coo, network, box_length)
    # Add the box
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Visualization of Tetrahedrons',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            aspectmode='data',  # Ensures the aspect ratio is correct
        ),
        width=800,
        height=800,
        showlegend=False
    )

    # Show the plot
    fig.show()


def draw_real_tetrahedrons_and_triangles(tetras_neighbors, triangles_neighbors, patch_coo, network, box_length):
    """
    Plot real tetrahedrons and triangles in the same Plotly 3D plot.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    # Plot all patches
    for patch_id, coord in patch_coo.items():
        coord = np.array([coord[2], coord[1], coord[0]])  # Reorder to x, y, z
        if patch_id in network.keys():
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                text=[f'{patch_id}'],
                textposition='top center',
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(size=1, color='black', opacity=0.5),
                text=[f'{patch_id}'],
                textposition='top center',
                name='Patch',
                hoverinfo='text'
            ))

    # Plot tetrahedrons
    for identifier, neighbors_list in tetras_neighbors.items():
        for neighbors in neighbors_list:
            p1 = patch_coo[identifier]
            p2 = patch_coo[neighbors[0]]
            p3 = patch_coo[neighbors[1]]
            p4 = patch_coo[neighbors[2]]
            p1 = np.array([p1[2], p1[1], p1[0]])
            p2 = np.array([p2[2], p2[1], p2[0]])
            p3 = np.array([p3[2], p3[1], p3[0]])
            p4 = np.array([p4[2], p4[1], p4[0]])
            # Add tetrahedron faces
            faces = [
                (p1, p2, p3), (p1, p2, p4), (p1, p3, p4), (p2, p3, p4)
            ]
            for face in faces:
                fig.add_trace(go.Mesh3d(
                    x=[face[0][0], face[1][0], face[2][0]],
                    y=[face[0][1], face[1][1], face[2][1]],
                    z=[face[0][2], face[1][2], face[2][2]],
                    color='orange',
                    opacity=0.5,
                    name='Tetrahedron Face',
                    hoverinfo='none',
                    showscale=False
                ))

            #Add lines connecting the vertices
            edges = [
                (p1, p2), (p1, p3), (p1, p4),
                (p2, p3), (p2, p4), (p3, p4)
            ]
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[edge[0][0], edge[1][0]],
                    y=[edge[0][1], edge[1][1]],
                    z=[edge[0][2], edge[1][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Tetrahedron Edge',
                    hoverinfo='none'                
                ))            

    # Plot triangles
    for identifier_triangle, neighbors_list_triangle in triangles_neighbors.items():
        for neighbors_triangle in neighbors_list_triangle:
            p1_triangle = patch_coo[identifier_triangle]
            p2_triangle = patch_coo[neighbors_triangle[0]]
            p3_triangle = patch_coo[neighbors_triangle[1]]
            p1_triangle = np.array([p1_triangle[2], p1_triangle[1], p1_triangle[0]])
            p2_triangle= np.array([p2_triangle[2], p2_triangle[1], p2_triangle[0]])
            p3_triangle = np.array([p3_triangle[2], p3_triangle[1], p3_triangle[0]])
            fig.add_trace(go.Mesh3d(
                x=[p1_triangle[0], p2_triangle[0], p3_triangle[0]],
                y=[p1_triangle[1], p2_triangle[1], p3_triangle[1]],
                z=[p1_triangle[2], p2_triangle[2], p3_triangle[2]],
                color='purple',
                opacity=0.5,
                name='Triangle',
                hoverinfo='none',
                showscale=False
            ))
            #Add lines connecting the vertices
            edges = [
                (p1_triangle, p2_triangle), (p1_triangle, p3_triangle),
                (p2_triangle, p3_triangle)
            ]
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[edge[0][0], edge[1][0]],
                    y=[edge[0][1], edge[1][1]],
                    z=[edge[0][2], edge[1][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Triangle Edge',
                    hoverinfo='none'                
                ))                 

    # Add the box
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Visualization of Real Tetrahedrons and Triangles',
        scene=dict(
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            zaxis_title='Z (μm)',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            aspectmode='data'
        ),
        width=800,
        height=800,
        showlegend=False
    )

    fig.show()



def draw_spheres_from_dataframe(df, radius, box_length):
    """
    Draw spheres for the centers contained in the DataFrame using Plotly.

    Parameters:
        df : pd.DataFrame
            DataFrame containing the centers with columns ['ID', 'x', 'y', 'z'].
        radius : float
            Radius of the spheres to be drawn.
        box_length : tuple
            Dimensions of the box in the format (z, y, x).
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Loop through each row in the DataFrame
    for _, row in df.iterrows():
        x_center, y_center, z_center = row['x'], row['y'], row['z']

        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, 7)  # Reduced resolution for performance
        v = np.linspace(0, np.pi, 7)
        x = x_center + radius * np.outer(np.cos(u), np.sin(v))
        y = y_center + radius * np.outer(np.sin(u), np.sin(v))
        z = z_center + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Add the sphere as a mesh to the figure
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=1,
            colorscale='Blues',
            showscale=False
        ))

    # Update layout with box dimensions
    fig.update_layout(
        title='3D Visualization of Spheres',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[0, box_length[2]]),
            yaxis=dict(range=[0, box_length[1]]),
            zaxis=dict(range=[0, box_length[0]]),
            #aspectmode='data'  # Ensures the aspect ratio is correct
        ),
        width=800,
        height=800
    )

    # Add the box to the plot
    add_box_to_plot(fig, box_length)

    # Show the plot
    fig.show()


def plot_clusters_with_labels(patch_coo, cluster_members, box_length):
    """
    Plots clusters in 3D using Plotly, labels each point with the key and the list contained in cluster_members,
    and assigns a unique color to clusters based on their length. Additionally, connects patches in the same neighbor list with lines.

    Args:
        patch_coo (dict): Dictionary containing patch coordinates.
                          Format: {patch_id: [z, y, x], ...}
        cluster_members (dict): Dictionary containing cluster members.
                                Format: {cluster_id: [[member1, member2, ...], ...]}
        box_length (tuple): Dimensions of the box in the format (z, y, x).
    """
    # Define a color map for cluster lengths
    color_palette = [
        'orange', 'blue', 'green', 'purple', 'red', 'black', 'cyan', 'magenta', 'brown'
    ]

    # Create a Plotly figure
    fig = go.Figure()

    # Create a mapping for cluster lengths and their colors
    cluster_length_color_map = {}

    # Iterate through clusters
    for cluster_id, neighbors in cluster_members.items():
        if not neighbors:
            print(f'Cluster {cluster_id} has no neighbors')
            continue

        # Get the coordinates of the cluster patch itself
        if cluster_id not in patch_coo:
            print(f"Cluster ID {cluster_id} not found in patch_coo")
            continue
        cluster_coord = patch_coo[cluster_id]
        cluster_coord = [cluster_coord[2], cluster_coord[1], cluster_coord[0]]  # Convert z, y, x to x, y, z

        # Get the coordinates of the neighbors
        neighbor_coords = []
        for neighbor_list in neighbors:
            for neighbor_id in neighbor_list:
                if neighbor_id in patch_coo:
                    coord = patch_coo[neighbor_id]
                    neighbor_coords.append([coord[2], coord[1], coord[0]])  # Convert z, y, x to x, y, z
                else:
                    print(f"Neighbor ID {neighbor_id} not found in patch_coo")

        # Determine cluster length and assign a color
        cluster_length = len(neighbor_coords) + 1
        color = color_palette[(cluster_length - 1) % len(color_palette)]  # Cycle through colors if needed

        # Add the cluster length and color to the map (for legend)
        if cluster_length not in cluster_length_color_map:
            cluster_length_color_map[cluster_length] = color

        # Add the cluster patch itself (main patch)
        fig.add_trace(go.Scatter3d(
            x=[cluster_coord[0]],
            y=[cluster_coord[1]],
            z=[cluster_coord[2]],
            mode='markers',
            marker=dict(size=2, color=color),
            name=f'Cluster {cluster_id} - Main Patch',
            text=f'Cluster {cluster_id} - Main Patch',
            hoverinfo='text'
        ))

        # Add the neighbors (blue points)
        if neighbor_coords:
            x, y, z = zip(*neighbor_coords)
            hover_texts = [f'Neighbor of cluster {cluster_id}' for _ in neighbor_coords]
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=2, color=color),
                name=f'Cluster {cluster_id} - Neighbors',
                text=hover_texts,
                hoverinfo='text'
            ))

        # Connect the cluster patch to its neighbors with lines
        for neighbor_coord in neighbor_coords:
            fig.add_trace(go.Scatter3d(
                x=[cluster_coord[0], neighbor_coord[0]],
                y=[cluster_coord[1], neighbor_coord[1]],
                z=[cluster_coord[2], neighbor_coord[2]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='none'
            ))

    # Add legend entries for cluster lengths
    for cluster_length, color in cluster_length_color_map.items():
        fig.add_trace(go.Scatter3d(
            x=[None],  # Dummy point for legend
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(size=2, color=color),
            name=f'Clusters of Length {cluster_length}'
        ))

    # Add the box to the plot
    add_box_to_plot(fig, box_length)

    # Update layout
    fig.update_layout(
        title='3D Cluster Visualization',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='data'  # Ensures the aspect ratio is correct
        ),
        width=800,
        height=800,
        showlegend=True
    )

    # Show the plot
    fig.show()



def add_box_to_plot(fig, box_length):
    """
    Adds a 3D box to the Plotly figure.

    Parameters:
        fig : plotly.graph_objects.Figure
            The Plotly figure to which the box will be added.
        box_length : tuple
            Dimensions of the box in the format (z, y, x).
    """
    # Define the box vertices
    x = [0, 0, box_length[2], box_length[2], 0, 0, box_length[2], box_length[2]]
    y = [0, box_length[1], box_length[1], 0, 0, box_length[1], box_length[1], 0]
    z = [0, 0, 0, 0, box_length[0], box_length[0], box_length[0], box_length[0]]

    # Define the box edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Add the edges to the plot
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[x[edge[0]], x[edge[1]]],
            y=[y[edge[0]], y[edge[1]]],
            z=[z[edge[0]], z[edge[1]]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))





