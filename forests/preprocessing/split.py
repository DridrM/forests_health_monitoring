import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def split_array(arr: np.array, n_tiles = 4) -> list:
    """Split an array into perfect square number equal tiles
    The number of tiles n_tiles must be a perfect square"""
    
    # Output tile list
    list_split = []
    
    # Compute sqrt of n_tiles for row and col split number
    axis_split = int(np.sqrt(n_tiles))
    
    # Split first vertically, and then horizontally
    for tiles in np.array_split(arr, axis_split, axis = 0):
        for tile in np.array_split(tiles, axis_split, axis = 1):
            list_split.append(tile)
    
    return list_split


def plot_splitted_image(split_image: list) -> None:
    """Plot a splitted image"""
    
    # The number of lines and columns of the plot
    axis_split = int(np.sqrt(len(split_image)))

    # List of indices in order to navigate into axs
    indices_list = list(product([i for i in range(axis_split)], repeat = 2))

    # Instanciate the figure and the axes
    fig, axs = plt.subplots(nrows = axis_split, ncols = axis_split)

    # Iterate over the axs and create subplots with sub images
    for tile, (i, j) in zip(split_image, indices_list):
        axs[i, j].imshow(tile)

    plt.show()

