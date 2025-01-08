 ### Dwell Calculations      
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import scipy.sparse as scsp
import scipy.optimize as scop
from scipy.spatial import KDTree
import scipy.ndimage as snd

debug_mode = 'full'
"""
full
checks
output
none

"""

def debug(message, thresh='full'):
    
    thresh_meaning = {'full': 0, 'checks': 1, 'output': 2, 'none': 3 }
    
    requested_level = thresh_meaning[debug_mode]
    required_level = thresh_meaning[thresh]
    
    if requested_level <= required_level:
        print(message)

 
 
def calculate_prox_mat(self, layer_number):
    coords, resists = self.get_points_and_resistances(layer_number)
    tree = KDTree(coords)
    distance_threshold = 3*self.sigma
    distance_mat = tree.sparse_distance_matrix(tree, distance_threshold, output_type="coo_matrix")
    resistance = resists[distance_mat.row]
    prox_matrix = distance_mat.copy()
    prox_matrix.data = self.proximity_function(distance_mat.data, resistance)
    return prox_matrix

def solve_layer(self, proximity_matrix, dz, tol=1e-2):
    "From F3ast"
    upper_bound = dz / proximity_matrix.diagonal()
    y = dz*np.ones(proximity_matrix.shape[1])
    result = scop.lsq_linear(proximity_matrix, y, bounds=(0, upper_bound), tol=tol)
    return result.x

def solve_all_layers(self):
    "adapted from f3ast"
    layer_list = []
    debug("Calculating proximity matrices")
    for layer_number in range(self.binary_array.shape[2]):
        prox_mat = self.calculate_prox_mat(layer_number)
        layer_solution = self.solve_layer(prox_mat, self.layer_height, tol=1e-3)
        layer_list.append(layer_solution)
    debug("Proximity matrices calculated")
    return layer_list

def calculate_dwells(self):
    debug("Calculating dwell points")
    dwell_list = []
    layers = self.solve_all_layers()
    for layer_number, layer in enumerate(layers):
        coordinates_in_layer, _ = self.get_points_and_resistances(layer_number)
        for index, point in enumerate(layer):
            x_coordinate = coordinates_in_layer[index][0]
            y_coordinate = coordinates_in_layer[index][1]
            dwell = point
            dwell_list.append([dwell, x_coordinate, y_coordinate]) 
    self.dwell_list = dwell_list
    print(f" Calculated {len(dwell_list)} dwell points")
    return dwell_list


def proximity_function(self, distances, resistance):
    """ from f3ast """
    return (
        self.growth_rate
        * np.exp(-self.k * resistance)
        * np.exp(-(distances**2) / (2 * self.sigma**2))
    )

def expand_points(stream_step_size, factor):
    return