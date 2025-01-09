 ### Dwell Calculations      
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop
from scipy.spatial import KDTree
from importlib import import_module
from scipy.special import expit, logit

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


class Solver:
    def __init__(self,
                 structure,
                 calibration_parameters,
                 ):
        
        self._structure = structure
        self.k = calibration_parameters["k"]
        self.growth_rate = calibration_parameters["GR0"]
        self.layer_height = structure.layer_height
        
        pass

    
    def pre_checks(self):
        """
        Check if the structure has all its resistances calculated. Maybe other checks later.
        """
        if self._structure.resistance_calculated_flag == True:
            pass
        else:
            ##eventually this should just do the calculation and continue.
            print("Error: Please calculate resistances of the structure first")
            return
        


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



class ProximityMethod(Solver):

    def __init__(self):
        super().__init__()

        pass
 
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


    def proximity_function(self, distances, resistance):
        """ from f3ast """
        return (
            self.growth_rate
            * np.exp(-self.k * resistance)
            * np.exp(-(distances**2) / (2 * self.sigma**2))
        )


    
class ConvolutionMethod(Solver):

    def __init__(self,
                 structure,
                 sigma,
                 calibration_parameters,
                 kernel_size=15,
                 filter='gaussian',
                 **kwargs):
        super().__init__(structure=structure, calibration_parameters=calibration_parameters)

        self.kernel = import_module(".kernel", "f4mine")
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.pitch = kwargs.get("pitch", self._structure.calculate_pitch(output_nm=True))
        self.Filter = None
    
        if filter == 'gaussian':
            self.filter_type="GAUSSIAN"
            self.gaussian()
        elif filter== 'inverse_gaussian':
            self.filter_type="INVGAUSS"
            self.inverse_gaussian()
        else:
            print("Sorry, you can't customize a filter yet!")
               
    def gaussian(self):
        layer = np.zeros_like(self._structure.total_resistances[:,:,0])
        self.Filter = self.kernel.Gauss(self.pitch, self.sigma, layer, self.kernel_size)
        return self.Filter
    
    def inverse_gaussian(self):
        layer = np.zeros_like(self._structure.total_resistances[:,:,0])
        self.Filter = self.kernel.InverseGauss(self.pitch, self.sigma, layer, self.kernel_size)
        return self.Filter
    
    def convolution_solve(self):
        res_3d = expit(-self.k * self._structure.total_resistances)
        
        convolved_3d = np.zeros_like(res_3d).astype(np.complex128)
        
        for layer_number in range(res_3d.shape[2]):
            layer = res_3d[:,:,layer_number]
            convolved_3d[:,:,layer_number] = self.kernel.apply_filter(layer, self.Filter)
        
        self.masked_array = self.convolution_mask(convolved_3d)
        self.dwells = np.divide(self.layer_height, self.masked_array, where=(self.masked_array>1e-5))
        self.solved_array = convolved_3d


    def convolution_mask(self, solved_array):
        masked_array = np.where(self._structure.binary_array >.5, solved_array, 0.0)
        return masked_array

if __name__ == "__main__":

    pass