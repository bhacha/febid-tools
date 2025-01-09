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
        self.dwell_list = []
        self.dwells = None
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
        


    def calculate_dwells(self, SEM_settings,):
        SEM = SEM_settings
        self.structure_size = self._structure.structure_size_nm[0]/SEM.pixel_size
        debug(f"Structure Size: {self.structure_size}", thresh='full')
        debug(f"Structure Size nm : {self._structure.structure_size_nm}")
        structure_xspan = [int(SEM.field_center[0] - self.structure_size/2), int(SEM.field_center[0] + self.structure_size/2)]
        structure_yspan = [int(SEM.field_center[1] - self.structure_size/2), int(SEM.field_center[1] + self.structure_size/2)]
        debug(f"X span: {structure_xspan[1] - structure_xspan[0]}")
        
        step_size_x = self.structure_size/self._structure.binary_array.shape[0]
        step_size_y = self.structure_size/self._structure.binary_array.shape[1]
        
        print(f"Step Size: {step_size_x}")
        
        point_xrange = range(structure_xspan[0], structure_xspan[1], int(step_size_x))
        point_yrange = range(structure_yspan[0], structure_yspan[1], int(step_size_y))

        debug("Calculating dwell points")
        fablist = []
        coord_list = []
        max_layer = self.dwells.shape[2]
        dwell_output = []
        for layer in range(max_layer):
            fab_indices = np.argwhere(self.dwells[:,:,layer])
            
            for [x, y] in fab_indices:
                dwell_time = self.dwells[x, y, layer]
                if dwell_time <= 3e10:
                    xpos = point_xrange[x]
                    ypos = point_yrange[y]
                    coordinate = [xpos, ypos]
                    coord_list.append(coordinate)
                    dwell_output.append(dwell_time)
                    fablist.append([dwell_time, xpos,ypos])
                else:
                    pass
            
        self.dwell_list = dwell_output
        self._structure.dwell_list = self.dwell_list
        self.fablist = fablist
        print(f" Calculated {len(dwell_output)} dwell points")
        return fablist
    
    

    def prepare_stream(self, SEM_settings, min_dwell=100, max_dwell=8e5):
        debug("Preparing Stream")

        if len(self.dwell_list) < 5:
            fablist = self.calculate_dwells(SEM_settings)
        else:
            fablist = self.fablist

        stream_list = []
        for spot in range(len(fablist)):
            coord_x = int(fablist[spot][1]) 
            coord_y = int(fablist[spot][2])
            #print(f"X: {coord_x}, Y: {coord_y}")
            dwell = int(fablist[spot][0])

            if (dwell>=min_dwell) and (dwell<= max_dwell):
                stream_line = (int(dwell), coord_x, coord_y)
                stream_list.append(stream_line)

        self.stream_list = stream_list       
        return stream_list


    def output_stream(self, filename, SEM_settings, sample_factor=1, min_dwell=500, max_dwell=5e5):
        stream_list = self.prepare_stream(SEM_settings, min_dwell=min_dwell, max_dwell=max_dwell)
        numpoints = len(stream_list)
        total_dwell = 0
        with open(filename+'.str', 'w') as f:
            f.write('s16\n1\n')
            f.write(str(numpoints)+'\n')
            for k in range(0,int(numpoints), sample_factor):
                xstring = str(stream_list[k][1])
                ystring = str(stream_list[k][2])
                dwell = stream_list[k][0]
                dwellstring = str(dwell)
                linestring = dwellstring+" "+xstring+" "+ystring+" "
                if k < int(numpoints)-1:
                    f.write(linestring + '\n')
                else:
                    f.write(linestring + " "+"0")
                total_dwell += dwell
        print(total_dwell/1e7)




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
        
        self.masked_array = self.convolution_mask(convolved_3d).real
        self.dwells = np.divide(self.layer_height, self.masked_array, where=(self.masked_array>1e-5))
        self.solved_array = convolved_3d

    def solve_all_layers(self):
        layers = []
        for k in range(self.dwells.shape[2]):
            layer = self.dwells[:,:,k].flatten().tolist()
            layers.append(layer)
        
        return layers


    def convolution_mask(self, solved_array):
        masked_array = np.where(self._structure.binary_array >.5, solved_array, 0.0)
        return masked_array

if __name__ == "__main__":

    pass