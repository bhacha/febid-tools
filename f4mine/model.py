import numpy as np


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


class Model():
    
    def __init__(self, structure, get_parameters=True, **kwargs):
        
        self._struct = structure
        self.binary_array = structure.binary_array
        if get_parameters:
            self.get_layer_parameters()
            
            
    def get_layer_parameters(self):
        """Do whatever calculations are needed"""
        pass

    
class ArrayModel(Model):
    
    def __init__(
        self,
        structure,
        sigma,
        single_pixel_width: float = 50,
        **kwargs
    ):
        super().__init__(structure, **kwargs)
        self.rho = kwargs.get("rho", 1)
        self.sigma = sigma
        self.single_pixel_width = single_pixel_width
        
        self.total_resistances = np.zeros_like(self.binary_array, dtype=np.float64)
        self.resistance_calculated_flag = False 
    
        if self._struct.areas is None:
            self._struct.section()
    
    def calculate_resistance(self):
        
        """
        calculate the total structure resistance. Begin with calculating the resistance of each slice, then determine whethere these add in series or parallel based on the number of regions in the adjacent slices. If the number of regions changes in adjacent layers, that implies that something has merged or separated and thus the resistance should be calculated in parallel.
        """
        
        """
        TO-DO
        
        Incorporate the serial summation inside of the parallel one. The parallel addition should only take place at the interfaces between regions with different euler numbers. It should NOT use the series-calculated resistances above this point, but rather begin calculating them anew, factoring in the parallel-resistance regions below.
        
        I imagine treating the first region simply with the series calculation, then incorporating the simple adding of the layer_resistance in the parallel function for layers != 0 (the layers not at the interface). 
        
        """
        
        ### split arrays into euler-equivalent stacks, then find the regions where the euler number changes. 
        _, unique_region_indices = self.euler_resistance_subregions()
        
        # calculate isolated layer resistances
        layer_resistances = self.calculate_layer_resistance()
        debug("Layer resistances calculated")

        for layer_number in range(layer_resistances.shape[2]):
            layer = layer_resistances[:,:,layer_number]
            if layer_number == 0:
                previous_layer = np.zeros_like(layer)
            else:
                previous_layer = self.total_resistances[:,:, layer_number-1]
            
            
            np.add(layer, previous_layer, where=np.where(layer != 0, True, False), out = self.total_resistances[:,:,layer_number])
            # #we actually don't need to do parallel stuff for the first region with a different euler number, since it's the first part where branching happens it won't get the advantage of the different conduction paths. 
            # if layer_number not in unique_region_indices[1:]:
            #     np.add(layer, previous_layer, where=np.where(layer != 0, True, False), out = self.total_resistances[:,:,layer_number])
            # else:
            #     self.calculate_parallel_resistance(layer, layer_number)
            
        debug("Total resistances calculated")
        self.resistance_calculated_flag = True
        self._struct.resistance_calculated_flag = True
        self._struct.update_resistance_array(self.total_resistances)
        
        return
    
    def euler_resistance_subregions(self):
        """
        split the array into regions with different euler numbers. In each region, the resistance should add in series. The regions then add in parallel with the ones below it.
        """
        layer_resistances = self.calculate_layer_resistance()
        euler_arr = np.asarray(self._struct.euler_numbers)
        unique_euler_layers = np.where(euler_arr[:-1] != euler_arr[1:])[0]
        split_arr = np.split(layer_resistances, unique_euler_layers, axis=2)
        return split_arr, unique_euler_layers
    
    def calculate_layer_resistance(self):
        """
        Calculate the resistance of each point, assuming they obey the equations for resistance of a wire. 

        This will be:
        R = rho * (Length/Area)
        R = integral ( rho/Area) dL

        The rho will be a fudge factor, but the area is calculated already and the length will come from the inter-slice proximity.

        Basically, for each region, get the area and figure out if any regions are in the stack below it. If so, add the resistance at that point (going from slice to slice, the resistance is proportional to the length so it is simply additive. No reciprocal needed). connectivity and parallel resistance should come into this at some point. But alas.
        """
        
        layer_resistances = np.zeros_like(self.binary_array, dtype=np.float64)

        #calculate resistances for the single layer. Assume layer_height is constant.
        resistance_constant = self.rho*self._struct.layer_height
        
        ## normalize for single pixel line being 50 nm
        normalized_areas = self._struct.areas / (50**2)
        
        for layer_number in range((self.binary_array.shape[-1])):
            ### create an array where coordinates are the resistances in the layer. np divide is to do rho*dL/Area
            single_layer_resistance = np.divide(resistance_constant, 
                                        normalized_areas[:,:, layer_number],
                                        out=np.zeros_like(normalized_areas[:,:,layer_number], dtype=np.float64), where=normalized_areas[:,:,layer_number]>0)

            
            layer_resistances[:,:,layer_number] = single_layer_resistance

        self._layer_resistances = layer_resistances
        return layer_resistances
    
    def calculate_series_resistance(self, resistances_euler_subregion):
        """
        calculate the thermal resistance, assuming the regions add in series. This is true if the topology does not change from layer to layer. If the number of regions changes, that indicates multiple "branches" have connected and then the parallel calculation should be done. 
        
        Still, the parallel calculation will involve just reciprocally adding the resistances calculated here
        """
        
        layer_resistances_in_region = resistances_euler_subregion
        summed_resistances_in_region = np.zeros_like(layer_resistances_in_region)
        
        ### Serial adding in each region
        for layer in range(layer_resistances_in_region.shape[2]):
            ## iterate through the layers in the region
            resistance = layer_resistances_in_region[:,:,layer]
            if layer == 0:
                previous_resistance = np.zeros_like(layer_resistances_in_region[:,:,layer])
            else:
                previous_resistance = resistance
                
            resistance += previous_resistance    
            
            summed_resistances_in_region[:,:, layer] = resistance
                        
        return summed_resistances_in_region
    
    def calculate_parallel_resistance(self, resistance_layer, layer_index):
        """
        calculate the thermal resistance, assuming the regions add in parallel. This will involve reciprocally summing the resistances of the regions that are connected in series.
        """
        number_of_paths = np.abs(self._struct.euler_numbers)

        previous_paths = number_of_paths[layer_index-1] # get previous Euler number
        current_paths = number_of_paths[layer_index] # current Euler number
        path_difference = np.abs(current_paths - previous_paths) ### get the difference in paths
        debug(f"The path difference is {path_difference}")

        resistance = resistance_layer
        ## add the resistances dividing by the change in Euler number to approximate the parallel thermal resistance
        self.total_resistances[:,:, layer_index] = resistance + np.divide(self.total_resistances[:,:,layer_index-1], path_difference, out=np.zeros_like(resistance, dtype=np.float64),
        where=self.total_resistances[:,:,layer_index-1]!=0)










if __name__ == "__main__":
    import structure
    import sem
    import matplotlib.pyplot as plt

    sem_settings = sem.SEM(addressable_pixels=[65536, 56576],
                        screen_width=10.2e3,
                        )


    file = "branchtrio-SizedUp.stl"
    
    branched_trio = structure.import_mesh(file, pitch=100)
    
    size_x = 1000 #nm
    size_y = 1000 #nm
    size_z = 1000 #nm
    branched_trio.set_size([size_x, size_y, size_z])

    sigma = 4.4 # in nm, dwell size

    mod = ArrayModel(branched_trio, sigma=sigma, single_pixel_width=50, rho=1)
    

    resistances = mod.calculate_resistance()
    layres = mod.calculate_layer_resistance()
    
    print(resistances.shape)
    plt.figure()    
    plt.imshow(resistances[:,:,35], cmap='viridis')
    plt.colorbar()
    
    plt.figure()    
    plt.imshow(resistances[:,:,40], cmap='viridis')
    plt.colorbar()
    
    plt.figure()    
    plt.imshow(resistances[:,:,45], cmap='viridis')
    plt.colorbar()
    
