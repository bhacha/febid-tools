import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import skimage.graph as sg
import scipy.signal as sps
import trimesh as tm
import trimesh.voxel as tv

np.set_printoptions(threshold=np.inf)


class Structure:

    def __init__(self,
                 filepath,
                 SEM_settings,
                 structure_size_nm,
                 **kwargs):
        
        """
        
        Parameters
        -------------
        
        filepath : str
            numpy array or stl
            
        SEM_settings : SEM
            SEM object to get settings from
            
        structure_size_nm : list [X, Y, Z] or float
            if list/array, X Y and Z are defined directly. If float, then the structure is assumed to be a cube.
        """
                
        self.SEM_settings = SEM_settings
        
        self.threshold = kwargs.get("threshold", None)
        self.pitch = kwargs.get("pitch", None)
        
        self.growth_rate = .025
        self.k = 2.5
        self.sigma = 4
        self.rho = 20
        

        ### file loading
        if filepath.endswith(".stl"):
            self.binary_array = self.import_stl(filepath, self.pitch)
        elif filepath.endswith(".npy"):
            self.binary_array = self.import_numpy(filepath, threshold=self.threshold, binarize=True)
            

        ### sizing
        try:
            if len(structure_size_nm) == 3:
                self.structure_size_nm = structure_size_nm
            else:
                self.structure_size_nm = [structure_size_nm, structure_size_nm, structure_size_nm]
        except TypeError:
            self.structure_size_nm = [structure_size_nm, structure_size_nm, structure_size_nm]

        ### generate list of points, and define the layer_height
        self.get_coordinates()

        
        ### sectioning
        self.labelled_regions, self.centers, self.areas, self.bounding_boxes, self.euler_numbers = self.calculate_regions_in_slices(return_bounds=True)
        
        ### initialize resistance matrix
        self.total_resistances = np.zeros_like(self.binary_array, dtype=np.float64) 
        

    @property
    def shape(self):
        shape=self.binary_array.shape
        return shape        

    def import_stl(self, filepath, pitch):
        """
        import stl using trimesh, then convert to a filled numpy array
        
        Parameters
        -----------
        filepath : str
            filepath to stl object
        
        pitch : float
            resulting voxel sizes
        
        Returns
        --------
        nummat : numpy array
            boolean matrix with areas inside structure being True and outside being False. 3 dimensions, size being set by the pitch. (X,Y,Z)        
        
        """
        stl_struct = tm.load_mesh(filepath)
        struc_mat = stl_struct.voxelized(pitch=pitch).fill()
        nummat = np.asanyarray(struc_mat.matrix)
        return nummat

    def import_numpy(self, filepath, binarize=True, **kwargs):
        """
        import numpy array, then threshold (optional) to get binary matrix
        
        Parameters
        -----------
        filepath : str
            filepath to saved numpy array
               
        binarize : boolean
            if True, converts to binary matrix. This is the default, since that needs to happen for the rest of the processing.
            
        kwargs:
            threshold: float
                Value used to convert to boolean/binary matrix. If the values are above the threshold, this sets them to True (1).
        
        Returns
        --------
        array : numpy array
            boolean matrix with areas inside structure being True and outside being False. 3 dimensions, with same size as input array       
        
        """
        
        threshold = kwargs.get("threshold", 1)
        
        epsilon_array = np.load(filepath)
        if binarize == True:
            array = np.where(epsilon_array>threshold, False, True)
        else:
            array = epsilon_array
        return array

    def get_points(self):
 
        """
        This converts the array into points with (x,y,z) values in index units. 

        """
        z_index = 0
        points = []
        for layer in self.nonzero_indices:
            ### Go through each layer and create tuples for the xy indices
            xy_indices = zip(layer[0], layer[1])
            for n in xy_indices:
                ### Export all the points (in array units), with their z index
                points.append((n[0], n[1], z_index))

            z_index +=1

        #unfortunately the tuple is in order (Z, X, Y) for some reason????
        return points
    
    def get_coordinates(self):
        """
        Convert to x,y,z based on desired size and layer_height
        """
        ## calculate desired size per step. nm/array_pixel
        x_step_size = self.structure_size_nm[0] / self.binary_array.shape[0]
        y_step_size = self.structure_size_nm[1] / self.binary_array.shape[1] 
        z_step_size = self.structure_size_nm[2] / self.binary_array.shape[2] 
        
        self.layer_height = z_step_size    
        
        array_fab_points = np.argwhere(self.binary_array)
        
        nm_fab_points = array_fab_points * np.array([x_step_size, y_step_size, z_step_size])
        
        return nm_fab_points

    ### Sectioning/Regions and sizing
    def calculate_regions_in_slices(self, return_bounds=False):
        """
        Returns the array with the same shape as binary_array but with nonzero elements for regions that are connected in the layer.
        
        Parameters
        -----------
            binary_array : np.array()
                binarized 3D-array
            
            return_bounds : bool
                return the bounding boxes or not (default False)

        Returns
        --------
            full_labels : 
                array of all the points, with an 
            nonzero_indices :
                positions of the non-zero elements, in index units
            area_labels :
                areas of each of the labelled regions, index^2 units 

        """

        nonzero_indices = []
        centroid_list = []
        bounds_list = []
        euler_numbers = []
        ## initialize numpy arrays to store our things.
        area_array = np.zeros_like(self.binary_array, dtype=np.float64)
        full_labels = np.zeros_like(self.binary_array, dtype=np.float64)

        for i in range(self.binary_array.shape[-1]):
            """
            Going slice by slice, label connected regions and calculate their area

            A single layer of the resulting area array has the same number of entries as there are labelled regions. The caveat is that labels start at i=1 and the list of areas starts with l=0. So to get the area of the region labelled 2, you would need to query area[1]. this poses problems with indexing the area array *using* the labels array. 

            The fix is to append a 0 to the beginning of the list. This should never be accessed, but if it is we will know because area of 0 isn't possible (it wouldn't be labelled)

            """
            slice = self.binary_array[:,:,i]
            
            slice_labels = sm.label(slice)
            regions = sm.regionprops(slice_labels)
            
            euler_numbers.append(sm.euler_number(slice, connectivity=2))
            
            """
            Create a list of the areas of the labelled regions in each slice, then make a list of those lists to get a list for all the slices.
            """

            slice_areas = []
            centroids = []
            bounds = []
            
            for n in regions:
                slice_areas.append(n.area)
                centroids.append(n.centroid)
                bounds.append([n.bbox])
            

            # put a padding zero at the front, as mentioned above
            slice_areas.insert(0, 0)
            centroids.insert(0, [0,0])
            bounds.insert(0, [(0,0,0,0)])

            slice_areas = np.asarray(slice_areas)
            centroids = np.round(np.asarray(centroids), 0)
            bounds = np.asarray(bounds)
            ### Append to the whole-structure lists
            full_labels[:,:,i] = np.asarray(slice_labels)
        
            centroid_list.append(centroids)
            bounds_list.append(bounds)
            ### Use numpy indexing to replace the array of labels with an array of the areas of those labelled regions
            area_array[:,:,i] = slice_areas[slice_labels]

            
            """
            I want the indices in the original matrix that are labelled. This corresponds to the indices of the elements that are connected in the original matrix. The output here is essentially the coordinates for fabrication. This is probably not going to be useful since we want to track other properties too (connectivity, area, etc.)
            
            It's not currently being returned since I'm unsure of its value.
            """
            nonzero_indices.append(np.nonzero(slice_labels))


            """
            The optional bounds output is a list of arrays, each array has xmin, ymin, xmax, ymax. 
            each one can be reshaped to 
            
            bound_array = bounding_boxes[layer_number].reshape(-1, 4)
            
            or possibly
            
            bound_array = bounding_boxes.reshape(-1, 4, binary_array.shape[-1])
        
        
            so that each row is one bounding box with [xmin, ymin, xmax, ymax]
                        
            """
            
        if return_bounds == False:
            return full_labels, centroid_list, area_array, euler_numbers
        elif return_bounds == True:
            return full_labels, centroid_list, area_array, bounds_list, euler_numbers


    ### Sizing stuff is TBD, not sure which values are best to have.
    def calculate_index_size(self, output_nm = False):
        """
        returns the actual size (in pixels) of each index unit in an array, based on the desired fabrication size


        Parameters
        -----------
            binary_array :
                array
                
            structure_size_nm : 
                desired structure size (scale)
                
            output_nm : bool
                if True, outputs the pitch in nanometers. Default is false, so output is in SEM pixel units (used by the stream file)

        Returns
        -----------
            array_pitch : 
                size of each index step in an array, in SEM pixel units. Multiply index positions by this number to get pixel coordinates
        
        """
        SEM = self.SEM_settings
        
        ##get pixel size
        self.pix_size = SEM.pixel_size

        ## convert nm structure size to pixels
        self.structure_size = self.structure_size_nm[0]/self.pix_size

        ### Right now, assume the pixels are square. This is probably something that can be fixed later though.
        array_pitch = self.structure_size / np.max(self.binary_array.shape[:1])
        
        if output_nm == True:
            array_pitch = self.structure_size_nm[0]/ np.max(self.binary_array.shape[:1])
        
        return array_pitch

    def calculate_structure_size(self, units='pixels'):
        SEM = self.SEM_settings
        ##get pixel size, in nm
        self.pix_size = SEM.pixel_size
        
        if units == 'pixels':
            self.calculate_structure_size_pix()
        elif units == 'nm':
            self.calculate_structure_size_nm()
        
    def calculate_structure_size_nm(self):
        pass

    def calculate_structure_size_pix(self):       
        ## convert nm structure size to pixels
        structure_size = self.structure_size_nm[0]/self.pix_size
        
        structure_pix_xspan = [int(SEM.field_center[0] - structure_size/2), int(SEM.field_center[0] + structure_size/2)]
        structure_pix_yspan = [int(SEM.field_center[1] - structure_size/2), int(SEM.field_center[1] + structure_size/2)]

        '''I use range in the next step because linspace gives floats and I don't want to worry about rounding methods causing  distortion. The tradeoff is that I need to calculate the step size instead of the number of steps. This will get rounded, but then apply the same step size to everything. I think this would reduce distortion since values won't round away from one another.
        '''
        step_size_x = structure_size/self.shape[0]
        step_size_y = structure_size/self.shape[1]
        
        point_xrange = range(structure_pix_xspan[0], structure_pix_xspan[1], int(step_size_x))
        point_yrange = range(structure_pix_yspan[0], structure_pix_yspan[1], int(step_size_y))
        
        
        

    ### Resistance calculations
    
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
        
        layer_resistances = self.calculate_layer_resistance()

        for layer_number in range(layer_resistances.shape[2]):
            layer = layer_resistances[:,:,layer_number]
            if layer_number == 0:
                previous_layer = np.zeros_like(layer)
            else:
                previous_layer = self.total_resistances[:,:, layer_number-1]

            #we actually don't need to do parallel stuff for the first region with a different euler number, since it's the first part where branching happens so it won't get the advantage of the different conduction regions. 
            if layer_number not in unique_region_indices[1:]:
                self.total_resistances[:,:,layer_number] = layer + (previous_layer)
            else:
                self.calculate_parallel_resistance(layer, layer_number)
            
        
        return self.total_resistances
        
    def euler_resistance_subregions(self):
        """
        split the array into regions with different euler numbers. In each region, the resistance should add in series. The regions then add in parallel with the ones below it.
        """
        layer_resistances = self.calculate_layer_resistance()
        euler_arr = np.asarray(self.euler_numbers)
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
        rho = self.rho
        resistance_constant = rho*self.layer_height
        print(resistance_constant)
        for layer_number in range((self.binary_array.shape[-1])):

            ### create an array where coordinates are the resistances in the layer. np divide is to do rho*dL/Area
            single_layer_resistance = np.divide(resistance_constant, 
                                        self.areas[:,:,layer_number],
                                        out=np.zeros_like(self.areas[:,:,layer_number], dtype=np.float64), where=self.areas[:,:,layer_number]!=0)

            
            layer_resistances[:,:,layer_number] = single_layer_resistance

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
        number_of_paths = np.abs(self.euler_numbers)

        previous_paths = number_of_paths[layer_index-1] # get previous Euler number
        current_paths = number_of_paths[layer_index] # current Euler number
        path_difference = np.abs(current_paths - previous_paths)+1 ### get the difference in paths


        resistance = resistance_layer
        ## add the resistances dividing by the change in Euler number to approximate the parallel thermal resistance
        self.total_resistances[:,:, layer_index] = resistance + np.divide(path_difference,self.total_resistances[:,:,layer_index-1], out=np.zeros_like(resistance, dtype=np.float64),
        where=self.total_resistances[:,:,layer_index-1]!=0)


    ### Sizing calculations
   
    
    ### Dwell Calculations
    def calculate_dwells(self):
        SEM = self.SEM_settings
        
        
        self.index_size = self.calculate_index_size(self.structure_size_nm[0])
        
        ## generate range of addressed pixels
        structure_xspan = [int(SEM.field_center[0] - self.structure_size/2), int(SEM.field_center[0] + self.structure_size/2)]
        structure_yspan = [int(SEM.field_center[1] - self.structure_size/2), int(SEM.field_center[1] + self.structure_size/2)]

        step_size_x = self.structure_size/self.binary_array.shape[0]
        step_size_y = self.structure_size/self.binary_array.shape[1]
        
        point_xrange = range(structure_xspan[0], structure_xspan[1], int(step_size_x))
        point_yrange = range(structure_yspan[0], structure_yspan[1], int(step_size_y))

        dwells = []

        
        resistance_array = self.total_resistances
        # print(np.unique(resistance_array))

        # create a new array, which uses the fabrication binary_array as a mask for the resistances
        fabspot_array = np.where(self.binary_array != 0, resistance_array, 0)
        
        ## for each layer, get indices of the fabrication spots
        fablist = []
        coord_list = []
        max_layer = fabspot_array.shape[2]
        for layer in range(fabspot_array.shape[2]):
            fab_indices = np.argwhere(fabspot_array[:,:,layer])
            fablist.append(fab_indices)
            
            for [x, y] in fab_indices:
                dwell_time = self.dwell_model(resistance_array[x,y,layer], layer, max_layer)
                if dwell_time <= 3e5:
                    xpos = point_xrange[x]
                    ypos = point_yrange[y]
                    coordinate = [xpos, ypos]
                    coord_list.append(coordinate)
                    dwells.append(dwell_time)
                else:
                    pass
        
        coord_arr = np.asarray(coord_list)           
        return coord_arr, dwells
            
    def dwell_model(self, resistance, layer, max_layer):
        
        ### There's some proximity matrix term that accounts for the gaussian spot, but I don't know how to implement that yet
        # gauss_term = np.exp(-(r**2)/(2*self.sigma**2))
        gauss_term = .5
        growth_term = self.growth_rate*np.exp((-self.k*resistance)) 
        total_height = max_layer/(self.layer_height*(layer+1))
    
        dwell_time = total_height/(growth_term*gauss_term)
        
        return dwell_time

    def output_stream(self, filename, sample_factor=1):
        total_dwell = 0
        coord_arr, dwells = self.calculate_dwells()
        numpoints = str(int(coord_arr.shape[0]/sample_factor))
        with open(filename+'.str', 'w') as f:
            f.write('s16\n1\n')
            f.write(numpoints+'\n')
            for k in range(0,int(numpoints), sample_factor):
                xstring = str(coord_arr[k, 0])
                ystring = str(coord_arr[k, 1])
                dwell = round(dwells[k])
                dwellstring = str(dwell)
                linestring = dwellstring+" "+xstring+" "+ystring+" "
                if k < int(numpoints)-1:
                    f.write(linestring + '\n')
                else:
                    f.write(linestring + " "+"0")
                total_dwell += dwell
        print(total_dwell/1e7)
        
    ### Helper Functions
    
    def plot_slice(self, slice, colorbar=True):
        plt.figure()
        plt.imshow(self.binary_array[:,:,slice], aspect='equal', origin='lower')
        if colorbar==True:
            plt.colorbar()            
        else:
            pass


if __name__ == "__main__":
    from sem import * 
    import time

    sem_settings = SEM(addressable_pixels=[65536, 56576],
                        screen_width=10.2e3,
                        )


    file = "branchtrio-SizedUp.stl"



    structure = Structure(file, sem_settings, pitch=75, structure_size_nm=[1000, 1000, 1000])
    structure.calculate_resistance()

    structure.output_stream('testfile.str', sample_factor=2)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(fabpoints[:, 0], fabpoints[:,1], fabpoints[:,2])
    # ax.invert_xaxis()
    
    #%%
    # plt.imshow(structure.total_resistances[40,:,:50], origin='lower', vmax=1)
    # plt.colorbar()


    # plt.figure()
    # plt.imshow(structure.total_resistances[:,:,100], origin='lower')
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(structure.total_resistances[:,:,30], origin='lower')
    # plt.colorbar()
   
 
    
    #    file = "Test3DEpsilonArray.npy"
    # array = import_numpy(file)
    # threshold = 9
    # binary_array = binarize(array, threshold=threshold)

    # layer_height = 100 #nm
    # growth_rate = 200 #nm/s
    # k_term = 1.1 #nm

    # dwell_list = []
    # for n in np.linspace(0, 2, 30):
    #         dwell = dwell_model(layer_height, growth_rate, k_term, sigma=4, resistance=n, r=4)
    #         dwell_list.append(dwell)
    #         plt.scatter(n, dwell)

    
    # plt.xlabel("Resistance")
    # plt.ylabel("Dwell Time")
    # plt.show()
    # bins = 900
    # dwell_sum = np.sort(dwell_list)
    # dwell_fun = np.array(range(bins))/float(bins)

    # plt.figure()
    # plt.plot(dwell_sum, dwell_fun)
    # plt.show()

    ## Add a block to help orient things
    # binary_array[30:90, 50:90, :20] = True

    
    # labels, indices, areas = calculate_regions_in_slices(binary_array[:,:,:].astype(int))

    # test = calculate_resistance(binary_array[:,:,:40])

    # plot_slice = 30

    # print(areas[:,:,plot_slice])

    # plt.figure()
    # plt.imshow(areas[:,:,plot_slice], cmap='viridis')
    # plt.title("Areas")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(labels[:,:,plot_slice], cmap='Paired')
    # plt.title("Labelled Regions")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(binary_array[:,:,plot_slice], cmap='Paired')
    # plt.title("Raw Binary Array")
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(test[:,:,plot_slice], cmap='viridis')
    # plt.title("resistance array")
    # plt.colorbar()


    # plt.show()
# %%
