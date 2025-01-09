import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import trimesh as tm
# from . import structure_kernel as stk
import scipy.ndimage as snd
import skimage.transform as skt

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


def import_mesh(filepath, pitch):
    """
    Import mesh object. Convert to image/array using pitch
    
    Parameters
    -----------
    
    filepath : str
        path to file
        
    pitches : int
        array spacing used to reconstruct mesh
    
    """
    return Structure(filepath, pitch=pitch)


def import_image(filepath, threshold):
    """
    Import an image or numpy array. Use the threshold to convert to binary image.
    
    Parameters
    -----------
    
    filepath : str
        path to file
    
    threshold : float, int
        number to compare pixel/element values with
    
    """
    return Structure(filepath, threshold=threshold)
    



class Structure:

    def __init__(self,
                 filepath,
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
        self.threshold = kwargs.get("threshold", None)
        self.input_pitch = kwargs.get("pitch", None)

        ### file loading
        if filepath.endswith(".stl"):
            self.binary_array = self._import_stl(filepath, self.input_pitch, scale=1).astype(int)
        elif filepath.endswith(".npy"):
            self.binary_array = self._import_numpy(filepath, threshold=self.threshold, binarize=True).astype(int)
        
        
        ### List of Properties ###
        self.structure_size_nm = None  #size in nm
        self.labelled_regions = None   # discrete, connected regions in each layer
        self.centers_pix = None        # center locations of regions, in pixels
        self.areas_pix = None          # areas of regions, in pixels
        self.bounding_boxes_pix = None # bounds of regions, in pixels
        self.euler_numbers = None      # number of connected regions, minus number of holes. changes in euler numbers between layers indicates features have merged/separated
        self.structure_size_fab = None # Size of structure in fab pixels
        self.pix_size = None           # Pixel size of fab SEM
        self.pitch_nm = None           # Pitch of input array in nm units
        self.pitch_fab = None          # pitch of input array in fab pixels
        
        self.areas = None              # region areas in nm
        
        ### initialize resistance matrix
        self.total_resistances = np.zeros_like(self.binary_array, dtype=np.float64)
        self.resistance_calculated_flag = False 

        
        self.dwell_list = []
        
    def _import_stl(self, filepath, pitch, scale=False):
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
        print("Slicing structure...")
        if scale != False:
            stl_struct.apply_scale(scale)
        struc_mat = stl_struct.voxelized(pitch=pitch).fill()
        nummat = np.asanyarray(struc_mat.matrix)
        print("Slicing Completed!")
        return nummat

    def _import_numpy(self, filepath, binarize=True, **kwargs):
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

    def get_points_and_resistances(self, slice_number):
        if self.resistance_calculated_flag == False:
            self.calculate_resistance()
    
        ## calculate desired size per step. nm/array_pixel
        x_step_size = self.structure_size_nm[0] / self.binary_array.shape[0]
        y_step_size = self.structure_size_nm[1] / self.binary_array.shape[1]

        layer = self.binary_array[:,:,slice_number]
        res_layer = self.total_resistances[:,:,slice_number]

        points_in_slice = np.nonzero(layer.astype(np.float32))
        indices_in_slice = list(map(list, zip(*points_in_slice)))

        coords_in_slice = np.asarray(indices_in_slice)
        resistances_in_slice = res_layer[points_in_slice]
        
        coordinates = coords_in_slice * np.array([x_step_size, y_step_size])
        return coordinates, resistances_in_slice


    ### Sectioning/Regions and sizing
    
    def set_size(self, structure_size):
        """
        Define structure size in nm , to calculate pixel to size conversions.
        
        Parameters
        -----------
        structure_size : float or 3-list of floats
            If float, assumes cubic structure. List defines [x, y, z]
        
        """
        
        try:
            if len(structure_size) == 3:
                self.structure_size_nm = structure_size
            else:
                self.structure_size_nm = [structure_size, structure_size, structure_size]
        except TypeError:
            self.structure_size_nm = [structure_size, structure_size, structure_size]
            
        self.layer_height = self.structure_size_nm[2] / self.binary_array.shape[2] 

    def calculate_pitch(self, output_nm=False, **kwargs):        
        """
        Calculate the pitch size given the size of the input and desired size of the structure. With SEM settings, can also calculate the pitch in units of fab pixels (output_nm=False)
        
        Parameters
        ------------
        
        sem_settings : sem.SEM() object
        
        output_nm : bool
            default is False. If true, outputs the pitch in nanometers. False outputs pitch in fab pixel units.
        
        """
        
        sem_settings = kwargs.get("sem")

        if output_nm == False:
            
            if self.pix_size is None:
                self.pix_size = sem_settings.pixel_size
            
            self.structure_size_fab = self.structure_size_nm[0]/self.pix_size
            array_pitch = self.structure_size_fab / self.binary_array.shape[0]
            
        else:
            array_pitch = self.structure_size_nm[0]/ self.binary_array.shape[0]
        
        
        return array_pitch

    def section(self):
        if self.structure_size_nm == None:
            print("Error! use structure.set_size() first!")
            pass
        else:
            self.labelled_regions, self.centers_pix, self.areas_pix, self.bounding_boxes_pix, self.euler_numbers = self.calculate_regions_in_slices(return_bounds=True)
    
        
            self.pitch_nm = self.calculate_pitch(output_nm=True)
            self.centers = [center_pix * np.array(self.pitch_nm) for center_pix in self.centers_pix]
            self.areas = self.areas_pix * (self.pitch_nm**2)
    
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

    def get_coordinates(self):
        """
        Convert fab spots to x,y,z based on desired size
        """
        debug("Calculating Coordinates in Nanometers... ", thresh='output')
        ## calculate desired size per step. nm/array_pixel
        x_step_size = self.structure_size_nm[0] / self.binary_array.shape[0]
        y_step_size = self.structure_size_nm[1] / self.binary_array.shape[1] 
        z_step_size = self.layer_height
        
        array_fab_points = np.argwhere(self.binary_array)
        
        nm_fab_points = array_fab_points * np.array([x_step_size, y_step_size, z_step_size])
        
        debug("Coordinates in Nanometers Calculated!", thresh='output')
        
        return nm_fab_points
        
    def resize(self, output_size):
        """
        Take the structure's binary and resistance arrays and scale them up to match output_size.

        Parameters
        -----------

            output_size : {float, (3,) tuple of floats} 
                If float, resizes to a cube with edge lengths of output_size. Resizes to match tuple. 
        
        """
        unscaled_res = self.total_resistances
        unscaled_binary = self.binary_array
        
        try:
            if len(output_size) == 3:
                output_size = output_size
            else:
                output_size = [output_size, output_size, output_size]
        except TypeError:
            output_size = [output_size, output_size, output_size]
        
        scaled_res = skt.resize(unscaled_res, output_size, order=2, mode='constant')
        
        scaled_struct = skt.resize(unscaled_binary, output_size, order=1, mode='constant')
        new_thresh = (np.max(scaled_struct)-np.min(scaled_struct)) / 2
        scaled_binary = np.where(scaled_struct>new_thresh,1,0).astype(np.float32)
        
        self.binary_array = scaled_binary
        self.layer_height = self.structure_size_nm[2] / self.binary_array.shape[2]
        self.total_resistances = scaled_res
        

    ### Helper Functions
    def plot_slice(self, view_slice, type, colorbar=True):
        " Plotting wrapper function. WIP"
        
            # if isinstance()
            # else:
        array_slice = np.s_[:, :, view_slice]
        

        
        if type == 'structure':
            plot_array = self.binary_array[array_slice]
            cmap = 'binary'
        elif type == 'dwells':
            pass
        elif type == 'resistances':
            plot_array = self.total_resistances[array_slice]
            cmap='viridis'
        elif type == 'layer-res':
            plot_array = self._layer_resistances[array_slice]
            cmap='viridis'
        else:
            pass
    
        plt.figure()
        plt.imshow(plot_array, aspect='equal', origin='lower', cmap=cmap)
        if colorbar==True:
            plt.colorbar()            
        else:
            pass

    
    def update_binary_array(self,new_array):
        self._prev_binary_array = self.binary_array
        self.binary_array = new_array
        
    
    def update_resistance_array(self, new_array):
        self._prev_total_resistances = self.total_resistances
        self.total_resistances = new_array
    
    

if __name__ == "__main__":
    from sem import * 
    import time


    sem_settings = SEM(addressable_pixels=[65536, 56576],
                        screen_width=10.2e3,
                        )


    file = "branchtrio-SizedUp.stl"
    
    branched_trio = import_mesh(file, pitch=100)
    
    branched_trio.resize([500,500,500])
  
    plt.imshow(branched_trio.binary_array[:,:, 50])
    plt.colorbar()
