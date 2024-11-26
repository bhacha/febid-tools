import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import skimage.graph as sg
import scipy.ndimage as snd
np.set_printoptions(threshold=np.inf)

class SEM:
    def __init__(
            self,
            addressable_pixels,
            screen_width
    ):
        self.addressable_pixels = addressable_pixels
        self.screen_width = screen_width
        self.field_center = [addressable_pixels[0]/2, addressable_pixels[1]/2]
        self.pixel_size = screen_width/addressable_pixels[0] 

def import_numpy(filepath):
    epsilon_array = np.load(filepath)
    return epsilon_array

def get_points(binary_array):
    full_labels, nonzero_indices, area_labels = calculate_regions_in_slices(binary_array)

    """
    This converts the array into points with (x,y,z) values in index units. 

    """
    z_index = 0
    points = []
    for layer in nonzero_indices:
        ### Go through each layer and create tuples for the xy indices
        xy_indices = zip(layer[0], layer[1])
        for n in xy_indices:
            ### Export all the points (in array units), with their z index
            points.append((n[0], n[1], z_index))

        z_index +=1

    #unfortunately the tuple is in order (Z, X, Y) for some reason????
    return points

class Structure:
    
    def __init__(self):
        pass

def binarize(array, threshold):
    binary_array = np.where(array>threshold, False, True)
    return binary_array



"""
f3ast and my other functions use a layer-by-layer approach. I can't help but feel like a kernel convolve/correlate approach is better. My thinking is that an x-y proximity matrix can be represented via weights in one row, while resistances can be represented by weights in columns. A wide, interconnected structure should have more adjacent pixels/voxels and thus would have larger sums. Of course, the resulting matrix would have a lot of meaningless weights in areas that are not supposed to be fabricated, but the array could be used as a lookup table of sorts. The indices of the nonzero pixels in binary_array could be used to query the same points in the convolved matrix and return the resistance or proximity weight. 
"""

def calculate_neighbors_kernel(binary_array, pitch, sigma):

    """
    Param
    -------
        pitch : float
            spacing between pixels in nm units

        sigma : float
            beam waist thing in nm
    
    """
    """The following sets the bottom layer to be the main contribution to the kernel. This means contributions from points above don't affect the convolved output. The idea is that each point depends only on the points below it. The center point is set to 1, so that blank areas above 
    This should give a pretty decent local approximation, but does not give the full branch_lengths.
    """
    

    neighbor_kernel = np.ones((3,3,3))
    neighbor_kernel[:, :, :2] = 0
    neighbor_kernel[:,:,2] = 5
    neighbor_kernel[1,1,1] = 1
    neighbor_arr = snd.correlate(binary_array.astype(int), neighbor_kernel)
    

    return neighbor_arr

















####
"""
The below functions use layer-by-layer calculations and loops to come up with a resistance based on areas and heights.
"""
def calculate_regions_in_slices(binary_array, return_bounds=False):
    """
    Returns the array with the same shape as binary_array but with nonzero elements for regions that are connected in the layer.
    
    Parameters
    -----------
        binary_array : np.array()
            binarized 3D-array

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
    ## initialize numpy arrays to store our things.
    area_array = np.zeros_like(binary_array, dtype=np.float64)
    full_labels = np.zeros_like(binary_array, dtype=np.float64)

    for i in range(binary_array.shape[-1]):
        """
        Going slice by slice, label connected regions and calculate their area

        A single layer of the resulting area array has the same number of entries as there are labelled regions. The caveat is that labels start at i=1 and the list of areas starts with l=0. So to get the area of the region labelled 2, you would need to query area[1]. this poses problems with indexing the area array *using* the labels array. 

        The fix is to append a 0 to the beginning of the list. This should never be accessed, but if it is we will know because area of 0 isn't possible (it wouldn't be labelled)

        """
        slice = binary_array[:,:,i]
        
        slice_labels = sm.label(slice)
        regions = sm.regionprops(slice_labels)
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
        return full_labels, centroid_list, area_array
    elif return_bounds == True:
        return full_labels, centroid_list, area_array, bounds_list

def calculate_resistance(binary_array, layer_height = 1):
    """
    Calculate the resistance of each point, assuming they obey the equations for resistance of a wire. 

    This will be:
      R = rho * (Length/Area)
      R = integral ( rho/Area) dL

    The rho will be a fudge factor, but the area is calculated already and the length will come from the inter-slice proximity.

    Basically, for each region, get the area and figure out if any regions are in the stack below it. If so, add the resistance at that point (going from slice to slice, the resistance is proportional to the length so it is simply additive. No reciprocal needed)

    

    connectivity and parallel resistance should come into this at some point. But alas.

    """
    labelled_regions, centers, areas, bounding_boxes = calculate_regions_in_slices(binary_array, return_bounds=True)


    resistances = np.zeros_like(binary_array, dtype=np.float64)

    for layer_index in range((binary_array.shape[-1])):
        # redefine in case we want to shift this later by +1 or -1
        layer_number = layer_index

        #calculate resistances for the single layer. Assume layer_height is constant.
        rho = 1
        resistance_constant = rho*layer_height

        ### create an array where coordinates are the resistances in the layer. np divide is to do rho*dL/Area
        single_layer_resistance = np.divide(resistance_constant, 
                                     areas[:,:,layer_number],
                                       out=np.zeros_like(areas[:,:,layer_number], dtype=np.float64), where=areas[:,:,layer_number]!=0)

        if layer_number == 0:
            resistances[:,:,layer_number] = 0
            previous_layer_resistance = single_layer_resistance

        else:
            resistances[:,:,layer_number] = single_layer_resistance + previous_layer_resistance

            previous_layer_resistance += single_layer_resistance

    return resistances

def calculate__index_size(binary_array, SEM, structure_size_nm):
    """
    returns the actual size (in pixels) of each index unit in an array, based on the desired fabrication size


    Parameters
    -----------
        binary_array :
            array

        SEM : SEM() object
            object of SEM class, containing the settings 

        structure_size_nm : 
            desired structure size (scale)

    Returns
    -----------
        array_pitch : 
            size of each index step in an array, in pixel units. Multiply index positions by this number to get pixel coordinates
    
    
    """
    
    ##get pixel size
    pix_size = SEM.pixel_size

    ## convert nm structure size to pixels
    structure_size = structure_size_nm/pix_size

    ## generate range of addressed pixels
    structure_xspan = [int(SEM.field_center[0] - structure_size/2), int(SEM.field_center[0] + structure_size/2)]
    structure_yspan = [int(SEM.field_center[1] - structure_size/2), int(SEM.field_center[1] + structure_size/2)]

    ### Right now, assume the pixels are square. This is probably something that can be fixed later though.
    array_pitch = structure_size / np.max(binary_array.shape[:1])

def calculate_dwells(binary_array, layer_height, growth_rate, k, sigma):

    resistance_array = calculate_resistance(binary_array)
    # print(np.unique(resistance_array))


    # create a new array, which uses the fabrication binary_array as a mask for the resistances
    fabspot_array = np.where(binary_array != 0, resistance_array, 0)

    return fabspot_array
    

def dwell_model(layer_height, growth_rate, k, sigma, resistance, r):
    
    ### There's some proximity matrix term that accounts for the gaussian spot, but I don't know how to implement that yet

    growth_term = growth_rate*np.exp((-k*resistance)) 
    gauss_term = np.exp(-(r**2)/(2*sigma**2))

    dwell_time = layer_height/(growth_term*gauss_term)
    
    return dwell_time



if __name__ == "__main__":
    file = "Test3DEpsilonArray.npy"
    array = import_numpy(file)
    threshold = 9
    binary_array = binarize(array, threshold=threshold)

    layer_height = 100 #nm
    growth_rate = 200 #nm/s
    k_term = 1.1 #nm

    dwell_list = []
    for n in np.linspace(0, 2, 30):
            dwell = dwell_model(layer_height, growth_rate, k_term, sigma=4, resistance=n, r=4)
            dwell_list.append(dwell)
            plt.scatter(n, dwell)

    
    plt.xlabel("Resistance")
    plt.ylabel("Dwell Time")
    plt.show()
    bins = 900
    dwell_sum = np.sort(dwell_list)
    dwell_fun = np.array(range(bins))/float(bins)

    plt.figure()
    plt.plot(dwell_sum, dwell_fun)
    plt.show()

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







