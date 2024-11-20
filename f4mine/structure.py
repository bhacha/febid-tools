import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import skimage.graph as sg
import scipy.ndimage as snd
np.set_printoptions(threshold=np.inf)

### Harmonic series approximate sum
def harmonic(f):
    """Since parallel resistance adds inverses, I am using the Euler-Mascheroni equation to quickly calculate the total"""
    f = round(f)
    if f>0:
        invres = ((np.log(f) + .5772156649) + 1/(2*f)) #Euler-Mascheroni
        
        
    else:
        res = 0
        
    return res

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

def calculate_neighbors_kernel(binary_array, pitch, sigma):
    """ I'm abandoning this in favor the other options"""
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

def calculate_dwells(binary_array, layer_height, growth_rate, k, sigma):

    resistance_array = calculate_resistance(binary_array)


    dwell_array = growth_rate
    



if __name__ == "__main__":
    file = "Test3DEpsilonArray.npy"
    array = import_numpy(file)
    threshold = 9
    binary_array = binarize(array, threshold=threshold)

    layer_height = 100 #nm
    growth_rate = 200 #nm/s
    k_term = 1.1 #nm


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







