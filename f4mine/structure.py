import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import skimage.graph as sg
import scipy.signal as sps
np.set_printoptions(threshold=np.inf)

def import_numpy(filepath):
    epsilon_array = np.load(filepath)
    return epsilon_array

class Structure:
    
    def __init__(self):
        pass

def binarize(array, threshold):
    binary_array = np.where(array>threshold, False, True)
    return binary_array

def calculate_neighbors_kernel(binary_array):
    ##Only calculate those adjacent and below the pixel
    arr_size = binary_array.shape
    neighbor_kernel = np.ones((5,5,5))
    ### These set the top two layers (z 0 and 1) to be 0. This means contributions from points above don't affect the convolved output
    ### The idea is that each point depends only on the points below it.

    ## This should give a pretty decent local approximation, but does not give the full branch_lengths.
    neighbor_kernel[:, :, :2] = 0
    neighbor_kernel[2,2,2] = 0
    
    
    neighbor_arr = sps.convolve(binary_array, neighbor_kernel)
    
    return neighbor_arr

def calculate_regions_in_slices(binary_array):
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
    full_labels = []
    area_labels = []
    for i in range(binary_array.shape[-1]):
        """
        Going slice by slice, label connected regions and calculate their area

        A single layer of the resulting area array has the same number of entries as there are labelled regions. The caveat is that labels start at i=1 and the list of areas starts with l=0. So to get the area of the region labelled 2, you would need to query area[1]. this poses problems with indexing the area array *using* the labels array. 

        The fix is to append a 0 to the beginning of the list. This should never be accessed, but if it is we will know because area of 0 isn't possible (it wouldn't be labelled)

        """
        slice = binary_array[...,i]
        

        slice_labels = sm.label(slice)
        regions = sm.regionprops(slice_labels)
        """
        Create a list of the areas of the labelled regions in each slice, then make a list of those lists to get a list for all the slices.
        
        """
        slice_areas = []
        for n in regions:
            slice_areas.append(n.area)
        
        # put a padding zero at the front, as mentioned above
        slice_areas.insert(0, 0)

        ### Append to the whole-structure lists
    
        area_labels.append(slice_areas)
        full_labels.append(slice_labels)
        """
        I want the indices in the original matrix that are labelled. This corresponds to the indices of the elements that are connected in the original matrix. 
        """
        nonzero_indices.append(np.nonzero(slice_labels))


    return full_labels, nonzero_indices, area_labels

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


### Harmonic series approximate sum
def harmonic(f):
    """Since parallel resistance adds inverses, I am using the Euler-Mascheroni equation to quickly calculate the total"""
    f = round(f)
    if f>0:
        invres = ((np.log(f) + .5772156649) + 1/(2*f)) #Euler-Mascheroni
        
        
    else:
        res = 0
        
    return res




# def calculate_dwell(resistances, growth_rate, k):
    
                
                
            



if __name__ == "__main__":
    file = "Test3DEpsilonArray.npy"
    array = import_numpy(file)
    threshold = 9
    binary_array = binarize(array, threshold=threshold)
    ## Add a block to help orient things
    binary_array[30:90, 50:90, :20] = True

    # print(binary_array.shape)  
    print(binary_array.shape)
    
    labels, indices, areas = calculate_regions_in_slices(binary_array)


    label_slice = labels[0]
    area_slice = np.asarray(areas[0])

    area = area_slice[label_slice]
    print




"""
The process to calculate resistance will be like this:

iterate over labelled regions. For each point in the region, query the area of that region using 

"""
















def create_area_array(binary_array):
    templabels = np.zeros_like(label_slice)
    m = 0
   
    for n in label_slice:
        w = 0
        for k in n:
            templabels[m, w] = area_slice[k]
            w += 1

        m += 1









# for k, label_layer in enumerate(labels[0:2]):
#     if k == 0:
#         previous_layers = np.zeros_like(label_layer)
#     else:
#         for index_position, label in enumerate(label_layer):
#             pass


    







    # _, points = calculate_z_length(binary_array)

    # points = np.asarray(points).reshape(-1, 3)
    # ax = fig.add_subplot(projection='3d')   
    # ax.scatter(points[:,0], points[:,1], points[:,2])
    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label")



    # plt.show()
    # plt.imshow(labels[0], cmap='Paired')
    # plt.colorbar()
    # labels, nums = sm.label(binary_array[:,:,15], return_num=True)
    # print(nums)
    # for k, n in enumerate(labels):
    #     print()
    