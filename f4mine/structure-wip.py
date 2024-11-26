import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sm
import skimage.graph as sg
import scipy.signal as sps
import trimesh as tm
import trimesh.voxel as tv
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


class Structure:

    def __init__(self,
                 filepath,
                 pitch,
                 threshold,
                 ):
        
        if filepath.endswith(".stl"):
            self.binary_matrix = import_stl(filepath, pitch)
        elif filepath.endswith(".npy"):
            self.binary_matrix = import_numpy()

def import_stl(filepath, pitch):
    stl_struct = tm.load_mesh(filepath)

    struc_mat = stl_struct.voxelized(pitch=pitch).fill()
    nummat = np.asanyarray(struc_mat.matrix)
    return nummat

def import_numpy(filepath, threshold):
    epsilon_array = np.load(filepath)
    binary_array = np.where(epsilon_array>threshold, False, True)
    return binary_array

def binarize(array, threshold):


def calculate_neighbors(binary_array):
    ##Only calculate those adjacent and below the pixel
    arr_size = binary_array.shape
    neighbor_kernel = np.ones((5,5,5))
    neighbor_kernel[:, :, :2] = 0
    neighbor_kernel[2,2,2] = 0
    
    
    neighbor_arr = sps.convolve(binary_array, neighbor_kernel)
    
    return neighbor_arr

### Harmonic series approximate sum
def harmonic(f):
    f = round(f)
    if f>0:
        invres = ((np.log(f) + .5772156649) + 1/(2*f)) #Euler-Mascheroni
        
        
    else:
        res = 0
        
    return res



def calculate_resistance(binary_array, single_pixel_width = 50 ):
    neighbors = calculate_neighbors(binary_array)


    harmvec = np.vectorize(harmonic)
    
    resistances = harmvec(neighbors)
    return resistances


# def calculate_dwell(resistances, growth_rate, k):
    

# connection_resistance = resistances_below[c] + 

# single_pixel_width * layer_sep / (brlens[j] + single_pixel_width)
                
                
            



if __name__ == "__main__":
    file = "TestGyroidEpsilonArray.npy"
    array = import_numpy(file)
    threshold = 4
    binary_array = binarize(array, threshold=threshold)
    # print(binary_array.shape)  
    
    resists = calculate_resistance(binary_array)
    
    print(resists[:,:,15])
    plt.figure()
    plt.imshow(resists[:,:,15])
    plt.colorbar()
    
    # plt.imshow(binary_array[:,:,11])
    # plt.colorbar()
    # labels, nums = sm.label(binary_array[:,:,15], return_num=True)
    # print(nums)
    # for k, n in enumerate(labels):
    #     print()
    