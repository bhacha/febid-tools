

#%%
import numpy as np
import trimesh as tm
import skimage.measure as skm
import matplotlib.pyplot as plt
import trimesh.voxel.ops as ops
import skimage.morphology as morph
import skimage.filters as filt
import skimage.transform as skt



def process_array(array, **kwargs):
        
    threshold = kwargs.get('Threshold', None)
    skeleton_flag = kwargs.get('Skeletonize', None)
    resize = kwargs.get('Resize', None)

    output_array = array
    
    ### Binarization
    if threshold == None:
        pass
    else:
        output_array = binarize_array(array, threshold)
     
    ### Skeletonization   
    if (skeleton_flag == None) or (skeleton_flag == False):
        pass
    else:
        print("Skeletonizing")
        binary_array = binarize_array(array, threshold)
        output_array = morph.skeletonize(binary_array)
        
    ### Resizing
    if resize == None:
        pass    
    else:
        resized_array = resize_array(output_array, resize)
        output_array = resized_array    

    
    return output_array            

def binarize_array(array, threshold):
    print(f"Making array binary, with a threshold of {threshold:,.2f}")
    binary_array = np.where(array<threshold, 1, 0)
    return binary_array
 
def resize_array(array, new_size):
    print(f"Resizing array to {new_size}")
    resized_array = skt.resize(array, output_shape=new_size)
    return resized_array

def output_stream(filename, xpos, ypos, dwell_time):
    numpoints = str(len(xpos))
    with open(filename+'.str', 'w') as f:
        f.write('s16\n1\n')
        f.write(numpoints+'\n')
        for k in range(int(numpoints)):
            xstring = str(xpos[k])
            ystring = str(ypos[k])
            dwellstring = str(dwell_time[k])
            linestring = dwellstring+" "+xstring+" "+ystring+" "
            if k < int(numpoints)-1:
                f.write(linestring + '\n')
            else:
                f.write(linestring + " "+"0")
     
class Streams():
    """
    Holds the important stream information (dwells, x, y, etc.)
    
    Attributes
    -----------
    addressable_pixels : ndarray
        addressable pixels in X and Y
    
    screen_width : float
        Horizontal FOV in nanometers    
    
    pixel_size : float
        X pixel size calculated from 'screen_width' and 'addressable pixels'. Note this is not a beam size!
    
    field_center : ndarray
        Center of the fabricatable region, in pixels
        

    """   
    def __init__(
        self,
        addressable_pixels,
        screen_width
        ):
        """
        Parameters
        -----------
        
        addressable_pixels : length-2 list, ndarray, tuple
            number of discrete pixels that the beam can be steered to, in format (X,Y)
            
        screen_width : float
            Horizontal field of view, in nanometers
            
        """

        self.pixels = np.asarray(addressable_pixels)
        self.width = screen_width
        self.pixel_size = screen_width/self.pixels[0] 
        self.field_center = [self.pixels[0]/2, self.pixels[1]/2]
    
    def array_to_stream(self, array, structure_size_nm, **kwargs):
        """
        take numpy array, convert to a stream file
        
        Parameters
        -----------
        array : ndarray
            3D array (TBD boolean or float)
        
        structure_size_nm : float
            desired size of structure in nanometers
        
        """
        
        dwell_time_orig = 20000

        array = process_array(array, **kwargs)
        
        image_size = array.shape[:2]
        
        ### convert from nm to px size
        structure_size = structure_size_nm/self.pixel_size
        
        
        structure_xspan = [int(self.field_center[0] - structure_size/2), int(self.field_center[0] + structure_size/2)]
        structure_yspan = [int(self.field_center[1] - structure_size/2), int(self.field_center[1] + structure_size/2)]

        '''I use range in the next step because linspace gives integers and I don't want to worry about rounding methods causing  distortion. The tradeoff is that I need to calculate the step size instead of the number of steps. This will get rounded, but then apply the same step size to everything. I think this would reduce distortion since values won't round away from one another.
        '''
        step_size_x = structure_size/image_size[0]
        step_size_y = structure_size/image_size[1]
        
        
        point_xrange = range(structure_xspan[0], structure_xspan[1], int(step_size_x))
        point_yrange = range(structure_yspan[0], structure_yspan[1], int(step_size_y))

        dwells = []
        xposs = []
        yposs = []
        k = array.shape[2]
        for slice in array:
            k += 1
            for x in range(slice.shape[0]):
                xpos = point_xrange[x]
                for y in range(slice.shape[1]):
                    ypos = point_yrange[y]
                    if slice[x, y] != 0:
                        dwell_time = round(dwell_time_orig / (np.exp(-(.45*k/(array.shape[2])))))
                        dwells.append(dwell_time)
                        xposs.append(xpos)
                        yposs.append(ypos)
                    else:
                        pass
        
                              
        return dwells, xposs, yposs

        
  
#%%

structure_size_nm = 1000 #nm

npyfile = "Test3DEpsilonArray.npy"
test_array = np.load(npyfile)
print(test_array.shape)
# test_array = test_array[:,:,:40]

addressable_pixels = [65536, 56576]
screen_width = 10.2e3

        
scope = SEM(addressable_pixels, screen_width)

streams = Streams(scope)

out1 = streams.array_to_stream(test_array, structure_size_nm, Threshold = 10)
        
output = output_stream("teststream2-longer", out1[1], out1[2], out1[0])
        
out = process_array(test_array, Threshold=10, MaxIters=8)

plt.imshow(out[:,:,10], cmap='binary_r')
# print(out[100,0,10])
# %%
