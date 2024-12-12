import numpy as np
import scipy.ndimage as snd

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
  
    
    neighbor_kernel = np.zeros((9,9,1))
    neighbor_kernel[1:7, 1:7, 0]= np.exp(-((1*pitch)**2) / (2*sigma**2))
    neighbor_kernel[2:6, 2:6, 0]= np.exp(-((2*pitch)**2) / (2*sigma**2))
    neighbor_kernel[3:5, 3:5,0] = np.exp(-((3*pitch)**2) / (2*sigma**2))
    neighbor_kernel[4, 4, 0] = np.exp(-(4*pitch**2) / (2*sigma**2))

    neighbor_arr = snd.correlate(binary_array.astype(int), neighbor_kernel, mode='constant', cval=0)


    return neighbor_arr

