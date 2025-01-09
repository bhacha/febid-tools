#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.signal as ss
import skimage.data as sd
from scipy.special import expit, logit


class Filter:

    def __init__(self):
        pass

    def display_kernel(self):
        self._display(self.kernel, "Kernel")

    def display_padded(self):
        self._display(self.padded_kernel, "Padded Kernel")

    def display_fft(self):
        self._display(self.filter.real, "FFT Filter")

    def _display(self, variable, title):
        plt.figure()
        plt.imshow(variable)
        plt.colorbar()
        plt.title(title)  

class Gauss(Filter):
    
    """
    Create a Gaussian kernel and then output the Fourier transformed kernel for convolution
    
    """

    def __init__(self,
                 pitch,
                 sigma,
                 image,
                 kernel_size=15
                 ):
        
        """
        The filter accepts a pitch (this is used to calculate the distance) and sigma (distance is relative to the sigma) to calculate a Gaussian array and fourier transform it.
        
        Parameters
        ------------
        
        pitch : float
            Nanometers per pixel in the resistance array.
            
        sigma : float
            approximately the beam radius. The gaussian is an exponential with pitch^2/sigma^2. 
            
        image : ndarray
            Resistance array or array of same size. This isn't actually used yet, it just sets the size of the kernel
        
        kernel_size : int
            number of pixels to actively calculate in the kernel. The kernel is technically the same size as the image, but it's padded with 0 outside of this size
             
        
        """
        
        super().__init__()
        if kernel_size%2==0:
            print("Kernel size must be an odd number. Rounding up now.")
            kernel_size+=1

        kernel_length = int((kernel_size-1)/2)
        
        ## Calculate a list of positions from 0 (beam center)
        distances = np.arange(0, pitch*kernel_length+1, pitch)
        
        ### beam is symmetric, so copy the distances and mirror them around 0. Sign doesn't matter.
        distance_arr = np.concatenate((np.flip(distances[1:kernel_size-1]), distances[:kernel_size-1])).astype(np.float64)
        
        #calculate the exponential term for each distance, making a 1D kernel
        gauss_arr = expit((-distance_arr**2)/(2*sigma**2))
        
        # multiply the row and column together to make a 2D kernel
        gauss_kernel = np.outer(gauss_arr.T, gauss_arr)
        
        #normalize kernel
        self.kernel = gauss_kernel/np.sum(gauss_kernel)


        #center the active kernel and pad to full size
        kernel_shell = np.zeros_like(image, dtype=np.float32)
        midpoints = [int(kernel_shell.shape[0]/2), int(kernel_shell.shape[1]/2)]
        padded_kernel = kernel_shell
        padded_kernel[(midpoints[0]-kernel_length-1):(midpoints[0]+kernel_length), (midpoints[1]-kernel_length-1):(midpoints[1]+kernel_length)] = self.kernel
        self.padded_kernel = padded_kernel
        #fourier transform
        self.gauss_fft = np.fft.fft2(padded_kernel)
   
        self.filter = self.gauss_fft

class InverseGauss(Filter):
     def __init__(self,
                 pitch,
                 sigma,
                 image,
                 kernel_size=10
                 ):
        
        super().__init__()
        if kernel_size%2==0:
            print("Kernel size must be an odd number. Rounding up now.")
            kernel_size+=1

        kernel_length = int((kernel_size-1)/2)
        
        max_distance_threshold = (10*sigma)
        max_distance_val = 25*sigma
        
        distances = np.arange(0, pitch*kernel_length+1, pitch)
        distances = np.where(distances <= max_distance_threshold, distances, max_distance_val)
        distance_arr = np.concatenate((np.flip(distances[1:kernel_size-1]), distances[:kernel_size-1])).astype(np.float64)
        
    
        inv_gauss_arr = expit((distance_arr**2)/(2*(sigma**2)))
        exploded_inv_arr = np.outer(inv_gauss_arr.T, inv_gauss_arr)
        inv_gauss_kernel = np.where(((exploded_inv_arr <= 1e6)), exploded_inv_arr, .01)
        self.kernel = inv_gauss_kernel/np.sum(inv_gauss_kernel)

        kernel_shell = np.zeros_like(image, dtype=np.float64)
        midpoints = [int(kernel_shell.shape[0]/2), int(kernel_shell.shape[1]/2)]
        padded_kernel = kernel_shell
        padded_kernel[(midpoints[0]-kernel_length-1):(midpoints[0]+kernel_length), (midpoints[1]-kernel_length-1):(midpoints[1]+kernel_length)] = self.kernel
        self.padded_kernel = padded_kernel
        self.inv_gauss_fft = np.fft.fft2(padded_kernel)
   
        self.filter = self.inv_gauss_fft



def apply_filter(image, Filter, inverse=False):
    """
    Apply a Filter object to an image/numpy stack.
    """
    filter = Filter.filter

    fft_image = np.fft.fft2(image.real)

    if inverse==True:
        convolve = fft_image/filter
        
    else:
        convolve = fft_image*filter

    output = np.fft.fftshift(np.fft.ifft2(convolve))
    return output
    



if __name__ == "__main__":
    image = np.zeros((200,200))
    image[80:110, 80:110] = 1


    filter_matrix = np.ones((21,21))
    filter_matrix[5:15, 5:15] = 20

    # ffmat = np.fft.fft2(matrix)
    # ffmat[70:, 70:] = 0
    # iffmat = np.fft.ifft2(ffmat)

    # plt.figure()
    # plt.imshow(matrix)

    # plt.figure()
    # plt.imshow(ffmat.real, norm='log')

    # plt.figure()
    # plt.imshow(iffmat.real)




    # gauss = Gauss(20,4,image)
    # gauss.display_kernel()


    invgauss = InverseGauss(20,4,image, kernel_size=31)
    # invgauss.display_kernel()


    # plt.figure()
    # plt.imshow(image)

    # newmat = apply_filter(image, gauss)
    # plt.figure()
    # plt.imshow(newmat.real)
    # plt.title("Gaussian")
    # plt.colorbar()

    invmat = apply_filter(image, invgauss)
    plt.figure()
    plt.imshow(invmat.real)
    plt.title("Inverse Gaussian")
    plt.colorbar()



# %%
