#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.signal as ss
import skimage.data as sd



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

    def __init__(self,
                 pitch,
                 sigma,
                 image,
                 kernel_size=15
                 ):
        
        super().__init__()
        if kernel_size%2==0:
            print("Kernel size must be an odd number. Rounding up now.")
            kernel_size+=1

        kernel_length = int((kernel_size-1)/2)
        distances = np.arange(0, pitch*kernel_length+1, pitch)
        
        distance_arr = np.concatenate((np.flip(distances[1:kernel_size-1]), distances[:kernel_size-1]))
        gauss_arr = np.exp((-distance_arr**2)/(2*sigma**2))
        gauss_kernel = np.outer(gauss_arr.T, gauss_arr)
        self.kernel = gauss_kernel/np.sum(gauss_kernel)

        kernel_shell = np.zeros_like(image, dtype=np.float32)
        midpoints = [int(kernel_shell.shape[0]/2), int(kernel_shell.shape[1]/2)]
        print(midpoints)

        padded_kernel = kernel_shell
        padded_kernel[(midpoints[0]-kernel_length-1):(midpoints[0]+kernel_length), (midpoints[1]-kernel_length-1):(midpoints[1]+kernel_length)] = self.kernel
        self.padded_kernel = padded_kernel
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
        distances = np.arange(0, pitch*kernel_length+1, pitch)
        
        distance_arr = np.concatenate((np.flip(distances[1:kernel_size-1]), distances[:kernel_size-1]))
        distance_max = np.max(distance_arr)
        inv_gauss_arr = np.exp((distance_arr**2)/(2*sigma**2))
        inv_gauss_kernel = np.outer(inv_gauss_arr.T, inv_gauss_arr)
        self.kernel = inv_gauss_kernel/np.sum(inv_gauss_kernel)

        kernel_shell = np.zeros_like(image, dtype=np.float32)
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




    gauss = Gauss(20,4,image)
    gauss.display_kernel()


    invgauss = InverseGauss(20,4,image)
    invgauss.display_kernel()


    plt.figure()
    plt.imshow(image)

    newmat = apply_filter(image, gauss)
    plt.figure()
    plt.imshow(newmat.real)
    plt.title("Gaussian")
    plt.colorbar()

    invmat = apply_filter(image, invgauss)
    plt.figure()
    plt.imshow(invmat.real)
    plt.title("Inverse Gaussian")
    plt.colorbar()



# %%
