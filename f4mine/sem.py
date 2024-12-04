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
        self.pixels = np.asarray(addressable_pixels)
        self.width = screen_width
        self.pixel_size = screen_width/self.pixels[0] 
        self.field_center = [self.pixels[0]/2, self.pixels[1]/2]
        
    