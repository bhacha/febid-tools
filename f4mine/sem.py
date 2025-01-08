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

