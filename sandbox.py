

#%%
import numpy as np
import trimesh as tm
import skimage.measure as skm
import matplotlib.pyplot as plt
import trimesh.voxel.ops as ops
import skimage.morphology as morph
import skimage.filters as filt
is_interactive = True

npyfile = "Test3DEpsilonArray.npy"

npyfile = npyfile if is_interactive==True else "LocalCoding/"+npyfile
epsilon_array = np.load(npyfile)

#%%
new_array = epsilon_array
lev = 8


addressable_pixels = [65536, 56576]
screen_width = 10.2e3

pixel_size = screen_width/addressable_pixels[0] 
print(pixel_size)
field_center = [addressable_pixels[0]/2, addressable_pixels[1]/2]

structure_size_nm = 1000 #nm
structure_size = structure_size_nm/pixel_size
structure_xspan = [int(field_center[0] - structure_size/2), int(field_center[0] + structure_size/2)]
structure_yspan = [int(field_center[1] - structure_size/2), int(field_center[1] + structure_size/2)]
print(f"the layer array is size {zslice.shape}")
print(f"the print area is {structure_xspan}")

step_size = structure_size/zslice.shape[0]

point_xrange = range(structure_xspan[0], structure_xspan[1], int(step_size))
point_yrange = range(structure_yspan[0], structure_yspan[1], int(step_size))

dwell_time = 32000

output_array = np.where(skeleton_array !=0, dwell_time, 0)
flatter = np.reshape(output_array, -1)
dwells = []
xposs = []
yposs = []




