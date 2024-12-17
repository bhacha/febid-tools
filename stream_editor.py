import sys
from numba import threading_layer
import numba
sys.path.append(r"C:/Users/bgh19/OneDrive/Coding/GithubRepos/f3ast")
sys.path.append(r"C:/Users/bgh19/OneDrive/Coding/GithubRepos/febid-tools/f4mine")
import f3ast
import numpy as np
import f4mine.stream_analyzer as fs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


file_path = "branchtrio-SizedUp.stl"
out_filename = "branchtrio_3x"

GR0 = 250e-3 # in um/s, base growth rate
k = 1.25 # in 1/nm?, thermal conductivity 
sigma = 4.4 # in nm, dwell size





settings = {}
settings["structure"] = {"pitch": 3, "fill": False}  # in nm
settings["stream_builder"] = {
    "addressable_pixels": [65536, 56576],
    "max_dwt": 5,  # in ms
    "cutoff_time": 0.01,  # in ms, for faster exporting: remove dwells below cutoff time
    "screen_width": 10.2e3,  # in nm, horizontal screen width / field of view
    # 'serpentine' or 'serial', scanning order between slices
    "scanning_order": "serpentine",
}
# pixel size for thermal resistance
settings["dd_model"] = {"single_pixel_width": 50}



struct = f3ast.Structure.from_file(file_path, **settings["structure"])
struct.mirror(normal=(1, 0, 0))
struct.centre()  # centers xy to zero and sets minimum z value to zero
struct.rescale(.05)  # scale the structure 3x

model = f3ast.DDModel(struct, GR0, k, sigma, **settings['dd_model'])
stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(model, **settings['stream_builder'])

dwell_layers = dwell_solver.get_dwells_slices()

# strm = stream_builder.get_stream()

# layer = 10
# plt.scatter(dwell_layers[layer][:,1], dwell_layers[layer][:,2], c=dwell_layers[layer][:,0])

# total_time = f"{strm.get_time().seconds}.{strm.get_time().microseconds/1e5:.0f}s"
# out_filename = f"{out_filename}-GR{GR0*1e3:.0f}-{total_time}"
# strm.write(f"{out_filename}.str")





def convert_array(array, addressable_pix, screen_width):

    ppn = addressable_pix[0]/screen_width

    center=np.array([int(addressable_pix[0]/2), int(addressable_pix[1]/2)])
    scaled_array=array[:,:3]
    scaled_array[:, 1:] *= ppn
    scaled_array[:, 1] +=  center[0]
    scaled_array[:, 2] +=  center[1]
    return scaled_array





def condense_dwells(radius, slices, calculate_new_time=True):
    condensed_point_list = []
        ### Iterate through the regions, then cluster the points using DBSCAN
    for n in range(len(slices)):
        point_array_regions = convert_array(slices[n], addressable_pix=[65536, 56576], screen_width=10.2e3)
        dwells = point_array_regions[:, 0].astype(float)
        points = point_array_regions[:, 1:].astype(float)
        clusters = DBSCAN(eps=radius, min_samples = int(10)).fit(points)
        labels = clusters.labels_
        unique_labels = set(labels)
            ### for each cluster of points, calculate the total dwell time in that region (in seconds)
        for regions in range(len(unique_labels)):
            point_in_region = points[labels==regions, :]
            centroid = np.mean(point_in_region, axis=0).astype(int)
            dwell_sum = int(np.sum(dwells[labels==regions]))
            if (np.all((centroid>0)) and dwell_sum>0):
                condensed_point = np.append(dwell_sum, centroid)
                condensed_point_list.append(condensed_point)
            else:
                pass
    
    condensed_point_array = np.asarray(condensed_point_list)
    print(condensed_point_array)
    print(condensed_point_array.shape)
    if calculate_new_time == True:
            dwells = condensed_point_array[:, 0]
            print(f"The total time of the condensed points is {np.sum(dwells)/1e7:.2f} seconds")

    return condensed_point_array


def write_stream(output_file, numpoints, dwell_times, xpos, ypos):
    numpoints = str(numpoints)

    with open(output_file, 'w') as f:
        f.write('s16\n1\n')
        f.write(numpoints+'\n')
    for k in range(int(numpoints)):
        xstring = str(xpos[k])
        ystring = str(ypos[k])
        dwellstring = str(dwell_times[k])
        linestring = dwellstring+" "+xstring+" "+ystring+" "
        if k < int(numpoints)-1:
            f.write(linestring + '\n')
        else:
            f.write(linestring + " "+"0")   


def dither_stream(condensed_points,  rad, sigma, dwell_time_max_ms = 3.5):
    "Taking the centroids and dwells, space them out a little bit into multiple points so that the dwells are shorter. This uses the sigma from the gaussian spot to keep points in the same general neighborhood from the fab perspective."
    
    stream_array = condensed_points
    dwell_time_max = dwell_time_max_ms * 1e-3 * 1e7 
    dwell_times = stream_array[:,0]
    xpos = stream_array[:, 1]
    ypos = stream_array [:, 2]

    out_xpos = []
    out_ypos = []
    out_dwells = []
    total_dwell = 0
    for number, _ in enumerate(xpos):
        x = xpos[number]
        y = ypos[number]
        mean = (x,y)
        cov = [[rad**2/(sigma**2),0], [0, rad**2/(sigma**2)]]
        number_dwells = int(dwell_times[number]/dwell_time_max)
        distributed_points = np.random.multivariate_normal(mean, cov=cov, size=(number_dwells)).astype(int)
        for points in distributed_points:
            distance = np.linalg.norm(points-np.array((x, y)))
            new_dwell = int(np.exp(-(distance**2) / (2*sigma**2))*(dwell_times[number]/number_dwells))
            if new_dwell > 10 :
                out_xpos.append(points[0])
                out_ypos.append(points[1])
                out_dwells.append(new_dwell)
                total_dwell += new_dwell
            else:
                pass

    numpoints = len(out_xpos)
    outfile = "dithered-newgyr-r402.str"
    write_stream(numpoints=numpoints, dwell_times=out_dwells, xpos=out_xpos, ypos=out_ypos, output_file=outfile)

    print(f"The total time of the dithered points is {total_dwell/1e7:.2f} seconds")  




radius = 20


condendwell = condense_dwells(radius=radius, slices = dwell_layers)

# condensed_point_list = []
# ### Iterate through the regions, then cluster the points using DBSCAN

# for n in range(10,20,2):
#     dwells = dwell_layers[n][:, 0].astype(float)
#     points = dwell_layers[n][:, 1:3].astype(float)
#     clusters = DBSCAN(eps=radius, min_samples = int(3)).fit(points)
#     labels = clusters.labels_
#     unique_labels = set(labels)
    
#     ### for each cluster of points, calculate the total dwell time in that region (in seconds)
# for regions in range(len(unique_labels)):
#     point_in_region = points[labels==regions, :]
#     # print(point_in_region.shape)
#     centroid = np.mean(point_in_region, axis=0).astype(float)
#     print(centroid)
#     dwell_sum = float(np.sum(dwells[labels==regions]))
#     if (np.all((centroid>0)) and dwell_sum>0):
#         condensed_point = np.append(dwell_sum, centroid)
#         condensed_point_list.append(condensed_point)
#     else:
#         pass
        
# condensed_point_array = np.asarray(condensed_point_list)

# print(condensed_point_array)