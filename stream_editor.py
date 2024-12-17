import sys
from numba import threading_layer
import numba
sys.path.append(r"C:/Users/bgh19/OneDrive/Coding/GithubRepos/f3ast")
sys.path.append(r"C:/Users/bgh19/OneDrive/Coding/GithubRepos/febid-tools/f4mine")
import f3ast
import numpy as np
import f4mine.stream_analyzer as fs
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

strm = stream_builder.get_stream()

layer = 10
plt.scatter(dwell_layers[layer][:,1], dwell_layers[layer][:,2], c=dwell_layers[layer][:,0])

# total_time = f"{strm.get_time().seconds}.{strm.get_time().microseconds/1e5:.0f}s"
# out_filename = f"{out_filename}-GR{GR0*1e3:.0f}-{total_time}"
# strm.write(f"{out_filename}.str")