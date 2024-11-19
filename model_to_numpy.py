import trimesh
import trimesh.voxel.ops as to
import numpy as np

mesh = trimesh.load_mesh("CoilBlockOnBottom.stl")

mesh.show()

meshvox = to.matrix_to_points