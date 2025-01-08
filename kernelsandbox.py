#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd








matrix = np.ones((100,100))
matrix[40:60, 30:50] = 2
# %%

gaussian_kernel =np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6,24,36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])

gaussmat= nd.correlate(matrix, gaussian_kernel)

plt.figure()
plt.imshow(matrix)
plt.figure()
plt.imshow(gaussmat)
# %%

inverse_gaussian = 1/gaussian_kernel

newmat = nd.correlate(gaussmat, inverse_gaussian)
plt.figure()
plt.imshow(newmat)


# %%
