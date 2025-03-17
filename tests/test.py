from src.fast3Dinterp import fast3Dinterp
import numpy as np

# Create or load a 3D array with NaN values
array_3d = np.random.rand(50, 50, 10)
array_3d[array_3d < 0.1] = np.nan

# Interpolate NaNs
result = fast3Dinterp(array_3d, 1, 1000)