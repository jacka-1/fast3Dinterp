import numpy as np

def fast3Dinterp(array_3d, tol, maxIter, init_val=None):
    """
    Interpolate NaN values in a 3D array using a Gauss-Seidel method.
    
    Parameters:
    -----------
    array_3d : numpy.ndarray
        The 3D array containing NaN values to be interpolated.
    tol : float, optional
        The convergence tolerance
    maxIter : int, optional
        Maximum number of iterations
    init_val : float, optional
        If provided, used as the initial fill value for NaNs; otherwise,
        each frame's NaNs are replaced with the frame mean (or 0 if all NaN).
    
    Returns:
    --------
    interpolatedData : numpy.ndarray
        The 3D array with NaN values interpolated.
    """
    size = array_3d.shape
    nanIndex = np.isnan(array_3d).nonzero()
    interpolatedData = array_3d.copy()

    # for each frame, assign either init_val, frame mean or zero as initial value to nans
    for i in range(array_3d.shape[2]):
        frame = array_3d[:, :, i]
        if init_val is not None:
            fill_val = init_val
        else:
            if np.all(np.isnan(frame)):
                fill_val = 0 
            else:
                fill_val = np.nanmean(frame)
        nan_frame_index = np.where(np.isnan(frame))
        interpolatedData[nan_frame_index[0], nan_frame_index[1], i] = fill_val
    print('Filled NaN with initial values')

    def sign(index, max_index):
        if index == 0:
            return [1, 0]  # only average with the next element
        elif index == max_index - 1:
            return [-1, 0]  # only average with the previous element
        else:
            return [-1, 1]  # average with both previous and next elements

    # Listing indexes of values that need interpolating
    filtered_nanIndex = [(x, y, z) for x, y, z in zip(*nanIndex)]
    if filtered_nanIndex:
        nanIndexX, nanIndexY, nanIndexZ = zip(*filtered_nanIndex)
        nanIndexX = np.array(nanIndexX)
        nanIndexY = np.array(nanIndexY)
        nanIndexZ = np.array(nanIndexZ)
    else:
        nanIndexX, nanIndexY, nanIndexZ = np.array([]), np.array([]), np.array([])

    print(f'Number of points to interpolate: {len(nanIndexX)}')
    print(f'{len(nanIndexX)*100/(size[0]*size[1]*size[2]):.2f} % of the initial 3D array')

    # Sign for each dimension separately
    signsX = np.array([sign(x, size[0]) for x in nanIndexX])
    signsY = np.array([sign(y, size[1]) for y in nanIndexY])
    signsZ = np.array([sign(z, size[2]) for z in nanIndexZ])
    
    # Start iterations of interpolation
    for r in range(maxIter):
        print('Round:', r)
        print('Points left:', len(signsX))
        if len(nanIndexX) > 0:
            d = 0
            for i in range(len(nanIndexX)):
                idx = i - d 
                x, y, z = nanIndexX[idx], nanIndexY[idx], nanIndexZ[idx]
                dx, dy, dz = signsX[idx], signsY[idx], signsZ[idx]
                neighbors = []
                if dx[0] != 0:  # average with the previous x neighbor
                    neighbors.append(interpolatedData[np.clip(x + dx[0], 0, size[0] - 1), y, z])
                if dx[1] != 0:  # average with the next x neighbor
                    neighbors.append(interpolatedData[np.clip(x + dx[1], 0, size[0] - 1), y, z])
                if dy[0] != 0:  # average with the previous y neighbor
                    neighbors.append(interpolatedData[x, np.clip(y + dy[0], 0, size[1] - 1), z])
                if dy[1] != 0:  # average with the next y neighbor
                    neighbors.append(interpolatedData[x, np.clip(y + dy[1], 0, size[1] - 1), z])
                if dz[0] != 0:  # average with the previous z neighbor
                    neighbors.append(interpolatedData[x, y, np.clip(z + dz[0], 0, size[2] - 1)])
                if dz[1] != 0:  # average with the next z neighbor
                    neighbors.append(interpolatedData[x, y, np.clip(z + dz[1], 0, size[2] - 1)])
                
                # Average with identified neighbors to interpolate the NaN value
                point = np.nanmean(neighbors)

                if abs(interpolatedData[x, y, z] - point) < tol:
                    # Value has converged; remove from list
                    nanIndexX = np.delete(nanIndexX, idx, axis=0)
                    nanIndexY = np.delete(nanIndexY, idx, axis=0)
                    nanIndexZ = np.delete(nanIndexZ, idx, axis=0)
                    signsX = np.delete(signsX, idx, axis=0)
                    signsY = np.delete(signsY, idx, axis=0)
                    signsZ = np.delete(signsZ, idx, axis=0)
                    d += 1
                else:
                    interpolatedData[x, y, z] = point
        else:
            break
    return interpolatedData
