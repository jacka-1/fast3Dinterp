import numpy as np

def fast3Dinterp(array_3d, tol, maxIter, init_val = None):
        """
        A function that inputs a 3D array that contains nans, and outputs the 3D array where the nan values
        have been replaced by linear interpolation, using an iteratibe Gauss-Seidel approach
        tol: if chnaging by less than this value between subsequnet itertions, the element has converged and is assumed interpolated
        maxIter: the maximum number of iterations to allow convergance
        init_value: (optional) the value to initially fill nans with (may improve convergance)
        """
        size = array_3d.shape
        nanIndex = np.isnan(array_3d).nonzero()
        interpolatedData = array_3d.copy()

        # for each frame, assign either frame mean, init_val or zero as initial value to nans
        for i in range(array_3d.shape[2]):
            frame = array_3d[:, :, i]
            if init_val:
                fill_val = init_val
            else:
                if np.all(np.isnan(frame)):
                    fill_val = 0 
                else:
                    fill_val = np.nanmean(frame)
                nan_frame_index = np.where(np.isnan(frame))
            interpolatedData[nan_frame_index[0], nan_frame_index[1], i] = fill_val
        print('filled nan with initial values')

        def sign(index, max_index):
            if index == 0:
                return [1, 0]  # only average with the next element
            elif index == max_index - 1:
                return [-1, 0]  #only average with the previous element
            else:
                return [-1, 1]  #average with both previous and next elements

        # listing indexes of values that need interpolating
        filtered_nanIndex = [(x, y, z) for x, y, z in zip(*nanIndex) if (x, y, z)]
        nanIndexX, nanIndexY, nanIndexZ = zip(*filtered_nanIndex) if filtered_nanIndex else (np.array([]), np.array([]), np.array([]))

        print(f'number of points to interpolate: {len(nanIndexX)}')
        print(f'{len(nanIndexX)*100/(size[0]*size[1]*size[2])} % of the initial 3d array')

        # sign for each dimension separately
        signsX = np.array([sign(x, size[0]) for x in nanIndexX])
        signsY = np.array([sign(y, size[1]) for y in nanIndexY])
        signsZ = np.array([sign(z, size[2]) for z in nanIndexZ])
        
        # start iterations of interpolation
        for r in range(maxIter):
            print('round:', r)
            print('points left:', len(signsX))
            if len(nanIndexX) > 0:
                d = 0
                for i in range(len(nanIndexX)):
                    idx = i-d 
                    x, y, z = nanIndexX[idx], nanIndexY[idx], nanIndexZ[idx] #assuming all nanIndex arrays are the same length
                    dx, dy, dz = signsX[idx], signsY[idx], signsZ[idx]
                    neighbors = []
                    if dx[0] != 0:  #average with the previous x neighbor
                        neighbors.append(interpolatedData[np.clip(x + dx[0], 0, size[0] - 1), y, z])
                    if dx[1] != 0:  #average with the next x neighbor
                        neighbors.append(interpolatedData[np.clip(x + dx[1], 0, size[0] - 1), y, z])

                    if dy[0] != 0:  #average with the previous y neighbor
                        neighbors.append(interpolatedData[x, np.clip(y + dy[0], 0, size[1] - 1), z])
                    if dy[1] != 0:  #average with the next y neighbor
                        neighbors.append(interpolatedData[x, np.clip(y + dy[1], 0, size[1] - 1), z])

                    if dz[0] != 0:  #average with the previous z neighbor
                        neighbors.append(interpolatedData[x, y, np.clip(z + dz[0], 0, size[2] - 1)])
                    if dz[1] != 0:  #average with the next z neighbor
                        neighbors.append(interpolatedData[x, y, np.clip(z + dz[1], 0, size[2] - 1)])
                    # average with identified neighbors to interpolate the NaN value
                    point = np.nanmean(neighbors)

                    if abs(interpolatedData[x, y, z]-point) < tol: # value has converged sufficently, remove from listof nans/signs, will remain constant from now
                        nanIndexX, nanIndexY, nanIndexZ = np.delete(nanIndexX, idx, axis=0), np.delete(nanIndexY, idx, axis=0), np.delete(nanIndexZ, idx, axis=0)
                        signsX, signsY, signsZ = np.delete(signsX, idx, axis=0), np.delete(signsY, idx, axis=0), np.delete(signsZ, idx, axis=0)
                        d += 1
                    else: # replace with newly interpolated value and continue
                        interpolatedData[x, y, z] = point
            else:
                break
        return interpolatedData
