import napari
import numpy as np


def reshape_points_for_napari(points, flip=True):
    """
    Convert a (T, N, D) array into a (T*N, D+1) array for Napari, adding time as the first coordinate.

    Parameters:
        points (numpy.ndarray): Input array of shape (T, N, D), where
                                T = number of time frames,
                                N = number of points per frame,
                                D = spatial dimensions (x, y, ...).
    
    Returns:
        numpy.ndarray: Reshaped array of shape (T*N, D+1), where columns are (t, spatial dims...).
    """
    if flip:
        points = np.flip(points, axis=-1)  # Flip the last dimension
    T, N, D = points.shape  # Extract dimensions
    t_coords = np.repeat(np.arange(T)[:, np.newaxis], N, axis=1)  # Create time column
    points_reshaped = np.column_stack((t_coords.flatten(), points.reshape(-1, D)))
    return points_reshaped


def reshape_points_4d(points, flip=True):
    """
    Points is in format (Z, T, N, D)
    Reshape to (Z*T*N, D+2)
    """

    if flip:
        points = np.flip(points, axis=-1)

    Z, T, N, D = points.shape  # Extract dimensions
    z_indexes = np.arange(Z).reshape(Z, 1, 1, 1)
    t_indexes = np.arange(T).reshape(1, T, 1, 1)

    z_broadcast = np.broadcast_to(z_indexes, (Z, T, N, 1))
    t_broadcast = np.broadcast_to(t_indexes, (Z, T, N, 1))

    full_coords = np.concatenate((z_broadcast, t_broadcast, points), axis=-1)
    full_coords = full_coords.reshape(-1, D+2)
    return full_coords
