#
# Adapted from https://github.com/EdwardFerdian/mri-tagging-strain
#

import numpy as np

def calculate_radial_strain(coords_batch, use_linear_strain=False):
    """
    Calculate radial strain for a batch of image sequences
    Args:
        coords_batch: [batch_size, n_frames, 2, 168] landmark coordinates
        use_linear_strain: whether to use linear strain formula
    Returns:
        rr_strains: [batch_size, n_frames, 1] radial strains
    """
    # point 0 is epi, point 6 is endo, for all 24 radials
    endo_batch = coords_batch[:, :, :, ::7]   # points 0
    epi_batch = coords_batch[:, :, :, 6::7]   # points 6
    
    diff = (epi_batch - endo_batch) ** 2
    summ = diff[:,:,0,:] + diff[:,:,1,:]  # x² + y²
    
    if use_linear_strain:
        summ = np.sqrt(summ)  # convert to L from L²
    
    summ_ed = summ[:,0,:]  # ED frame (frame 0)
    divv = summ / summ_ed[:, np.newaxis]
    
    rr_strains = (divv - 1) / (1 if use_linear_strain else 2)
    return np.mean(rr_strains, axis=2, keepdims=True)

def calculate_circumferential_strain(coords_batch, wall_index, use_linear_strain=False):
    """
    Calculate circumferential strain for a specific wall layer
    Args:
        coords_batch: [batch_size, n_frames, 2, 168] landmark coordinates
        wall_index: which wall layer to calculate (0-6)
        use_linear_strain: whether to use linear strain formula
    Returns:
        mid_cc: [batch_size, n_frames] circumferential strains
    """
    midwall_points = coords_batch[:, :, :, wall_index::7]  # [batch, frames, 2, 24]
    n_points = midwall_points.shape[3]
    
    # Calculate differences between consecutive points (circular)
    next_points = np.roll(midwall_points, -1, axis=3)
    cc_diff = (midwall_points - next_points) ** 2
    cc_sum = cc_diff[:,:,0,:] + cc_diff[:,:,1,:]  # x² + y²
    
    if use_linear_strain:
        cc_sum = np.sqrt(cc_sum)  # convert to L from L²
    
    cc_sum_ed = cc_sum[:,0,:]  # ED frame
    partial_cc = cc_sum / cc_sum_ed[:, np.newaxis]
    mid_cc = (partial_cc - 1) / (1 if use_linear_strain else 2)
    
    return np.mean(mid_cc, axis=2)  # average across all segments

def calculate_all_strains(uncropped_landmark_sequences):
    """
    Calculate all strain types (RR and CC, both linear and squared versions)
    Args:
        uncropped_landmark_sequences: [batch_size, n_frames, 2, 168] coordinates
    Returns:
        Tuple of (rr_strains, cc_strains, rr_linear_strains, cc_linear_strains)
    """
    rr_strains = calculate_radial_strain(uncropped_landmark_sequences, False)
    rr_linear_strains = calculate_radial_strain(uncropped_landmark_sequences, True)
    
    cc_strains = []
    cc_linear_strains = []
    for wall_index in range(7):
        cc_strains.append(calculate_circumferential_strain(
            uncropped_landmark_sequences, wall_index, False))
        cc_linear_strains.append(calculate_circumferential_strain(
            uncropped_landmark_sequences, wall_index, True))
    
    cc_strains = np.stack(cc_strains, axis=2)
    avg_cc = np.mean(cc_strains, axis=2, keepdims=True)
    cc_strains = np.concatenate([cc_strains, avg_cc], axis=2)
    
    cc_linear_strains = np.stack(cc_linear_strains, axis=2)
    avg_linear_cc = np.mean(cc_linear_strains, axis=2, keepdims=True)
    cc_linear_strains = np.concatenate([cc_linear_strains, avg_linear_cc], axis=2)
    
    return rr_strains, cc_strains, rr_linear_strains, cc_linear_strains