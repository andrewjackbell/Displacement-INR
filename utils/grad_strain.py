import numpy as np

def xy_strain_to_rc(reference_points, xy_strains):
    """
    Convert strain tensors from Cartesian (x,y) to Radial-Circumferential (r,c) basis using a simple centroid vector method.
    Args:
        reference_points: [N, 2] landmark coordinates in (x,y) format
        xy_strains: [T, N, 2, 2] strain tensors in (x,y) basis
    Returns:
        strains_rc: [T, N, 2, 2] strain tensors in (r,c) basis
    """

    centroid = np.mean(reference_points, axis=0)

    radial_vectors = reference_points - centroid
    radial_bases = radial_vectors / np.linalg.norm(radial_vectors, axis=1, keepdims=True)

    circumferential_bases = np.zeros_like(radial_bases)
    circumferential_bases[:, 0] = -radial_bases[:, 1]  # -y
    circumferential_bases[:, 1] = radial_bases[:, 0]   # x

    R = np.stack((radial_bases, circumferential_bases), axis=-1)
    strains_rc = np.swapaxes(R, -1, -2) @ xy_strains @ R

    return strains_rc


def rc_strain_maps(reference_points, xy_strains, seq_shape, point_format='xy'):
    """
    Create radial and circumferential strain maps from point-wise strain tensors.
    Args:
        reference_points: [N, 2] landmark coordinates
        xy_strains: [T, N, 2, 2] strain tensors in (x,y) basis
        seq_shape: shape of the output strain maps (T, H, W)
        point_format: 'xy' if reference_points are in (x,y), or 'ij' if in (row, col) format
    Returns:
        Tuple of (rr_image, cc_image) strain maps of shape [T, H, W]
    """

    strains_rc = xy_strain_to_rc(reference_points, xy_strains)

    # convert the point strain into an image by filling in a 'strain mask' in image space
    cc_image = np.full(seq_shape, np.nan, dtype=np.float32)
    rr_image = np.full(seq_shape, np.nan, dtype=np.float32)

    for i in range(strains_rc.shape[0]):
        for j in range(strains_rc.shape[1]):
            x, y = reference_points[j]
            if point_format == 'xy':
                y, x = x, y # Convert to (row, col) for image indexing
            
            cc_image[i, x, y] = strains_rc[i, j, 1, 1]  # εcc
            rr_image[i, x, y] = strains_rc[i, j, 0, 0]


    return rr_image, cc_image