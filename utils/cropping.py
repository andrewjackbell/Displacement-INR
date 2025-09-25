import numpy as np
import cv2

def crop_images(images, bb, target_shape=(128, 128)):
    """
    Crop the input image array to the bounding box specified by bb.
    The cropped image is then resized to the output_size using bicubic interpolation.
    Args:
        images: [n_frames, height, width] input image sequence
        bb: (x_min, y_min, x_max, y_max) bounding box coordinates
        target_shape: (height, width) desired output shape
    Returns:
        cropped_images: [n_frames, target_height, target_width] cropped and resized image sequence
    """
    
    n_frames, height, width = images.shape
    output_height, output_width = target_shape
    x1, y1, x2, y2 = np.floor(bb).astype(np.int32)
    cropped_images = np.empty((n_frames, output_height, output_width), dtype=images.dtype)
    for i in range(n_frames):
        
        image_section = images[i, y1:y2, x1:x2]
        resized_section = cv2.resize(image_section, target_shape, interpolation=cv2.INTER_CUBIC)
        cropped_images[i] = np.clip(resized_section, 0, 255)

    return cropped_images

def move_points(points, bb, cropped_size=(128, 128)):
    '''
    The points are moved to the new coordinate system defined by the bounding box and cropped size.
    Args:
        points: [n_frames, n_points, 2] landmark coordinates
        bb: (x_min, y_min, x_max, y_max) bounding box coordinates
        cropped_size: (h, w) desired output shape
    Returns:
        moved_points: [n_frames, n_points, 2] moved landmark coordinates
    '''

    x1, y1, x2, y2 = np.floor(bb).astype(np.int32)
    output_height, output_width = cropped_size

    moved_points = points - np.array([x1, y1], dtype=np.float32) # translate to (0,0)
    moved_points = moved_points / np.array([x2 - x1, y2 - y1], dtype=np.float32) # scale to 1
    moved_points = moved_points * np.array([output_height, output_width], dtype=np.float32) # scale to new size

    return moved_points

def move_points_back(moved_points, bb, original_size, cropped_size=(128, 128)):
    """
    Move points back to the original coordinate system defined by the bounding box and original image size.
    Args:
        moved_points: [n_frames, n_points, 2] moved landmark coordinates
        bb: (x_min, y_min, x_max, y_max) bounding box coordinates
        original_size: (h, w) original image size
        cropped_size: (h, w) size of the cropped image
    Returns:
        points: [n_frames, n_points, 2] landmark coordinates in the original image
    """
    # Convert to numpy arrays if needed
    bb = np.asarray(bb)
    moved_points = np.asarray(moved_points)
    
    # Get dimensions
    x1, y1, x2, y2 = bb.astype(np.int32)
    crop_w, crop_h = x2 - x1, y2 - y1
    out_h, out_w = cropped_size
    
    # Step 1: Reverse the final scaling to output size
    points = moved_points / np.array([out_h, out_w], dtype=np.float32)
    
    # Step 2: Reverse the resize operation (scale back to crop size)
    points = points * np.array([crop_w, crop_h], dtype=np.float32)
    
    # Step 3: Reverse the crop translation
    points = points + np.array([x1, y1], dtype=np.float32)
    
    # Clamp to image bounds
    orig_h, orig_w = original_size
    points[..., 0] = np.clip(points[..., 0], 0, orig_h - 1)
    points[..., 1] = np.clip(points[..., 1], 0, orig_w - 1)
    
    return points

def scale_displacements_back(displacements, original_size, cropped_size=(128, 128)):
    """
    Scale displacements to the new coordinate system defined by the bounding box.
    
    displacements: (N, F, P, 2) tensor of displacements
    original_size: (H, W) tuple for the original image size
    cropped_size: (H, W) tuple for the output size
    """
    
    orig_h, orig_w = original_size
    crop_h, crop_w = cropped_size
    
    # Scale displacements back to the original size
    scaled_displacements = displacements * np.array([orig_w / crop_w, orig_h / crop_h], dtype=np.float32)
    
    return scaled_displacements

def move_points_back_many(moved_points, bbs, original_size, cropped_size=(128, 128)):
    """
    Moved points and bbs now have shape [N, F, P, 2] and [N, 4]
    """
    moved_points = np.stack([move_points_back(moved_points[i], bbs[i], original_size, cropped_size) for i in range(len(moved_points))])
    return moved_points

def move_points_many(points, bbs, cropped_size=(128, 128)):
    """
    Move multiple sets of points to the new coordinate system defined by the bounding boxes.
    
    points: (N, F, P, 2) tensor of points
    bbs: (N, 4) tensor of bounding boxes
    cropped_size: (H, W) tuple for the output size
    """
    
    moved_points = []
    for i in range(points.shape[0]):
        moved_pts = move_points(points[i], bbs[i], cropped_size)
        moved_points.append(moved_pts)
    
    return np.stack(moved_points)
