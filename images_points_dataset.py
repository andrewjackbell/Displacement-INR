import torch
import os
import numpy as np
from utils.points import normalise_coords
from utils.cropping import crop_images, move_points

class IPDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading image and point sequences from the specified directory.
    This dataset assumes that each sample is stored in a .npz file containing 'images' and 'points' arrays.
    """

    def __init__(self, dataset_name, crop_size=None, normalise=True):

        dataset_dir = './data'
        self.data_path = os.path.join(dataset_dir, dataset_name)
        self.samples = sorted([os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".npz")])
        self.crop_size = crop_size
        self.normalise = normalise

    def __len__(self):
        return len(self.samples)

    
    def _preprocess(self, image_seq, point_seq, bounding_box):

        if self.crop_size:
            image_seq = crop_images(image_seq, bounding_box, self.crop_size)
            point_seq = move_points(point_seq, bounding_box, self.crop_size)
        if self.normalise:
            image_seq = image_seq / 255
            point_seq = normalise_coords(point_seq, image_seq.shape[-1])
            
        return image_seq, point_seq


    def __getitem__(self, idx):
    
        data = np.load(self.samples[idx])
        images = data['images'].astype(np.float32) # (T, H, W) 
        points = data['points'].astype(np.float32) # (T, N, 2)
        es_frame_n = data['es_frame_number'].item() # int
        slice_name = data['slice_name'].item() # str
        px_spacing = data['px_spacing'].item() # float
        bounding_box = data['bounding_box'].astype(np.float32) # (x_min, y_min, x_max, y_max)

        case_name = os.path.basename(self.samples[idx]).split('.')[0] # str (filename without extension)  
        time_coords = normalise_coords(np.arange(images.shape[0], dtype=np.float32), images.shape[0]-1)  # (T,)
        original_shape = np.array(images.shape[1:]) # (H, W)

        images, points = self._preprocess(images, points, bounding_box)

        return {'images': images,
                'points': points,
                'time_coords': time_coords,
                'original_shape': original_shape,
                'bounding_box': bounding_box,
                'es_frame_n': es_frame_n,
                'px_spacing': px_spacing,
                'case': case_name,
                'slice_name': slice_name}
    
    def get_item_by_case(self, case_name):
        """Retrieve the item by case name."""
        for idx, sample in enumerate(self.samples):
            if os.path.basename(sample).startswith(case_name):
                return self.__getitem__(idx)
        raise ValueError(f"Case {case_name} not found in dataset.")

if __name__ == "__main__":

    crop_size = None  # Optional, set to None for no cropping
    normalise = False
    dataset = IPDataset(dataset_name='example',
                             crop_size=crop_size, 
                             normalise=normalise)
    
    print(f"Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    import matplotlib.pyplot as plt
    plt.imshow(sample['images'][0], cmap='gray')
    plt.scatter(sample['points'][0,:,0], sample['points'][0,:,1], c='r', s=5)
    plt.title(f"Case: {sample['case']}, Slice: {sample['slice_name']}")
    plt.show()
