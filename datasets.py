import pickle
import torch
import numpy as np
import random
import torchvision
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

# --------------------- IO --------------------- #

def unpickle(f):
    with open(f, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# --------------------- Cropping helpers --------------------- #

def new_random_crop(labels, crop_size):
    crop_map = create_crop_map(labels, crop_size)
    good_indices = find_good_samples(labels, crop_map, crop_size)
    return crop_map, good_indices

def create_crop_map(labels, crop_size):
    """Create one random (x,y) top-left crop per sample."""
    height, width = labels.shape[1], labels.shape[2]
    crop_map = []
    for _ in range(len(labels)):
        x_shift = random.randint(0, height - crop_size)
        y_shift = random.randint(0, width - crop_size)
        crop_map.append((x_shift, y_shift))
    return np.array(crop_map, dtype=np.int32)

def get_cropped_sample(index, crop_map, crop_size, data, labels):
    x_shift, y_shift = crop_map[index]
    cropped_features = data[index, :, x_shift:x_shift+crop_size, y_shift:y_shift+crop_size]
    cropped_label   = labels[index,    x_shift:x_shift+crop_size, y_shift:y_shift+crop_size]
    return cropped_features, cropped_label

def find_good_samples(labels, crop_map, crop_size):
    """
    Keep crops whose label patch contains no invalid (-1) pixels.
    """
    good_indices = []
    for i in range(len(labels)):
        x_shift, y_shift = crop_map[i]
        patch = labels[i, x_shift:x_shift+crop_size, y_shift:y_shift+crop_size]
        if np.all(patch != -1):
            good_indices.append(i)
    return np.array(good_indices, dtype=np.int32)

# --------------------- Base Dataset with re-seeding --------------------- #

class _BaseWildfireDataset(torch.utils.data.Dataset):
    """
    Shared utilities for reseeding crops (new random crops each epoch).
    """
    def reseed_crops(self, seed=None):
        """
        Call this at the start of each epoch, e.g.:
            dataset.reseed_crops(epoch)
        """
        if seed is not None:
            random.seed(int(seed))
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)
        # If subclass has oversampling lists, refresh them
        if hasattr(self, "_refresh_oversampling"):
            self._refresh_oversampling()

# --------------------- Plain random-crop dataset --------------------- #

class WildfireDataset(_BaseWildfireDataset):
    def __init__(self, data_filename, labels_filename, features=None, crop_size=64, seed=1):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = int(crop_size)

        random.seed(int(seed))
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)

        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing WildfireDataset")
        
    def __len__(self):
        return int(len(self.good_indices))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(self.good_indices[index])
        
        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        x = torch.from_numpy(cropped_features).float().contiguous()          # [C,H,W]
        y = torch.from_numpy(np.expand_dims(cropped_label, axis=0)).float()  # [1,H,W]
        return x, y

# --------------------- Rotated random-crop dataset --------------------- #

class RotatedWildfireDataset(_BaseWildfireDataset):
    # Note: wind direction features may not be rotation-invariant.
    def __init__(self, data_filename, labels_filename, features=None, crop_size=64, seed=1):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = int(crop_size)

        random.seed(int(seed))
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)
        
        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing RotatedWildfireDataset")

        self._rotations = [0, 90, 180, 270]

    def __len__(self):
        return int(len(self.good_indices) * len(self._rotations))

    def __getitem__(self, index):
        g = len(self.good_indices)
        rotation_index = index // g
        base_index = index % g

        index = int(self.good_indices[base_index])
        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        x = torch.from_numpy(cropped_features).float().contiguous()           # [C,H,W]
        y = torch.from_numpy(np.expand_dims(cropped_label, axis=0)).float()   # [1,H,W]

        angle = self._rotations[rotation_index]
        # Explicit interpolation: bilinear for features, nearest for labels
        x = rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR)
        y = rotate(y, angle=angle, interpolation=InterpolationMode.NEAREST)
        return x, y

# --------------------- Oversampled (class-balanced) dataset --------------------- #

class OversampledWildfireDataset(_BaseWildfireDataset):
    """
    Oversamples crops that have at least `fire_threshold` fraction of fire pixels.
    Uses `pos_prob` to control how often a batch draws from fire-rich crops.
    """
    def __init__(self,
                 data_filename,
                 labels_filename,
                 features=None,
                 crop_size=64,
                 fire_threshold=0.02,  # >= 2% fire pixels in a crop
                 pos_prob=0.5,         # probability to draw a positive (fire-rich) crop
                 seed=1):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = int(crop_size)
        self.fire_threshold = float(fire_threshold)
        self.pos_prob = float(pos_prob)

        random.seed(int(seed))
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)

        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None

        self._refresh_oversampling()  # builds oversample_indices
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing OversampledWildfireDataset")

    def _refresh_oversampling(self):
        self.oversample_indices = self._find_samples_for_oversampling()

    def __len__(self):
        # Return base size; sampling logic controls class mix
        return int(len(self.good_indices))

    def __getitem__(self, idx):
        draw_pos = (random.random() < self.pos_prob) and (len(self.oversample_indices) > 0)
        if draw_pos:
            index = self._get_random_oversample_index()
        else:
            index = int(self.good_indices[idx % len(self.good_indices)])

        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        x = torch.from_numpy(cropped_features).float().contiguous()           # [C,H,W]
        y = torch.from_numpy(np.expand_dims(cropped_label, axis=0)).float()   # [1,H,W]
        return x, y

    def _find_samples_for_oversampling(self):
        oversample_indices = []
        area = self.crop_size * self.crop_size  # FIX: 64x64 = 4096 (not 1024)
        for index in self.good_indices:
            x_shift, y_shift = self.crop_map[index]
            lab = self.labels[index, x_shift:x_shift+self.crop_size, y_shift:y_shift+self.crop_size]
            # count fire pixels (==1). invalid (-1) have been excluded by good_indices
            fire_pixels = int((lab == 1).sum())
            if (fire_pixels / area) >= self.fire_threshold:
                oversample_indices.append(int(index))
        return oversample_indices

    def _get_random_oversample_index(self):
        # Guard for empty list (shouldn’t be called when empty due to check)
        if not self.oversample_indices:
            return int(random.choice(self.good_indices))
        return int(random.choice(self.oversample_indices))