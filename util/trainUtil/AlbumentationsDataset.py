from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AlbumentationsDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        image = np.array(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
