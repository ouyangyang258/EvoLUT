# encoding=utf-8
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os.path as osp


class ImageDataset:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []
        img_files = sorted(os.listdir(image_dir))
        for img_file in tqdm(img_files, desc='预加载图像'):
            img_path = osp.join(self.image_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            self.data.append((img_tensor, img_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
