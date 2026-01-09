import os
import os.path as osp
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter

EXP = [1 << i for i in range(10)]


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []
        img_files = sorted(os.listdir(image_dir))
        for img_file in img_files:
            img_path = osp.join(self.image_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            self.data.append((img_tensor, img_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def cubeIndex1(r, g, b, n):
    return r + g * EXP[n] + b * EXP[n] * EXP[n]


def GetRGBpix(n):
    dataset = ImageDataset(r"C:\Users\22279\Documents\EvoLUT\data_images")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    exp_n = torch.tensor(EXP[n], device='cuda', dtype=torch.float16)
    exp_val = exp_n - 1
    total_count = Counter()

    for img_tensors, img_files in dataloader:
        img_tensors = img_tensors.to('cuda', non_blocking=True, dtype=torch.float16)

        with torch.cuda.amp.autocast():
            # Perform a transformation on color channels
            r = img_tensors[:, 0, :, :] * exp_val  # [B, H, W]
            g = img_tensors[:, 1, :, :] * exp_val
            b = img_tensors[:, 2, :, :] * exp_val
            # Retrieve the LUT index for each pixel (directly return the index without performing interpolation)

            # Calculate floor and ceiling values
            rL = torch.floor(r).long()
            gL = torch.floor(g).long()
            bL = torch.floor(b).long()

            rH = torch.ceil(r).long()
            gH = torch.ceil(g).long()
            bH = torch.ceil(b).long()

            index00 = cubeIndex1(rL, gL, bL, n)
            index01 = cubeIndex1(rL, gL, bH, n)
            index10 = cubeIndex1(rL, gH, bL, n)
            index11 = cubeIndex1(rL, gH, bH, n)
            index20 = cubeIndex1(rH, gL, bL, n)
            index21 = cubeIndex1(rH, gL, bH, n)
            index30 = cubeIndex1(rH, gH, bL, n)
            index31 = cubeIndex1(rH, gH, bH, n)

            indexH_list = [index00, index01, index10, index11, index20, index21, index30, index31]


            for indexH in indexH_list:
                flattened = indexH.flatten()

                count_map = Counter(flattened.tolist())

                total_count.update(count_map)

        # 4. Sort by frequency of occurrence and output
    sorted_count = total_count.most_common()

    # 5. Output result
    for number, count in sorted_count:
        print(f"Number of rows in LUT: {number}, Occurrences: {count}")

    # 返回过滤后的索引
    total_count_new_1 = Counter({key: value for key, value in total_count.items() if value >= 0})
    remove_numbers = {}  # You can define the specific index you want to remove here
    total_count_new_2 = Counter({key: value for key, value in total_count_new_1.items() if key not in remove_numbers})

    return list(total_count_new_2.keys()), len(total_count_new_2.keys()), len(total_count_new_1.keys()), len(
        total_count.keys())


