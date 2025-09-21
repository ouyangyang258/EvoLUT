import os
import os.path as osp
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter

EXP = [1 << i for i in range(10)]  # 使用位移运算简化EXP的计算


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
    """计算LUT索引"""
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
            # 对颜色通道进行变换
            r = img_tensors[:, 0, :, :] * exp_val  # [B, H, W]
            g = img_tensors[:, 1, :, :] * exp_val
            b = img_tensors[:, 2, :, :] * exp_val  # 蓝色通道，计算但不在此阶段参与插值
            # 获取每个像素点的LUT索引（直接返回索引，而不计算插值）

            # 计算floor和ceil值
            rL = torch.floor(r).long()
            gL = torch.floor(g).long()
            bL = torch.floor(b).long()

            rH = torch.ceil(r).long()
            gH = torch.ceil(g).long()
            bH = torch.ceil(b).long()

            # 计算4个邻近点的LUT索引
            # 确保rL, gL, bL 是单个整数 (使用 `.item()` 将其转换为 Python 标量)
            index00 = cubeIndex1(rL, gL, bL, n)
            index01 = cubeIndex1(rL, gL, bH, n)
            index10 = cubeIndex1(rL, gH, bL, n)
            index11 = cubeIndex1(rL, gH, bH, n)
            index20 = cubeIndex1(rH, gL, bL, n)
            index21 = cubeIndex1(rH, gL, bH, n)
            index30 = cubeIndex1(rH, gH, bL, n)
            index31 = cubeIndex1(rH, gH, bH, n)

            indexH_list = [index00, index01, index10, index11, index20, index21, index30, index31]

            # 遍历每个 indexH，将每个 indexH 的统计结果累加到 total_count 中
            for indexH in indexH_list:
                # 展平为一维张量
                flattened = indexH.flatten()

                # 使用 Counter 统计每个数字的出现次数
                count_map = Counter(flattened.tolist())

                # 将当前 count_map 累加到 total_count 中
                total_count.update(count_map)

        # 4. 使用 most_common() 按出现次数排序并输出
    sorted_count = total_count.most_common()  # 按从大到小排序

    # 5. 输出结果.
    for number, count in sorted_count:
        print(f"LUT中的行数: {number}, 出现次数: {count}")

    # 返回过滤后的索引
    total_count_new_1 = Counter({key: value for key, value in total_count.items() if value >= 0})
    remove_numbers = {}  # 这里可以定义你希望移除的特定索引
    total_count_new_2 = Counter({key: value for key, value in total_count_new_1.items() if key not in remove_numbers})

    return list(total_count_new_2.keys()), len(total_count_new_2.keys()), len(total_count_new_1.keys()), len(
        total_count.keys())


