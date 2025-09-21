# coding:utf-8
import os
import os.path as osp
import time
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

from main import PROJECT_ROOT

EXP = [1 << i for i in range(10)]  # 使用位移运算简化EXP的计算


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            # transforms.Resize((512, 512)),
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


def cubeIndex(r, g, b, exp_n):
    return r + g * exp_n + b * exp_n * exp_n


def cubeIndex1(r, g, b, n):
    return r + g * EXP[n] + b * EXP[n] * EXP[n]


def mix(a, b, c):
    return a + (b - a) * (c - c.floor())


def getCube(i, cubepath):
    path = f'{cubepath}/luts_{i + 1}.cube'
    with open(path) as fd:
        lines = fd.readlines()
    rgbFloatCube = []
    cubeDataStart = False

    for l in lines:
        if cubeDataStart:
            rgbStr = l.strip().split()
            if len(rgbStr) == 3:
                rgbFloatCube.append((float(rgbStr[0]), float(rgbStr[1]), float(rgbStr[2])))
        if l.startswith("#LUT data points"):
            cubeDataStart = True
    return np.array(rgbFloatCube, dtype=np.float32)


def save_images(toColor2, img_files, output_dir):
    toColor2 = toColor2.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    for j in range(toColor2.shape[0]):
        img_result = Image.fromarray(toColor2[j])
        png_file = img_files[j]
        png_save_path = osp.join(output_dir, png_file)
        img_result.save(png_save_path, 'JPEG')


def Mapping(i, cubepath, times, n, dataloader):
    rgbFloatCube = getCube(i, cubepath)
    rgbFloatTensor = torch.tensor(rgbFloatCube, dtype=torch.float16, device='cuda')

    output_dir = PROJECT_ROOT + f'/main_result/result/New Test{times}/Test{i + 1}/Test_images'
    os.makedirs(output_dir, exist_ok=True)

    exp_n = torch.tensor(EXP[n], device='cuda', dtype=torch.float16)
    exp_val = exp_n - 1

    # scaler = torch.cuda.amp.GradScaler()

    # total_count = Counter()
    for img_tensors, img_files in dataloader:
        img_tensors = img_tensors.to('cuda', non_blocking=True, dtype=torch.float16)  # 使用非阻塞传输

        with torch.cuda.amp.autocast():
            # 对颜色通道进行变换
            r = img_tensors[:, 0, :, :] * exp_val  # [B, H, W]
            g = img_tensors[:, 1, :, :] * exp_val
            b = img_tensors[:, 2, :, :] * exp_val

            # 获取上限和下限
            rL = torch.floor(r).long()
            gL = torch.floor(g).long()
            bL = torch.floor(b).long()

            rH = torch.ceil(r).long()
            gH = torch.ceil(g).long()
            bH = torch.ceil(b).long()

            # 计算8个邻近点的LUT值
            indexLLL = cubeIndex1(rL, gL, bL, n)
            indexLLH = cubeIndex1(rL, gL, bH, n)
            indexLHL = cubeIndex1(rL, gH, bL, n)
            indexLHH = cubeIndex1(rL, gH, bH, n)
            indexHLL = cubeIndex1(rH, gL, bL, n)
            indexHLH = cubeIndex1(rH, gL, bH, n)
            indexHHL = cubeIndex1(rH, gH, bL, n)
            indexHHH = cubeIndex1(rH, gH, bH, n)

            # 获取颜色数据
            toColorLLL = rgbFloatTensor[indexLLL]
            toColorLLH = rgbFloatTensor[indexLLH]
            toColorLHL = rgbFloatTensor[indexLHL]
            toColorLHH = rgbFloatTensor[indexLHH]
            toColorHLL = rgbFloatTensor[indexHLL]
            toColorHLH = rgbFloatTensor[indexHLH]
            toColorHHL = rgbFloatTensor[indexHHL]
            toColorHHH = rgbFloatTensor[indexHHH]

            # 插值
            delta_r = r - rL.float()
            delta_g = g - gL.float()
            delta_b = b - bL.float()

            # 对蓝色通道进行插值
            toB_LL = mix(toColorLLL[:, :, :, 2], toColorLLH[:, :, :, 2], delta_b)  # 计算B方向的插值
            toB_LH = mix(toColorLHL[:, :, :, 2], toColorLHH[:, :, :, 2], delta_b)
            toB_HL = mix(toColorHLL[:, :, :, 2], toColorHLH[:, :, :, 2], delta_b)
            toB_HH = mix(toColorHHL[:, :, :, 2], toColorHHH[:, :, :, 2], delta_b)

            toB_L = mix(toB_LL, toB_LH, delta_g)  # 计算G方向的插值
            toB_H = mix(toB_HL, toB_HH, delta_g)
            toB = mix(toB_L, toB_H, delta_r)  # 计算R方向的插值

            # 对绿色通道进行插值
            toG_LL = mix(toColorLLL[:, :, :, 1], toColorLLH[:, :, :, 1], delta_b)  # 计算B方向的插值
            toG_LH = mix(toColorLHL[:, :, :, 1], toColorLHH[:, :, :, 1], delta_b)
            toG_HL = mix(toColorHLL[:, :, :, 1], toColorHLH[:, :, :, 1], delta_b)
            toG_HH = mix(toColorHHL[:, :, :, 1], toColorHHH[:, :, :, 1], delta_b)

            toG_L = mix(toG_LL, toG_LH, delta_g)  # 计算G方向的插值
            toG_H = mix(toG_HL, toG_HH, delta_g)
            toG = mix(toG_L, toG_H, delta_r)  # 计算R方向的插值

            # 对红色通道进行插值
            toR_LL = mix(toColorLLL[:, :, :, 0], toColorLLH[:, :, :, 0], delta_b)  # 计算B方向的插值
            toR_LH = mix(toColorLHL[:, :, :, 0], toColorLHH[:, :, :, 0], delta_b)
            toR_HL = mix(toColorHLL[:, :, :, 0], toColorHLH[:, :, :, 0], delta_b)
            toR_HH = mix(toColorHHL[:, :, :, 0], toColorHHH[:, :, :, 0], delta_b)

            toR_L = mix(toR_LL, toR_LH, delta_g)  # 计算G方向的插值
            toR_H = mix(toR_HL, toR_HH, delta_g)
            toR = mix(toR_L, toR_H, delta_r)  # 计算R方向的插值
            # 将颜色值转换回0-255并存储
            toColor2 = torch.stack((toR, toG, toB), dim=1) * 255  # [B, C, H, W]
            toColor2 = toColor2.clamp(0, 255).byte()

        save_images(toColor2, img_files, output_dir)
        # save_images(toColor2, img_files, output_dir)
        # 释放显存
        del img_tensors, toColor2
        # torch.cuda.empty_cache()
    # xxx=list(total_count.keys())
    # cube_file_path = r'C:\Users\22279\Documents\MMSeg-HCCNSGA-II\luts_102.cube'
    # output_file_path = r'C:\Users\22279\Documents\MMSeg-HCCNSGA-II\luts_102_change0.cube'
    # modify_cube_file_exclude_numbers(cube_file_path, xxx, output_file_path)
    # 创建结果目录
    # dir_result = f'C:/Users/22279/Documents/Labelme example_CC_alexnet/cell/New Test{times}/Test{i + 1}/results'
    # os.makedirs(dir_result, exist_ok=True)


if __name__ == '__main__':
    n = 4
    times = 1
    m = 1
    batch_size = 1  # 根据显存大小调整batch_size
    num_workers = 0  # 根据CPU核心数量和共享内存大小调整num_workers
    # 创建线程池
    thread_pool = ThreadPoolExecutor(max_workers=1)
    # 实例化数据集和数据加载器（只执行一次）
    #dataset = ImageDataset(r"C:\Users\22279\Documents\The NEW Data Evolving for Supervised Learning_experiment\image classification\Flower\EfficientNetLite0\First512-NoCut\Test75\all_train_img")
    dataset = ImageDataset(r"C:\Users\22279\Documents\The NEW Data Evolving for Supervised Learning_experiment\image classification\Bird\EfficientNetLite0\First\Test16\all_train_img")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    start = time.time()
    for i in tqdm(range(m), colour='blue', desc='正在映射并保存图片'):
        Mapping(
            i,
            cubepath='ourLuts',  # 替换为您的实际路径
            times=times,
            n=n,
            dataloader=dataloader,
            thread_pool=thread_pool
        )
    end = time.time()
    print(f"总耗时: {end - start} 秒")
    # 关闭线程池
    thread_pool.shutdown()
