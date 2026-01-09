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

EXP = [1 << i for i in range(10)]


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []
        img_files = sorted(os.listdir(image_dir))
        for img_file in tqdm(img_files, desc='Preloading images'):
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

    for img_tensors, img_files in dataloader:
        img_tensors = img_tensors.to('cuda', non_blocking=True, dtype=torch.float16)

        with torch.cuda.amp.autocast():
            r = img_tensors[:, 0, :, :] * exp_val  # [B, H, W]
            g = img_tensors[:, 1, :, :] * exp_val
            b = img_tensors[:, 2, :, :] * exp_val

            rL = torch.floor(r).long()
            gL = torch.floor(g).long()
            bL = torch.floor(b).long()

            rH = torch.ceil(r).long()
            gH = torch.ceil(g).long()
            bH = torch.ceil(b).long()

            # Calculate the LUT values for 8 neighboring points
            indexLLL = cubeIndex1(rL, gL, bL, n)
            indexLLH = cubeIndex1(rL, gL, bH, n)
            indexLHL = cubeIndex1(rL, gH, bL, n)
            indexLHH = cubeIndex1(rL, gH, bH, n)
            indexHLL = cubeIndex1(rH, gL, bL, n)
            indexHLH = cubeIndex1(rH, gL, bH, n)
            indexHHL = cubeIndex1(rH, gH, bL, n)
            indexHHH = cubeIndex1(rH, gH, bH, n)


            toColorLLL = rgbFloatTensor[indexLLL]
            toColorLLH = rgbFloatTensor[indexLLH]
            toColorLHL = rgbFloatTensor[indexLHL]
            toColorLHH = rgbFloatTensor[indexLHH]
            toColorHLL = rgbFloatTensor[indexHLL]
            toColorHLH = rgbFloatTensor[indexHLH]
            toColorHHL = rgbFloatTensor[indexHHL]
            toColorHHH = rgbFloatTensor[indexHHH]

            # Interpolation
            delta_r = r - rL.float()
            delta_g = g - gL.float()
            delta_b = b - bL.float()

            # Interpolate the blue channel
            toB_LL = mix(toColorLLL[:, :, :, 2], toColorLLH[:, :, :, 2], delta_b)  # Calculate the interpolation in the B direction
            toB_LH = mix(toColorLHL[:, :, :, 2], toColorLHH[:, :, :, 2], delta_b)
            toB_HL = mix(toColorHLL[:, :, :, 2], toColorHLH[:, :, :, 2], delta_b)
            toB_HH = mix(toColorHHL[:, :, :, 2], toColorHHH[:, :, :, 2], delta_b)

            toB_L = mix(toB_LL, toB_LH, delta_g)  # Calculate the interpolation in the G direction
            toB_H = mix(toB_HL, toB_HH, delta_g)
            toB = mix(toB_L, toB_H, delta_r)  # Calculate the interpolation in the R direction

            # Perform interpolation on the green channel
            toG_LL = mix(toColorLLL[:, :, :, 1], toColorLLH[:, :, :, 1], delta_b)
            toG_LH = mix(toColorLHL[:, :, :, 1], toColorLHH[:, :, :, 1], delta_b)
            toG_HL = mix(toColorHLL[:, :, :, 1], toColorHLH[:, :, :, 1], delta_b)
            toG_HH = mix(toColorHHL[:, :, :, 1], toColorHHH[:, :, :, 1], delta_b)

            toG_L = mix(toG_LL, toG_LH, delta_g)
            toG_H = mix(toG_HL, toG_HH, delta_g)
            toG = mix(toG_L, toG_H, delta_r)

            # Interpolate the red channel
            toR_LL = mix(toColorLLL[:, :, :, 0], toColorLLH[:, :, :, 0], delta_b)
            toR_LH = mix(toColorLHL[:, :, :, 0], toColorLHH[:, :, :, 0], delta_b)
            toR_HL = mix(toColorHLL[:, :, :, 0], toColorHLH[:, :, :, 0], delta_b)
            toR_HH = mix(toColorHHL[:, :, :, 0], toColorHHH[:, :, :, 0], delta_b)

            toR_L = mix(toR_LL, toR_LH, delta_g)
            toR_H = mix(toR_HL, toR_HH, delta_g)
            toR = mix(toR_L, toR_H, delta_r)

            # Convert the color values back to the 0-255 range and store them.
            toColor2 = torch.stack((toR, toG, toB), dim=1) * 255  # [B, C, H, W]
            toColor2 = toColor2.clamp(0, 255).byte()

        save_images(toColor2, img_files, output_dir)
        del img_tensors, toColor2


if __name__ == '__main__':
    n = 4
    times = 1
    m = 1
    batch_size = 1
    num_workers = 0
    thread_pool = ThreadPoolExecutor(max_workers=1)
    dataset = ImageDataset(r"C:\Users\22279\Documents\The NEW Data Evolving for Supervised Learning_experiment\image classification\Bird\EfficientNetLite0\First\Test16\all_train_img")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    start = time.time()
    for i in tqdm(range(m), colour='blue', desc='Mapping and saving the image'):
        Mapping(
            i,
            cubepath='ourLuts',
            times=times,
            n=n,
            dataloader=dataloader,
            thread_pool=thread_pool
        )
    end = time.time()
    print(f"Total time: {end - start} seconds")
    thread_pool.shutdown()
