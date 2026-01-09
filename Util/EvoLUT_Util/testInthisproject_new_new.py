import json
import os
import time
from torchvision import transforms
from model import LeNet5
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import multiprocessing as mp
from PIL import Image
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_path

def load_class_mapping(json_name):
    with open(json_name, 'r') as file:
        return json.load(file)

def process_batch(batch_imgs, batch_paths, model, class_mapping):
    predictions = torch.softmax(model(batch_imgs), dim=1)
    total_loss = 0
    for j, img_path in enumerate(batch_paths):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        main_part = base_name.split('_')[0]
        index = next((key for key, value in class_mapping.items() if value == main_part), None)
        if index is not None:
            total_loss += predictions[j, int(index)].item()
    return total_loss

def calculate_folder_loss(args):
    folder_path, json_name, scripted_model_path, batch_size, device, data_transform, class_mapping = args
    model = torch.jit.load(scripted_model_path, map_location=device)
    model.eval()

    img_filenames = os.listdir(folder_path)
    img_paths = [os.path.join(folder_path, img_filename) for img_filename in img_filenames]
    dataset = ImageDataset(img_paths, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    total_loss = 0
    z = len(img_paths)
    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            total_loss += process_batch(batch_imgs, batch_paths, model, class_mapping)

    average_loss = total_loss / z
    return 1 - average_loss

if __name__ == '__main__':
    mp.set_start_method('spawn')
    start = time.time()

    weights_path = "/mnt/dir/ImageClassificationHCCGA/result/Fruit/LeNet5/result_base_3/model/epoch280_acc0.724500.pth"
    num_classes = 10
    json_name = '/home/user004/project/oybd/ImageClassificationHCCGA/class_indices_fruit.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    num_workers = 2

    # 主进程加载模型和数据转换器
    model = LeNet5(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    scripted_model_path = "/home/user004/project/oybd/ImageClassificationHCCGA/scripted_model.pt"
    # torch.jit.script(model).save(scripted_model_path)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    class_mapping = load_class_mapping(json_name)

    times = 1
    m = 100
    folder_paths = [f"/mnt/dir/Labelme example_CC_alexnet/cell/New Test{times}/Test{i + 1}/Test_images/" for i in range(m)]
    args = [(folder, json_name, scripted_model_path, batch_size, device, data_transform, class_mapping) for folder in folder_paths]

    with Pool(processes=num_workers) as pool:
        losses = pool.map(calculate_folder_loss, args)

    for i, loss in enumerate(losses):
        print(f"Folder {i + 1}: Loss = {loss}")
    end = time.time()
    print(f"Total time: {end - start} seconds")