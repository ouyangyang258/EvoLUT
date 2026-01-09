import json
import math
import os
import copy
import random
import shutil
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Util.constants import PROJECT_ROOT
from Util.EvoLUT_Util.segmentAndmerge import segmentHilbert, mergeHilber
from Util.EvoLUT_Util.copydir import copydir
from multiprocessing import Pool
from torchvision import transforms
from model import EfficientNetLite0, MobileNetV2
from Util.EvoLUT_Util.testInthisproject_new_new import load_class_mapping, calculate_folder_loss
from Util.EvoLUT_Util import Util, Select, Operator, mapping_plus_tensor_a100_trilinear_interpolation, \
    getUsedRGBpix_trilinear, get_new_luts, get_hilbert_new, get_lutsAndcube, hilbert_get_cube, Initial_def
from Util.EvoLUT_Util.Util import deletTxt, draw, save_to_txt
from Util.Train_Util import util

with open("main_config.json", "r") as f:
    config = json.load(f)
data_transform = transforms.Compose([
    transforms.Resize(config["input_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["normalize_mean"], std=config["normalize_std"])
])
# global variable
num = 0
json_name = config["json_name"]
list_index = []
list_answer = []
scripted_model_path = "scripted_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = config["weights_path"]
num_classes = 6
keep_count = 3
# Set the number of images to z
z = util.getImagesNumber()
# Set the capacity of the current 3D LUTs to v × v × v
v = 4
# v = 2^n
n = int(math.log(v, 2))
# Set the number of collaborative optimization groups to c
c = 4
# Set initial population size m
m = 100
# Set optimization frequency
epoch = 5
# Whether to use pruning operation
useCut = False


def initialization(m, n, c, num, times, usedPixList):
    # ------------------------------1. Generate initial population----------------------------------
    for i in tqdm(range(m), colour='blue', desc='Randomly generating 3D LUTs files and Cube files'):
        luts_name = f"ourLuts/luts_{i + 1}.cube"
        cube_name = f"ourCUBES/cube_{i + 1}.pickle"
        get_lutsAndcube.getLutsAndCube(LutsName=luts_name, CubeName=cube_name, n=n, usedPixList=usedPixList)

        dir_dst_luts = PROJECT_ROOT + f'/main_result/result/New Test1/Test{i + 1}'
        os.makedirs(dir_dst_luts, exist_ok=True)
        shutil.copy(luts_name, dir_dst_luts)
    # ------------------------------ 2. Map and save images ----------------------------------------
    for i in tqdm(range(m), colour='blue', desc='Map and save images'):
        mapping_plus_tensor_a100_trilinear_interpolation.Mapping(i, cubepath="ourLuts", times=1, n=n,
                                                                 dataloader=dataloader)
    # ------------------------------ 3. Generate Hilbert curve encoding ----------------------------
    solution = [[] for _ in range(m)]
    for j in tqdm(range(m), colour='blue', desc='Generate Hilbert curve encoding'):
        cube_name = f"ourCUBES/cube_{j + 1}.pickle"
        with open(f"ourHILBERTS/hilbert_{j + 1}.txt", "w") as f:
            get_hilbert_new.getHilbert(n=n, CubeName=cube_name, file_handle=f, solution=solution, jj=j)
        with open(f"ourLitterHILBERTS/hilbert_{j + 1}.txt", "w") as f:
            get_hilbert_new.getLitterHilbert(n=n, CubeName=cube_name, file_handle=f, solution=solution, jj=j)
    # ------------------------------ 4. Calculate fitness value ------------------------------------
    file_path = f"accuracy/accuracy{times}.txt"
    class_mapping = load_class_mapping(json_name)
    folder_paths = [
        os.path.join(PROJECT_ROOT, f"main_result/result/New Test{times}/Test{i + 1}/Test_images/")
        for i in range(m)
    ]
    args = [(folder, json_name, scripted_model_path, 20, device, data_transform, class_mapping)
            for folder in folder_paths]

    with Pool(processes=2) as pool:
        losses = pool.map(calculate_folder_loss, args)

    Min = min(losses)
    with open(file_path, "w") as f:
        for i, loss in enumerate(losses):
            f.write(str(loss) + "\n")
            print(f"{i + 1}: Loss = {loss}")

    list_answer.append(Min)
    num += 1
    list_index.append(num)
    print(f"The {times} round of optimization has been completed!")
    # ------------------------------ 5. Select the optimal population ----------------------------------------
    best_pop = get_best_population(times)
    copydir("ourHILBERTS/", "BestPopulation/hilbert/", best_pop, "BestPopulation/hilbert/thebestpopulathon.txt")
    copydir("ourLitterHILBERTS/", "BestPopulation/hilbert/", best_pop,
            "BestPopulation/hilbert/thebestLittlepopulathon.txt")
    copydir("ourLuts/", "BestPopulation/lut/", best_pop, "BestPopulation/lut/thebestpopulathon.cube")
    copydir("ourCUBES/", "BestPopulation/cube/", best_pop, "BestPopulation/cube/thebestpopulathon.pickle")
    # ------------------------------ 6. tournament selection  ----------------------------------------
    numbers = [float(line.strip()) for line in open(file_path)]
    selected = Select.tournament_selection(numbers, int(m * 0.6), "Util/EvoLUT_Util/Select_list.txt")
    New_solution = [solution[i - 1] for i in selected]

    # Copy and rename Hilbert files
    os.makedirs("ourLitterHILBERTS-New", exist_ok=True)
    for new_idx, ori_idx in enumerate(selected, start=1):
        src = os.path.join("ourLitterHILBERTS", f"hilbert_{ori_idx}.txt")
        dst = os.path.join("ourLitterHILBERTS-New", f"hilbert_{new_idx}.txt")
        shutil.copy(src, dst)
    # ------------------------------ 7. Grouping Hilbert Curve ----------------------------------------
    segmentHilbert(c, "BestPopulation/hilbert/thebestLittlepopulathon.txt", "BestPopulation/segmentLittleHilbert")
    if os.path.exists('BestPopulation/segmentLittleHilbert_change'):
        shutil.rmtree('BestPopulation/segmentLittleHilbert_change')
    shutil.copytree('BestPopulation/segmentLittleHilbert', 'BestPopulation/segmentLittleHilbert_change')

    for j in tqdm(range(m), colour="blue", desc="Grouping Hilbert Curve"):
        segmentHilbert(c, f"ourLitterHILBERTS-New/hilbert_{j + 1}.txt", f"theSegHilbert/hilbert_{j + 1}")

    return New_solution


def evolution(times, m, n, c, num, solution):
    for j in tqdm(range(c), colour='blue', desc="Cross mutation yields new Hilbert curve encoding"):
        # ========== Load Hilbert curve ==========
        path = 'theSegHilbert' if times == 1 else 'theSegHilbert_new'
        list1 = []
        for i in range(m):
            hilbert_path = f"{path}/hilbert_{i + 1}/hilbert_seg_{j + 1}.txt"
            with open(hilbert_path, "r") as f:
                list1.append([int(line.strip()) for line in f])

        # ========== Crossover and variation ==========
        numbers = list(range(m))
        for _ in range(len(numbers) // 2):
            x, y = random.sample(numbers, 2)
            numbers = list(set(numbers) - {x, y})
            Operator.Crossover(x, y, list1)
        Operator.RandomVariation(list=list1, n=m, Variation_probability=0.1)
        # Save new Hilbert curve
        for i in range(m):
            folder = f"theSegHilbert_new/hilbert_{i + 1}"
            os.makedirs(folder, exist_ok=True)
            with open(f"{folder}/hilbert_seg_{j + 1}.txt", "w") as f:
                f.writelines(f"{num}\n" for num in list1[i])

        if os.path.exists('theSegHilbert_middle'):
            shutil.rmtree('theSegHilbert_middle')
        shutil.copytree('theSegHilbert', 'theSegHilbert_middle')

        # ========== Generate a new solution ==========
        solution2, solutionMeg = [], copy.deepcopy(solution)
        for i in range(m):
            shutil.copy(
                f"theSegHilbert_new/hilbert_{i + 1}/hilbert_seg_{j + 1}.txt",
                f"BestPopulation/segmentLittleHilbert_change/hilbert_seg_{j + 1}.txt"
            )
            mergeHilber('BestPopulation/segmentLittleHilbert_change', f'theMergeHilbert/hilbert_{i + 1}.txt')
            with open(f'theMergeHilbert/hilbert_{i + 1}.txt') as f:
                new_data = [int(line.strip()) for line in f]
            k = 0
            for x, val in enumerate(solutionMeg[i]):
                if val != 0:
                    solutionMeg[i][x] = new_data[k]
                    k += 1
            with open(f"theMergeHilbert_return/hilbert_{i + 1}.txt", "w") as f:
                f.writelines(f"{v}\n" for v in solutionMeg[i])
            with open(f"theMergeHilbert_return/hilbert_{i + 1}.txt") as f:
                solution2.append([int(line.strip()) for line in f])
        # ========== Map and save 3D LUTs ==========
        for i in tqdm(range(m), colour='blue', desc='Map and save images'):
            hilbert_get_cube.main2(n=n, l=i + 1, times=times, list=solution2[i])
            luts = f"ourNewLuts/luts_{i + 1}.cube"
            cube = f"ourNewCUBES/New cube_{i + 1}.pickle"
            get_new_luts.main(LutsName=luts, CubeName=cube, n=n)
            dst = f"{PROJECT_ROOT}/main_result/result/New Test{times + 1}/Test{i + 1}"
            os.makedirs(dst, exist_ok=True)
            shutil.copy(luts, dst)
            mapping_plus_tensor_a100_trilinear_interpolation.Mapping(
                i, cubepath='ourNewLuts', times=times + 1, n=n, dataloader=dataloader
            )
        # ========== Adaptation value calculation ==========
        acc_dir = f"accuracy/accuracy{times}"
        os.makedirs(acc_dir, exist_ok=True)
        acc_path = f"{acc_dir}/accuracy_{j + 1}.txt"
        with open(acc_path, "w") as f:
            folder_paths = [f"{PROJECT_ROOT}/main_result/result/New Test{times + 1}/Test{i + 1}/Test_images" for i
                            in range(m)]
            args = [
                (folder, json_name, scripted_model_path, 20, device, data_transform, load_class_mapping(json_name))
                for folder in folder_paths]
            with Pool(processes=2) as pool:
                losses = pool.map(calculate_folder_loss, args)
            Min = min(losses)
            for loss in losses:
                print("Loss =", loss)
                f.write(f"{loss}\n")

        # ========== Selection and Update ==========
        if os.path.exists('theSegHilbert_new_test'):
            shutil.rmtree('theSegHilbert_new_test')
        shutil.copytree('theSegHilbert_new', 'theSegHilbert_new_test')
        with open(acc_path) as f:
            numbers_1 = [float(line.strip()) for line in f]
        Select.tournament_selection(numbers_1, int(m * 0.6), "Util/EvoLUT_Util/Select_list.txt")

        # --------------The number is saved in Selectilist.txt, and the selected luts table is stored in the project----------------
        with open("Util/EvoLUT_Util/Select_list.txt") as f:
            selectlist = [line.strip() for line in f]
        for x in range(m):
            shutil.copy(
                f"theSegHilbert_new/hilbert_{selectlist[x]}/hilbert_seg_{j + 1}.txt",
                f"theSegHilbert_new_test/hilbert_{x + 1}/hilbert_seg_{j + 1}.txt"
            )

        shutil.rmtree('theSegHilbert_new')
        shutil.copytree('theSegHilbert_new_test', 'theSegHilbert_new')
        # ========== Update the optimal solution ==========
        if Min <= list_answer[-1]:
            num += 1
            list_answer.append(Min)
            list_index.append(num)
            the_best_population = get_best_population(j + 1, times)
            copydir('theMergeHilbert_return/', 'BestPopulation/hilbert/', the_best_population,
                    'BestPopulation/hilbert/thebestpopulathon.txt')
            copydir('theMergeHilbert/', 'BestPopulation/hilbert/', the_best_population,
                    'BestPopulation/hilbert/thebestLittlepopulathon.txt')

        segmentHilbert(c, 'BestPopulation/hilbert/thebestLittlepopulathon.txt', 'BestPopulation/segmentLittleHilbert')
        if os.path.exists('BestPopulation/segmentLittleHilbert_change'):
            shutil.rmtree('BestPopulation/segmentLittleHilbert_change')
        shutil.copytree('BestPopulation/segmentLittleHilbert', 'BestPopulation/segmentLittleHilbert_change')

    return solution


def get_best_population(times, c=None):
    if c is None:
        file_path = f'accuracy/accuracy{times}.txt'
    else:
        file_path = f'accuracy/accuracy{times}/accuracy_{c}.txt'

    try:
        with open(file_path, 'r') as f:
            min_value, best_line = float('inf'), None
            for idx, line in enumerate(f, 1):
                numbers = [float(x) for x in line.split()]
                if numbers:
                    value = min(numbers)
                    if value < min_value:
                        min_value, best_line = value, idx

        if best_line:
            print(f"The minimum loss value for this round is {min_value}, corresponding to the {best_line} th individual")
            return best_line
        else:
            print(f"There is no valid data in {file_path}")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"error: {e}")
        return None


def main():
    deletTxt()
    torch.cuda.empty_cache()

    # load model
    model = EfficientNetLite0(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Script based model saving
    torch.jit.script(model).save(scripted_model_path)
    print(f"The scripted model has been saved to: {scripted_model_path}")

    # Whether to use pruning operation
    if useCut:
        usedPixList, lenusedPixList, len2, len1 = getUsedRGBpix_trilinear.GetRGBpix(n)
    else:
        usedPixList = list(range(v * v * v))

    # *********************************** initialization *******************************************
    print("Initialization started：")
    solution = initialization(m, n, c, 0, 1, usedPixList)
    # *********************************** optimize *******************************************
    print("Start optimizing 3D LUTs：")
    num = 1
    for i in range(epoch):
        print(f"The {i+1}/{epoch} round of optimization begins")

        folder_to_delete = PROJECT_ROOT + f"/main_result/result/New Test{i - keep_count}"
        if i > keep_count and os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)

        solution = evolution(i + 1, m, n, c, num, solution)

        save_to_txt(list_index, list_answer)
        draw(list_index, list_answer, i + 1)
        num = len(list_answer)

    print("3D LUTs optimization completed")


if __name__ == '__main__':
    dataloader = DataLoader(
        dataset=Initial_def.ImageDataset(config["data_path"]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    main()
