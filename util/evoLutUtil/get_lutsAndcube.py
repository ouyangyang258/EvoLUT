# encoding=utf-8
import pickle
from shutil import copy
import numpy as np
from tqdm import tqdm

from main import PROJECT_ROOT

global EXP
EXP = 10 * [1]
for i in range(1, 10):
    EXP[i] = EXP[i - 1] * 2


def saveLuts(path, times):
    if times == 1:
        fd = open(path)  # 选择一个cube映射文件
        lines = fd.readlines()
        y = 1
        for l in lines:
            x = int(l)
            dir_dst_Luts =PROJECT_ROOT + '/ourNewLuts/luts_{}.cube'.format(y)
            file = PROJECT_ROOT + '/ourLuts/luts_{}.cube'.format(x)
            copy(file, dir_dst_Luts)
            y += 1
    else:
        fd = open(path)  # 选择一个cube映射文件
        lines = fd.readlines()
        y = 1
        for l in lines:
            x = int(l)
            dir_dst_Luts = PROJECT_ROOT + '/ourNewLuts/luts_{}.cube'.format(y)
            file = PROJECT_ROOT + '/ourNewLuts-test/luts_{}.cube'.format(x)
            copy(file, dir_dst_Luts)
            y += 1


def getLutsAndCube(LutsName, CubeName, n, usedPixList):
    shape = (EXP[n], EXP[n], EXP[n], 3)
    y = np.random.randint(0, 256, size=shape)
    sum1 = 0
    rgbFloatCube_new = []
    for B in range(EXP[n]):
        for G in range(EXP[n]):
            for R in range(EXP[n]):
                if sum1 not in usedPixList:
                    y[B][G][R][0] = 0
                    y[B][G][R][1] = 0
                    y[B][G][R][2] = 0
                rgbFloat = "%s %s %s" % (
                    str(y[B][G][R][0].item() / 255), str(y[B][G][R][1].item() / 255), str(y[B][G][R][2].item() / 255))
                rgbFloatCube_new.append(rgbFloat)
                sum1 += 1

    with open(CubeName, 'wb') as file:  # 保存初始矩阵
        pickle.dump(y, file)
    file.close()

    file_handle = open(LutsName, mode='w')
    file_handle.write("#data domain\n"
                      "DOMAIN_MIN 0.0 0.0 0.0\n"
                      "DOMAIN_MAX 1.0 1.0 1.0\n"
                      "#LUT data points\n")

    for x in rgbFloatCube_new:
        file_handle.write(str(x))
        file_handle.write("\n")

    file_handle.close()
