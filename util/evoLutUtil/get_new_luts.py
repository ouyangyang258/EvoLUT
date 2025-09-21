# encoding=utf-8
import glob
import os
import pickle
import pprint
import shutil
from shutil import copy

import numpy as np
from numpy.core.defchararray import isdigit
from tqdm import tqdm

global EXP
EXP = 10 * [1]
for i in range(1, 10):
    EXP[i] = EXP[i - 1] * 2


def main(LutsName, CubeName, n):
    # x = np.random.randint(0, 256, size=(EXP[n], EXP[n], EXP[n], 3)).tolist()  # 随机生成矩阵
    with open(CubeName, 'rb') as file:
        x = pickle.load(file)
    # with open(CubeName, 'rb') as file:  # 读取初始矩阵
    #    x_begin = pickle.load(file)
    # file.close()
    # print(x_begin == x)

    rgbFloatCube_new = []
    for B in range(EXP[n]):
        for G in range(EXP[n]):
            for R in range(EXP[n]):
                rgbFloat = "%s %s %s" % (str(x[B][G][R][0] / 255), str(x[B][G][R][1] / 255), str(x[B][G][R][2] / 255))
                rgbFloatCube_new.append(rgbFloat)
    # pprint.pprint(rgbFloatCube_new)
    # print(len(rgbFloatCube_new))

    file_handle = open(LutsName, mode='w')
    file_handle.write("#data domain\n"
                      "DOMAIN_MIN 0.0 0.0 0.0\n"
                      "DOMAIN_MAX 1.0 1.0 1.0\n"
                      "#LUT data points\n")

    for i in rgbFloatCube_new:
        file_handle.write(str(i))
        file_handle.write("\n")
    file_handle.close()
    # print("已生成初始种群矩阵和对应的luts表")
    # pprint.pprint(x)


