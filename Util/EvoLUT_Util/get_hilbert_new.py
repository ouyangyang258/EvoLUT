import pickle
import numpy as np

hilbert_map = {'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
               'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
               'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
               'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
               }
un_hilbert_map = {'a': {0: (0, 0, 'd'), 1: (0, 1, 'a'), 3: (1, 0, 'b'), 2: (1, 1, 'a')},
                  'b': {2: (0, 0, 'b'), 1: (0, 1, 'b'), 3: (1, 0, 'a'), 0: (1, 1, 'c')},
                  'c': {2: (0, 0, 'c'), 3: (0, 1, 'd'), 1: (1, 0, 'c'), 0: (1, 1, 'b')},
                  'd': {0: (0, 0, 'a'), 3: (0, 1, 'c'), 1: (1, 0, 'd'), 2: (1, 1, 'd')}
                  }


def unsigned_int_mixRGB(R, G, B):  # Convert (R, G, B) to an integer
    result = (R << 16) | (G << 8) | B
    return result


def decomposeRGB(input):  # 将整数转化为（R，G，B）
    r = (input >> 16) & 0xff
    g = (input >> 8) & 0xff
    b = input & 0xff
    return [r, g, b]


def point_to_hilbert(x, y, order):  # 一维列表解码正方形矩阵
    current_square = 'a'
    position = 0
    for i in range(order - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position


def hilbert_to_point(d, order):  # 得到由hilbert曲线编码的点
    current_square = 'a'
    x = y = 0
    for i in range(order - 1, -1, -1):  # 3的二进制为11，然后左移2i倍，与d取按位与后右移2i倍，得到象限编码
        mask = 3 << (2 * i)
        quad_position = (d & mask) >> (2 * i)
        quad_x, quad_y, current_square = un_hilbert_map[current_square][quad_position]
        x |= 1 << i if quad_x else 0
        y |= 1 << i if quad_y else 0
    return x, y


def getLitterHilbert(n, CubeName, file_handle, solution, jj):
    EXP = 10 * [1]
    for i in range(1, 10):
        EXP[i] = EXP[i - 1] * 2
    list = []

    with open(CubeName, 'rb') as file:  # 读取初始矩阵
        x = pickle.load(file)
    file.close()

    y = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int)
    y1 = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int)
    for i in range(EXP[n]):
        for j in range(EXP[n]):
            for k in range(EXP[n]):
                y[i][j][k] = int(unsigned_int_mixRGB(x[i][j][k][0], x[i][j][k][1], x[i][j][k][2]))
    y_new = y.copy()
    for j in range(EXP[n]):
        if j % 2 != 0:
            y_new[j] = np.fliplr(y[j])  # x_new中偶数层的数据按对称轴翻转
        for i in range(EXP[n] * EXP[n]):
            (a, b) = hilbert_to_point(i, n)
            c = b
            b = a
            a = c
            list.append(y_new[j][a][b])

    for c in range(EXP[n]):
        for a in range(EXP[n]):
            for b in range(EXP[n]):
                y1[c][a][b] = list[c * EXP[n] * EXP[n] + point_to_hilbert(b, a, n)]

    for i in range(EXP[n]):
        if i % 2 != 0:
            y1[i] = np.fliplr(y1[i]).tolist()

    for i in list:
        if i != 0:
            file_handle.write(str(i))
            file_handle.write("\n")


def getHilbert(n, CubeName, file_handle, solution, jj):
    EXP = 10 * [1]
    for i in range(1, 10):
        EXP[i] = EXP[i - 1] * 2
    list = []

    with open(CubeName, 'rb') as file:  # 读取初始矩阵
        x = pickle.load(file)
    file.close()
    y = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int)
    y1 = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int)

    for i in range(EXP[n]):
        for j in range(EXP[n]):
            for k in range(EXP[n]):
                y[i][j][k] = int(unsigned_int_mixRGB(x[i][j][k][0], x[i][j][k][1], x[i][j][k][2]))

    y_new = y.copy()

    for j in range(EXP[n]):
        if j % 2 != 0:
            y_new[j] = np.fliplr(y[j])  # x_new中偶数层的数据按对称轴翻转
        for i in range(EXP[n] * EXP[n]):
            (a, b) = hilbert_to_point(i, n)
            c = b
            b = a
            a = c

            list.append(y_new[j][a][b])

    for c in range(EXP[n]):
        for a in range(EXP[n]):
            for b in range(EXP[n]):
                y1[c][a][b] = list[c * EXP[n] * EXP[n] + point_to_hilbert(b, a, n)]

    for i in range(EXP[n]):
        if i % 2 != 0:
            y1[i] = np.fliplr(y1[i]).tolist()

    for i in list:
        file_handle.write(str(i))
        solution[jj].append(i)
        file_handle.write("\n")
