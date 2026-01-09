# encoding=utf-8
import pickle
import numpy as np

from main import PROJECT_ROOT

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


def decomposeRGB(input):  # Convert an integer to (R, G, B)
    r = (input >> 16) & 0xff
    g = (input >> 8) & 0xff
    b = input & 0xff
    return [r, g, b]


def point_to_hilbert(x, y, order):
    current_square = 'a'
    position = 0
    for i in range(order - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position


def hilbert_to_point(d, order):
    current_square = 'a'
    x = y = 0
    for i in range(order - 1, -1, -1):
        mask = 3 << (2 * i)
        quad_position = (d & mask) >> (2 * i)
        quad_x, quad_y, current_square = un_hilbert_map[current_square][quad_position]
        x |= 1 << i if quad_x else 0
        y |= 1 << i if quad_y else 0
    return x, y


def main(n, file_handle, l, times):
    EXP = 10 * [1]
    for i in range(1, 10):
        EXP[i] = EXP[i - 1] * 2

    y = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    y_new = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    list = [int(line.strip()) for line in file_handle.readlines()]
    for c in range(EXP[n]):
        for a in range(EXP[n]):
            for b in range(EXP[n]):
                y[c][a][b] = list[c * EXP[n] * EXP[n] + point_to_hilbert(b, a, n)]

    for i in range(EXP[n]):
        if i % 2 != 0:
            y[i] = np.fliplr(y[i]).tolist()

    for i in range(EXP[n]):
        for j in range(EXP[n]):
            for k in range(EXP[n]):
                y_new[i][j][k] = decomposeRGB(int(y[i][j][k]))
    with open(PROJECT_ROOT + '/ourNewCUBES-test/New cube_{}.pickle'.format(l), 'wb') as file:
        pickle.dump(y_new, file)


def main2(n, l, times, list):
    EXP = 10 * [1]
    for i in range(1, 10):
        EXP[i] = EXP[i - 1] * 2

    y = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    y_new = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    for c in range(EXP[n]):
        for a in range(EXP[n]):
            for b in range(EXP[n]):
                y[c][a][b] = list[c * EXP[n] * EXP[n] + point_to_hilbert(b, a, n)]

    for i in range(EXP[n]):
        if i % 2 != 0:
            y[i] = np.fliplr(y[i]).tolist()

    for i in range(EXP[n]):
        for j in range(EXP[n]):
            for k in range(EXP[n]):
                y_new[i][j][k] = decomposeRGB(int(y[i][j][k]))
    with open(PROJECT_ROOT + '/ourNewCUBES/New cube_{}.pickle'.format(l), 'wb') as file:
        pickle.dump(y_new, file)

def main3(n, l, times, list):
    EXP = 10 * [1]
    for i in range(1, 10):
        EXP[i] = EXP[i - 1] * 2

    y = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    y_new = np.zeros([EXP[n], EXP[n], EXP[n]], dtype=int).tolist()
    for c in range(EXP[n]):
        for a in range(EXP[n]):
            for b in range(EXP[n]):
                y[c][a][b] = list[c * EXP[n] * EXP[n] + point_to_hilbert(b, a, n)]

    for i in range(EXP[n]):
        if i % 2 != 0:
            y[i] = np.fliplr(y[i]).tolist()

    for i in range(EXP[n]):
        for j in range(EXP[n]):
            for k in range(EXP[n]):
                y_new[i][j][k] = decomposeRGB(int(y[i][j][k]))
    with open(PROJECT_ROOT + '/ourNewCUBES-test/New cube_{}.pickle'.format(l), 'wb') as file:
        pickle.dump(y_new, file)


def extract_integers_from_txt(file_path):
    numbers = []

    with open(file_path, 'r') as file:
        for line in file:

            try:
                number = int(line.strip())
                numbers.append(number)
            except ValueError:
                pass

    return numbers

if __name__ == '__main__':
    m = 100
    numbers = extract_integers_from_txt(r"C:\Users\22279\Documents\Test2_alexnet\BestPopulation\hilbert\thebestpopulathon.txt")
    hilbertName = numbers
    CubeName = r"C:\Users\22279\Documents\Test2_alexnet\BestPopulation\cube\best.pickle"
    main3(n=3, l=1, times=1, list=numbers)
