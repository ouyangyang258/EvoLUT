# encoding=utf-8
import random

def Crossover(x, y, list):
    crossover_probability = 0.5
    for i in range(len(list[x])):
        random_number = random.random()
        if random_number < crossover_probability:
            c = list[x][i]
            list[x][i] = list[y][i]
            list[y][i] = c


def RandomVariation(list, n, Variation_probability):
    sum1 = 0
    variation_probability = Variation_probability
    for i in range(len(list)):
        for j in range(len(list[n - 1])):
            if list[i][j] != 0:
                random_number = random.random()
                if random_number < variation_probability:
                    list[i][j] = random.randint(0, int("111111111111111111111111", 2))
                    sum1 = sum1 + 1
    print(sum1)


