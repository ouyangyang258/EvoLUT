# -*- coding: utf-8 -*-
import os
import shutil
from matplotlib import pyplot as plt

from util.constants import PROJECT_ROOT


def save_to_txt(list_index, list_answer, filename="results.txt"):
    """
    将索引和对应结果保存为制表符分隔的文本文件。

    Args:
        list_index (list): 索引列表
        list_answer (list): 对应结果列表
        filename (str): 输出文件名，默认为 "results.txt"
    """
    file_path = os.path.join(PROJECT_ROOT, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Index\tAnswer\n")
        for idx, ans in zip(list_index, list_answer):
            f.write(f"{idx}\t{ans}\n")
    print(f"结果已保存至 {file_path}")

def draw(x, y, times):
    plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    # 创建折线图
    plt.plot(x, y, marker='o', markersize=4, color='darkblue', linestyle='-')  # 'o' 表示在数据点处绘制圆点
    plt.title('EvoLUT', fontsize=12) # 设置标题
    plt.xlabel('Iterations', fontsize=10)  # 设置X轴标签
    plt.ylabel('Accuracy', fontsize=10)  # 设置Y轴标签
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, (xi, yi) in enumerate(zip(x, y)):
        # f'{yi}'--->f'{' '}'
        plt.annotate(f'{""}', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.savefig(PROJECT_ROOT +'/resultImages/result_{}.png'.format(times))
    # plt.show()  # 显示图形
    pass
def getImagesNumber():
    folder_path = PROJECT_ROOT + 'data_images'  # 替换为实际文件夹的路径
    jpg_count = 0  # 用于计算.jpg文件的数量

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_count += 1

    # print(f"在文件夹 {folder_path} 中找到的.jpg文件数量为: {jpg_count}")
    return jpg_count

def delete_specified_files(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有内容
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 如果是文件，删除文件
            if os.path.isfile(file_path):
                os.remove(file_path)
            # 如果是文件夹，递归删除其内容
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            # print(f"Deleted: {file_path}")


def deletTxt():
    delete_specified_files('BestPopulation/cube')
    delete_specified_files('BestPopulation/hilbert')
    delete_specified_files('BestPopulation/lut')
    delete_specified_files('BestPopulation/segmentLittleHilbert')
    delete_specified_files('BestPopulation/segmentLittleHilbert_change')
    delete_specified_files('theMergeHilbert')
    delete_specified_files('theMergeHilbert_return')
    delete_specified_files('accuracy')
    delete_specified_files('theSegHilbert')
    delete_specified_files('theSegHilbert_middle')
    delete_specified_files('theSegHilbert_new')
    delete_specified_files('theSegHilbert_new_test')
    delete_specified_files('resultImages')
    delete_specified_files('ourLitterHILBERTS-New')
    delete_specified_files('ourLitterHILBERTS')
    delete_specified_files('ourHILBERTS')
    delete_specified_files('ourCUBES')
    delete_specified_files('ourLuts')
    delete_specified_files('ourNewCUBES')
    delete_specified_files('ourNewLuts')

if __name__ == '__main__':
    deletTxt()