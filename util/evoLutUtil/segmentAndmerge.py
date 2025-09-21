import os
import re

def segmentHilbert(c, hilbert_path, output_path):
    # 指定保存文件的目录路径
    output_directory = output_path

    # 创建目录（如果不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 读取原始文本文件
    with open(hilbert_path, 'r') as original_file:
        lines = original_file.readlines()

    # 计算每个文件的行数
    total_lines = len(lines)
    lines_per_file = total_lines // c

    # 将数据分为四个部分
    split_data = [lines[i:i + lines_per_file] for i in range(0, total_lines, lines_per_file)]

    # 保存四个部分为四个新文本文件，指定目录路径
    for i, data in enumerate(split_data):
        output_filename = os.path.join(output_directory, f'hilbert_seg_{i + 1}.txt')
        with open(output_filename, 'w') as output_file:
            output_file.writelines(data)

    #print("文件分割完成")

def sort_key(item):
    # 使用正则表达式提取字符串中的数字部分
    match = re.search(r'\d+', item)
    if match:
        # 返回提取的数字部分作为排序关键字
        return int(match.group())
    else:
        # 如果没有数字部分，默认返回0
        return 0

def mergeHilber(folder_path,output_path):
    # 获取文件夹中的所有txt文件
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    txt_files = sorted(txt_files, key=sort_key)
    # 合并后的文件名
    merged_file_name = output_path

    # 打开合并后的文件
    with open(merged_file_name, 'w') as merged_file:
        for txt_file in txt_files:
            txt_file_path = os.path.join(folder_path, txt_file)
            with open(txt_file_path, 'r') as input_file:
                lines = input_file.readlines()
                merged_file.writelines(lines)

    #print("文件合并完成")


if __name__ == '__main__':
    c = 12
    hilbert_path = '../../BestPopulation/hilbert/thebestpopulathon.txt'
    output_path = '../../BestPopulation/segmentHilbert'
    segmentHilbert(c, hilbert_path, output_path)

    # folder_path = 'BestPopulation/segmentHilbert_change'
    # output_path = 'theMergeHilbert/hilbert.txt'
    # mergeHilber(folder_path,output_path)
