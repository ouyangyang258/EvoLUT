# coding:utf-8
import os
import shutil

from util.constants import PROJECT_ROOT


def copydir(source_folder, destination_folder, line_to_copy, new_file_name):
    """
    从源文件夹中复制指定顺序的文件到目标文件夹，并重命名。

    Args:
        source_folder (str): 源文件夹路径
        destination_folder (str): 目标文件夹路径
        line_to_copy (int): 按排序后的顺序要复制的文件序号（1 开始）
        new_file_name (str): 复制到目标文件夹后的新文件名
    """
    # 定义排序规则
    sort_rules = {
        PROJECT_ROOT + '/ourHILBERTS/': lambda x: int(x[8:-4]),
        PROJECT_ROOT + '/ourLitterHILBERTS/': lambda x: int(x[8:-4]),
        PROJECT_ROOT + '/ourLuts/': lambda x: int(x[5:-5]),
        PROJECT_ROOT + '/ourCUBES/': lambda x: int(x[5:-7]),
        PROJECT_ROOT + '/ourNewHILBERTS-test/': lambda x: int(x[12:-4]),
        PROJECT_ROOT + '/ourNewCUBES-test/': lambda x: int(x[9:-7]),
        PROJECT_ROOT + '/ourNewLuts-test/': lambda x: int(x[5:-5]),
        PROJECT_ROOT + '/theMergeHilbert/': lambda x: int(x[8:-4]),
        PROJECT_ROOT + '/theMergeHilbert_return/': lambda x: int(x[8:-4]),
    }

    # 获取源文件夹的文件列表并排序
    file_list = os.listdir(source_folder)
    if source_folder in sort_rules:
        file_list.sort(key=sort_rules[source_folder])

    # 遍历文件并按序号选择要复制的文件
    file_position = 0
    for file_name in file_list:
        if not file_name.lower().endswith(('.txt', '.pickle', '.cube')):
            continue

        file_position += 1
        if file_position == line_to_copy:
            src_path = os.path.join(source_folder, file_name)
            dst_path = os.path.join(destination_folder, file_name)

            # 复制文件到目标文件夹
            shutil.copy(src_path, dst_path)

            # 如果目标文件名已存在，先删除再重命名
            if os.path.exists(new_file_name):
                os.remove(new_file_name)
            os.rename(dst_path, new_file_name)

            print(f"已将第 {line_to_copy} 个文件 {file_name} 复制到目标文件夹 {destination_folder} 并重命名为 {new_file_name}")
            break
