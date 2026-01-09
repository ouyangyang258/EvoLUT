import os
import re

def segmentHilbert(c, hilbert_path, output_path):
    output_directory = output_path

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(hilbert_path, 'r') as original_file:
        lines = original_file.readlines()

    total_lines = len(lines)
    lines_per_file = total_lines // c

    split_data = [lines[i:i + lines_per_file] for i in range(0, total_lines, lines_per_file)]

    for i, data in enumerate(split_data):
        output_filename = os.path.join(output_directory, f'hilbert_seg_{i + 1}.txt')
        with open(output_filename, 'w') as output_file:
            output_file.writelines(data)


def sort_key(item):
    match = re.search(r'\d+', item)
    if match:
        return int(match.group())
    else:
        return 0

def mergeHilber(folder_path,output_path):
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    txt_files = sorted(txt_files, key=sort_key)
    merged_file_name = output_path

    with open(merged_file_name, 'w') as merged_file:
        for txt_file in txt_files:
            txt_file_path = os.path.join(folder_path, txt_file)
            with open(txt_file_path, 'r') as input_file:
                lines = input_file.readlines()
                merged_file.writelines(lines)


if __name__ == '__main__':
    c = 12
    hilbert_path = '../../BestPopulation/hilbert/thebestpopulathon.txt'
    output_path = '../../BestPopulation/segmentHilbert'
    segmentHilbert(c, hilbert_path, output_path)
