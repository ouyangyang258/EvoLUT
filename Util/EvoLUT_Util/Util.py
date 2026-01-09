import os
import shutil
from matplotlib import pyplot as plt

from Util.constants import PROJECT_ROOT


def save_to_txt(list_index, list_answer, filename="results.txt"):
    file_path = os.path.join(PROJECT_ROOT, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Index\tAnswer\n")
        for idx, ans in zip(list_index, list_answer):
            f.write(f"{idx}\t{ans}\n")
    print(f"The results have been saved to {file_path}")

def draw(x, y, times):
    plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    # Create a line chart
    plt.plot(x, y, marker='o', markersize=4, color='darkblue', linestyle='-')
    plt.title('EvoLUT', fontsize=12)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(f'{""}', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.savefig(PROJECT_ROOT +'/resultImages/result_{}.png'.format(times))
    pass
def getImagesNumber():
    folder_path = PROJECT_ROOT + 'data_images'
    jpg_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_count += 1
    return jpg_count

def delete_specified_files(folder_path):
    if os.path.exists(folder_path):

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)



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