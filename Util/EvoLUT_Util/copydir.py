import os
import shutil

from Util.constants import PROJECT_ROOT


def copydir(source_folder, destination_folder, line_to_copy, new_file_name):
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

    file_list = os.listdir(source_folder)
    if source_folder in sort_rules:
        file_list.sort(key=sort_rules[source_folder])

    file_position = 0
    for file_name in file_list:
        if not file_name.lower().endswith(('.txt', '.pickle', '.cube')):
            continue

        file_position += 1
        if file_position == line_to_copy:
            src_path = os.path.join(source_folder, file_name)
            dst_path = os.path.join(destination_folder, file_name)

            shutil.copy(src_path, dst_path)

            if os.path.exists(new_file_name):
                os.remove(new_file_name)
            os.rename(dst_path, new_file_name)

            print(f"The {line_to_copy}th file, {file_name}, has been copied to the destination folder {destination_folder} and renamed to {new_file_name}.")
            break
