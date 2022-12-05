import os
import numpy as np
from ntu_120_get_and_transform_dataset import ntu120_get_all_data
from ntu_60_get_and_transform_dataset import ntu60_get_all_data
import time
import re

skeleton_images_folder_name = 'ntu_120'
train_folder_name = 'train'
test_folder_name = 'test'


# SsssCcccPpppRrrrAaaa with S = setup number, C = camera id, P = performer/subject, R = replication number, A = action class label
def get_valid_split(names, split_by='P', cross_subject=True):

    def get_id_from_filename(filename, pattern):
        m = re.search(pattern, filename)
        if m:
            id = int(m.group(1))
        else:
            print(f'Not possible to extract id: from file "{filename}", with pattern "{pattern}"')
            id = 0
        return id

    if split_by == 'S':
        # split by setup number (1-32)
        pattern = r'S(\d+)C'
    elif split_by == 'P':
        # split by performer/subject (1-40)
        pattern = r'P(\d+)R'
    elif split_by == 'R':
        # split by replication number/trial/execution number (1,2)
        pattern = r'R(\d+)A'
    else:
        print('split_by is set wrong: Please input "S", "P" or "R" for setup, person/subject or replication')
        return

    split_names = []
    split_ids = []
    for i in range(len(names)):
        name = names[i]
        split_id = get_id_from_filename(name, pattern)
        if split_by == 'S':
            # split by setup number (1-32)
            # if split_id in [1,3,5,7,9,12,14,16,17]:
            if split_id % 3 == 0:
                split_names.append(name)
                split_ids.append(i)
        elif split_by == 'P':
            # split by performer/subject (1-106)
            # available ids for the train set: [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            # 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            # 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
            if split_id % 3 == 2:
                split_names.append(name)
                split_ids.append(i)

        elif split_by == 'R':
            # split by replication number/trial/execution number (1,2)
            if split_id == 1:
                split_names.append(name)
                split_ids.append(i)
    if split_by == 'S':
        # split by setup number (1-32)
        valid = [f'S{i:03d}' for i in range(3, 33, 3)]
        print(f'Split by "{split_by}" (setup number)')
        print(f'Get all images with {valid}')
    elif split_by == 'P':
        # split by performer/subject (1-40)
        if cross_subject:
            valid = [f'P{i:03d}' for i in [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103] if i % 3 == 2]
        else:
            valid = [f'P{i:03d}' for i in range(3, 41, 3)]
        print(f'Split by "{split_by}" (performer/subject id)')
        print(f'Get all images with {valid}')
    elif split_by == 'R':
        # split by replication number/trial/execution number (1,2)
        print(f'Split by "{split_by}" (replication number)')
        print(f"Get all images with ['R001']")

    print(f'length: train:{len(names)-len(split_ids)}, val:{len(split_ids)}')
    # print(split_names)
    print(split_ids)


def get_train_or_test_path(base_path, train=True):
    if train:
        data_path = os.path.join(base_path, train_folder_name)
    else:
        data_path = os.path.join(base_path, test_folder_name)
    return data_path


# path to data
base_path = 'D:/Desktop/MA/Torch_Home/ntu_120'
desired_width = 76
fill_with_zeros = False

# get path from base path
# create cross subject
cross_subject = True

# dir_name = 'cross_subject' if cross_subject else 'cross_setup'
dir_name = 'cross_subject' if cross_subject else 'cross_setup'
dir_path = os.path.join(base_path, dir_name)
train_path = get_train_or_test_path(dir_path, True)
test_path = get_train_or_test_path(dir_path, False)

# get all data in a dict
all_train_data = ntu60_get_all_data(train_path, desired_width, fill_with_zeros)
# all_test_data = ntu60_get_all_data(test_path, desired_width, fill_with_zeros)

# test_data_names = all_test_data['names']
train_data_names = all_train_data['names']

print(dir_name, '\n')

# get_valid_split(test_data_names, 'S', cross_subject)
# get_valid_split(train_data_names, 'S', cross_subject)
# print()
# get_valid_split(test_data_names, 'P', cross_subject)
get_valid_split(train_data_names, 'P', cross_subject)
# print()
# get_valid_split(test_data_names, 'R', cross_subject)
# get_valid_split(train_data_names, 'R', cross_subject)
