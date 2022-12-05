import os
import numpy as np
import get_and_transform_dataset as get_data
import re

skeleton_images_folder_name = 'utdmhad_skeleton'
train_folder_name = 'train'
test_folder_name = 'test'
meta_data_sub_path = 'meta_classes.npy'


def get_train_or_test_path(base_path, train=True):
    skeleton_images_base_path = os.path.join(base_path, skeleton_images_folder_name)

    if train:
        data_path = os.path.join(skeleton_images_base_path, train_folder_name)
    else:
        data_path = os.path.join(skeleton_images_base_path, test_folder_name)
    return data_path


def getValidSplit(names, byS=True):
    # set activities_from_zero = True to get targets from 0..26 instead of 1..27
    def get_subject_id_from_filename(filename):
        pattern = r'_s(\d+)_'
        m = re.search(pattern, filename)
        subject_id = -1
        if m:
            subject_id = int(m.group(1))
        else:
            print('Not possible to extract subject_id: file:', filename)
            subject_id = 0
        return subject_id


    def get_trial_id_from_filename(filename):
        pattern = r'_t(\d+)_'
        m = re.search(pattern, filename)
        trial_id = -1
        if m:
            trial_id = int(m.group(1))
        else:
            print('Not possible to extract trial_id: file:', filename)
            trial_id = 0
        return trial_id
    split_names = []
    split_ids = []
    for i in range(len(names)):
        name = names[i]
        s_id = get_subject_id_from_filename(name)
        t_id = get_trial_id_from_filename(name)
        if byS:
            # if s_id % 4 != 0:
            #     split_names.append(name)
            #     split_ids.append(i)
            if s_id == 7:
                split_names.append(name)
                split_ids.append(i)
        else:
            if t_id % 2 == 1:
                split_names.append(name)
                split_ids.append(i)
    print(split_names)
    print(split_ids)
    if byS:
        test_if_same = [x for x in range(0, 112) if x % 16 > 11] + [x-1 for x in range(113, 432) if x % 16 > 11]
        print(test_if_same)
        print(test_if_same == split_ids)
        print(f'Length of all validation data: {len(test_if_same)}')
    else:
        test = list(range(0,363,2)) + list(range(363, 430, 2))
        print(test)
        print(test == split_ids)




# path to data
base_path = 'D:\\Desktop\\MA\\data_sets\\UTD-MHAD'
desired_width = 68
fill_with_zeros = False

# get path from base path
train_path = get_train_or_test_path(base_path, True)
test_path = get_train_or_test_path(base_path, False)

# get all data in a dict
all_train_data = get_data.get_all_data(train_path, desired_width, fill_with_zeros)
# all_test_data = get_data.get_all_data(test_path, desired_width, fill_with_zeros)
#
# test_data_names = all_test_data['names']
print(f'Length of all train data: {len(all_train_data["names"])}')
num_images = 431  # split the test_data

# valid_split = [x for x in range(num_images) if x % 2 == 0]
# print([test_data_names[i] for i in valid_split])
getValidSplit(all_train_data['names'], True)
# print('by person:')
# getValidSplit(test_data_names, True)
# print('by trial:')
# getValidSplit(test_data_names, False)
# split_odd_trial_ids = list(range(0,363,2)) + list(range(363,430,2))
# split_subject_2_and_6_ids = [x for x in range(0, 363) if x % 8 < 4] + [x for x in range(367, 430) if x % 8 < 3 or x % 8 == 7]
