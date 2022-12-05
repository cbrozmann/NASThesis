import os
from PIL import Image
import numpy as np
import re
from pathlib import Path


def is_train(filename, cross_subject=True):
    # filename looks like 'a'+action_number+'_s'+action_number+'_t'+trial+'_skeleton.mat'
    if cross_subject:
        # split by subject ID
        train_subject_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
        # SsssCcccPpppRrrrAaaa with S = setup number, C = camera id, P = performer/subject, R = replication number, A = action class label
        pattern = r'P(\d+)R'
        m = re.search(pattern, filename)
        if m:
            # is subject id in the list
            file_subject_id = int(m.group(1))
            return file_subject_id in train_subject_ids
        else:
            return False
    else:
        pattern = r'S(\d+)C'
        m = re.search(pattern, filename)
        if m:
            # is setup id odd?
            file_setup_id = int(m.group(1))
            return file_setup_id % 2 == 1
        else:
            return False


def list_filepaths_and_names(dir, file_extension='.npy'):
    filepaths = []
    filenames = []
    for root, dirs, files in os.walk(dir):
        # sort, because walk returns otherwise in arbitrary order
        dirs.sort()
        for name in sorted(files):
            if file_extension:
                if name.endswith(file_extension):
                    filepath = os.path.join(root, name)
                    filepaths.append(filepath)
                    filenames.append(name)
            else:
                filepath = os.path.join(root, name)
                filepaths.append(filepath)
                filenames.append(name)
    return filepaths, filenames


def transform_skeleton_to_image(path, round=False):
    # load skeleton data
    data = np.load(path, allow_pickle=True).item()
    skeleton_arr = data['skel_body0']
    # normalize data to range [0..1]
    matrix = np.swapaxes(skeleton_arr, 0, 1)
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    # to range [0..255] (still float)
    matrix = matrix * 255
    # to int [0..255]
    # should be rounded before the transformation from float to int?
    if round:
        matrix = np.around(matrix)
    # change type to int
    img_arr = matrix.astype(np.uint8)
    img = Image.fromarray(img_arr, 'RGB')
    return img


def transform_all_data(input_path, output_path, cross_subject=True, round=False):
    filepaths, filenames = list_filepaths_and_names(input_path, '.npy')
    # create train and test directories
    train_folder_path = Path(os.path.join(output_path, train_folder_name))
    test_folder_path = Path(os.path.join(output_path, test_folder_name))
    train_folder_path.mkdir(parents=True)
    test_folder_path.mkdir(parents=True)
    for filepath, filename in zip(filepaths, filenames):
        # get image from the path to the skeleton
        img = transform_skeleton_to_image(filepath, round)
        # should be assigned to train or test set?
        train = is_train(filename, cross_subject)
        # get the path were the image should be saved
        folder_name = train_folder_name if train else test_folder_name
        # for the name remove the .npy and add .png
        save_path = os.path.join(output_path, folder_name, filename[:-4]+'.png')
        # save image
        img.save(save_path)


input_path = 'D:/Desktop/MA/data_sets/NTURGB/data_transform/raw_npy'
output_root_path = 'D:/Desktop/MA/data_sets/NTURGB/data_transform/ntu120'

train_folder_name = 'train'
test_folder_name = 'test'

# create cross subject
cross_subject = True
dir_name = 'cross_subject' if cross_subject else 'cross_setup'
output_path = os.path.join(output_root_path, dir_name)
transform_all_data(input_path, output_path, cross_subject, False)

# create cross setup
cross_subject = False
dir_name = 'cross_subject' if cross_subject else 'cross_setup'
output_path = os.path.join(output_root_path, dir_name)
transform_all_data(input_path, output_path, cross_subject, False)