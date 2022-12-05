import scipy.io
import os
import numpy as np
import re
from PIL import Image
from pathlib import Path


def list_filepaths_and_names(dir, file_extension='.mat'):
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


# all odd subject numbers (1,3,5,7) belong to the training set
# all even (2,4,6,8) to the test set
def is_train_rule(subject_number):
    return subject_number % 2 == 1


def is_train(filename):
    # filename looks like 'a'+action_number+'_s'+action_number+'_t'+trial+'_skeleton.mat'
    pattern = r'_s(\d+)_t'
    m = re.search(pattern, filename)
    if m:
        subject_number = int(m.group(1))
        return is_train_rule(subject_number)
    return False


def transform_all_data(input_path, output_path, round=False):
    filepaths, filenames = list_filepaths_and_names(input_path, '.mat')
    # create train and test directories
    train_folder_path = Path(os.path.join(output_path, train_folder_name))
    test_folder_path = Path(os.path.join(output_path, test_folder_name))
    train_folder_path.mkdir(parents=True)
    test_folder_path.mkdir(parents=True)
    for filepath, filename in zip(filepaths, filenames):
        # get image from the path to the skeleton
        img = transform_skeleton_to_image(filepath, round)
        # should be assigned to train or test set?
        train = is_train(filename)
        # get the path were the image should be saved
        folder_name = train_folder_name if train else test_folder_name
        save_path = os.path.join(output_path, folder_name, filename+'.png')
        # save image
        img.save(save_path)


def transform_skeleton_to_image(path, round=False):
    # load skeleton data
    mat = scipy.io.loadmat(path)
    skeleton_dat = mat['d_skel']
    skeleton_arr = np.array(skeleton_dat)
    # normalize data to range [0..1]
    matrix = np.swapaxes(skeleton_arr, 1, 2)
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


input_path = 'D:/Desktop/MA/data_sets/UTD-MHAD/Skeleton'
output_root_path = 'D:/Desktop/MA/data_sets/UTD-MHAD'

dir_name = 'utdmhad_skeleton'
train_folder_name = 'train'
test_folder_name = 'test'

output_path = os.path.join(output_root_path, dir_name)

transform_all_data(input_path, output_path, False)
