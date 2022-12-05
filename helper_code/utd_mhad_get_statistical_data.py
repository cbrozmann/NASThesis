import os
import numpy as np
import get_and_transform_dataset as get_data

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


def get_classes_array(base_path):
    meta_path = os.path.join(base_path, meta_data_sub_path)
    with open(meta_path, "rb") as f:
        classes = np.load(f)


def print_statistical_data(all_data):
    names = all_data['names']
    dimensions = all_data['dimensions']
    data = all_data['data']
    targets = all_data['targets']

    dim_arr = np.array(dimensions)
    widths = dim_arr[:, 0]
    heights = dim_arr[:, 1]
    # folderName = os.path.basename(image_path)
    print('shape:', data.shape)
    # print(folderName + ':')
    print('width_max:', np.max(widths))
    print('width_min:', np.min(widths))
    print('width_mean:', np.mean(widths))
    print('width_std:', np.std(widths))
    print('width_median:', np.median(widths))
    # print min and max height, just write height if they are the same
    if np.max(heights) != np.min(heights):
        print('height_max:', np.max(heights))
        print('height_min:', np.min(heights))
    else:
        print('height:', np.min(heights))

    print('#images:', data.shape[0])
    print()
    rs = data[:, :, :, 0].flatten()
    gs = data[:, :, :, 1].flatten()
    bs = data[:, :, :, 2].flatten()
    print('means: [', np.mean(rs), np.mean(gs), np.mean(bs), ']')
    print('stds: [', np.std(rs), np.std(gs), np.std(bs), ']')


# path to data
base_path = 'D:\\Desktop\\MA\\data_sets\\UTD-MHAD'
desired_width = 68
fill_with_zeros = False

# get path from base path
train_path = get_train_or_test_path(base_path, True)
test_path = get_train_or_test_path(base_path, False)

# get all data in a dict
all_train_data = get_data.get_all_data(train_path, desired_width, fill_with_zeros)
all_test_data = get_data.get_all_data(test_path, desired_width, fill_with_zeros)

# get meta data (classes)
meta_path = os.path.join(base_path, meta_data_sub_path)
classes = []
with open(meta_path, "rb") as f:
    classes = np.load(f)
# print classes
print('Classes:')
print(classes, '\n')

# print statistical information from all data
print('train:')
print(get_data.get_target_from_filename("a1_s1_t1_skeleton.mat.png", True))
print(all_train_data['names'])
print(all_train_data["targets"])
# train_data_filenames = all_train_data['names']
print_statistical_data(all_train_data)
print()

print('test:')
# test_data_filenames = all_test_data['names']
print_statistical_data(all_test_data)
