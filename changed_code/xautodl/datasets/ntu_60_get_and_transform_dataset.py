import os
from PIL import Image
import numpy as np
import torchvision
import re


def recreate_rgb_array_from_data_row(data_row, height=25):
    # recreate rgb_array from data_row(cifar style)
    rgb_array = data_row.reshape((3, height, -1))
    rgb_array = rgb_array.transpose(1, 2, 0)
    return rgb_array


def sort_nicely(l):
    """ Sort the given list alphanumeric (sort strings like that 1,2,...,11 not 1,11,...,2). https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)


# set activities_from_zero = True to get targets from 0..26 instead of 1..27
def get_target_from_filename(filename, activities_from_zero=True):
    pattern = r'A(\d+).skeleton'
    m = re.search(pattern, filename)
    target = -1
    if m:
        target = int(m.group(1))
        if activities_from_zero and target > 0:
            target = target-1
    else:
        print('Not possible to extract activity: file:', filename)
        target = filename
    return target


def resize_rgb_array_with_torch(rgb_array, desired_width, height=25):
    # print('resize')
    image = Image.fromarray(rgb_array, 'RGB')
    resized_img = torchvision.transforms.Resize((height, desired_width))(image)
    resized_rgb_array = np.asarray(resized_img)
    return resized_rgb_array


def create_data_as_in_cifar(rgb_array, desired_width=0, fill=True):
    resized_rgb_array = resize_rgb_array(rgb_array, desired_width, fill)
    r_array = resized_rgb_array[:, :, 0]
    g_array = resized_rgb_array[:, :, 1]
    b_array = resized_rgb_array[:, :, 2]
    rs = r_array.flatten()
    gs = g_array.flatten()
    bs = b_array.flatten()
    return np.concatenate((rs, gs, bs))


def resize_rgb_array(rgb_array, desired_width=0, fill_with_zeros=True, make_even_shape=False):
    r_array = rgb_array[:, :, 0]
    dimension_of_array = r_array.shape
    height = dimension_of_array[0]
    current_width = dimension_of_array[1]
    resized_rgb_array = rgb_array
    if make_even_shape and height % 4 != 0:
        to_fill = 4 - (height % 4)
        # add a black row at the end
        resized_rgb_array = np.pad(resized_rgb_array, ((0, to_fill), (0, 0), (0, 0)), 'constant', constant_values=0)
        height = height + to_fill
    if not (desired_width == 0):
        if make_even_shape and desired_width % 4 != 0:
            to_fill = height % 4
            desired_width = desired_width + to_fill
        if fill_with_zeros:
            if current_width < desired_width:
                # print('fill')
                # pad black pixel (0,0,0) at the end of each row
                resized_rgb_array = np.pad(resized_rgb_array, ((0, 0), (0, desired_width - current_width), (0, 0)),
                                           'constant', constant_values=0)
            else:
                # if image should be filled but is too big, resize with torch
                resized_rgb_array = resize_rgb_array_with_torch(resized_rgb_array, desired_width, height)
        else:
            # if not should be filled with zeros, resize with torch
            resized_rgb_array = resize_rgb_array_with_torch(rgb_array, desired_width, height)
    return resized_rgb_array


def images_to_np_array(image_paths, desired_width=83, fill_with_zeros=False, activities_from_zero=True, make_even_shape=False):
    # use if alpha in images is not always 255
    # from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    def pure_pil_alpha_to_color(image, color=(255, 255, 255)):
        """Alpha composite an RGBA Image with a specified color.
        Source: http://stackoverflow.com/a/9459208/284318
        Keyword Arguments:
        image -- PIL RGBA Image object
        color -- Tuple r, g, b (default 255, 255, 255)
        """
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    # desired_width = 0 means no resize
    # if fill_with-zeros true and image is smaller than desired width, fill with black pixels
    # else resize with torchvision
    def image_to_np_array(image_path):
        rgba_image = Image.open(image_path)
        rgba_array = np.asarray(rgba_image)
        # if alpha not always 255
        # rgb_image = pure_pil_alpha_to_color(rgba_image)
        # rgb_array = np.asarray(rgb_image)
        rgb_array = rgba_array[:, :, :3]
        resized_rgb_array = resize_rgb_array(rgb_array, desired_width, fill_with_zeros, make_even_shape)
        # get the target
        filename = os.path.basename(image_path)
        target = get_target_from_filename(filename, activities_from_zero)
        return (resized_rgb_array, target)

    (data, targets) = zip(*map(image_to_np_array, image_paths))
    # stack data (instead of array with 3d np.arrays, data becomes a 4d np.array)
    data = np.stack(data)
    return data, targets


def list_files(dir):
    filepaths = []
    filenames = []
    dimensions = []
    for root, dirs, files in os.walk(dir):
        # sort, because walk returns otherwise in arbitrary order
        sort_nicely(dirs)
        sort_nicely(files)
        for name in files:
            if name.endswith('.png'):
                filepath = os.path.join(root, name)
                filepaths.append(filepath)
                filenames.append(name)
                image = Image.open(filepath)
                dimensions.append(image.size)
    return filepaths, filenames, dimensions


# main function
def ntu60_get_all_data(path, desired_width=83, fill_with_zeros=False, activities_from_zero=True, make_even_shape=False):
    paths, names, dimensions = list_files(path)
    data, targets = images_to_np_array(paths, desired_width, fill_with_zeros, activities_from_zero, make_even_shape)
    return {'paths': paths, 'names': names, 'dimensions': dimensions, 'data': data, 'targets': targets}

