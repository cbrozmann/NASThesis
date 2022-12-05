import bz2
import pickle
import _pickle as cPickle
import numpy as np
from PIL import Image
from nats_bench.api_utils import pickle_load
import re
import os
import matplotlib.pyplot as plt
import json
import timeit

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# safe numpy arrays in lists
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    return dict


def get_index_from_name(name):
    pattern = r'@(\d+)'
    m = re.search(pattern, name)
    if m:
        return int(m.group(1))
    else:
        print('Not possible to extract number from', name)
        return None


def get_validation_and_test_acc(current_data, validation_results_arr, test_results_arr):
    for key in current_data:
        if 'valid' in key:
            idx = get_index_from_name(key)
            validation_results_arr[idx] = current_data[key]
        elif 'ori-test' in key:
            idx = get_index_from_name(key)
            test_results_arr[idx] = current_data[key]
        else:
            print(f'Following key was missing:', key)


def get_train_acc(current_data, train_result_arr):
    for key in current_data:
        ind = int(key)
        train_result_arr[ind] = current_data[key]


def get_max_value_and_index(np_arr):
    max_val_value = np.amax(np_arr, axis=None, out=None)
    max_val_idx = np.where(np_arr == max_val_value)[0]
    return max_val_value, max_val_idx


def get_dict_for_one_architecture(data):
    data_dict = {}
    for number_epochs_str in data:
        for dataset_name_and_seed in data[number_epochs_str]['all_results']:
            dataset_name, seed = dataset_name_and_seed
            if dataset_name.endswith('-valid'):
                clear_dataset_name = dataset_name[:-6]
            else:
                clear_dataset_name = dataset_name
            if not clear_dataset_name in data_dict:
                data_dict[clear_dataset_name] = {}
            if not number_epochs_str in data_dict[clear_dataset_name]:
                data_dict[clear_dataset_name][number_epochs_str] = {}
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]={}
            number_epochs = int(number_epochs_str)
            # get @1 and @5 accuracies
            validation_acc1es = np.zeros(number_epochs)
            test_acc1es = np.zeros(number_epochs)
            train_acc1es = np.zeros(number_epochs)
            validation_acc5es = np.zeros(number_epochs)
            test_acc5es = np.zeros(number_epochs)
            train_acc5es = np.zeros(number_epochs)
            eval = 'eval_acc1es'
            current_data = data[number_epochs_str]['all_results'][dataset_name_and_seed][eval]
            get_validation_and_test_acc(current_data, validation_acc1es, test_acc1es)
            eval = 'eval_acc5es'
            current_data = data[number_epochs_str]['all_results'][dataset_name_and_seed][eval]
            get_validation_and_test_acc(current_data, validation_acc5es, test_acc5es)

            current_data = data[number_epochs_str]['all_results'][dataset_name_and_seed]['train_acc1es']
            get_train_acc(current_data, train_acc1es)
            current_data = data[number_epochs_str]['all_results'][dataset_name_and_seed]['train_acc5es']
            get_train_acc(current_data, train_acc5es)
            # save accuracies in the dict
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['val_acc1es'] = validation_acc1es
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['val_acc5es'] = validation_acc5es
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['test_acc1es'] = test_acc1es
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['test_acc5es'] = test_acc5es
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['train_acc1es'] = train_acc1es
            data_dict[clear_dataset_name][number_epochs_str][str(seed)]['train_acc5es'] = train_acc5es
            # save all additional information in the dict
            if 'additional_information' not in data_dict[clear_dataset_name][number_epochs_str]:
                data_dict[clear_dataset_name][number_epochs_str]['additional_information'] = {}
                for add_info in data[number_epochs_str]:
                    if add_info != 'all_results':
                        data_dict[clear_dataset_name][number_epochs_str]['additional_information'][add_info] = \
                            data[number_epochs_str][add_info]
                for add_info in data[number_epochs_str]['all_results'][dataset_name_and_seed]:
                    if add_info not in ['eval_acc1es', 'eval_acc5es', 'train_acc1es', 'train_acc5es']:
                        data_dict[clear_dataset_name][number_epochs_str]['additional_information'][add_info] = \
                            data[number_epochs_str]['all_results'][dataset_name_and_seed][add_info]
    return data_dict


def get_data_from_filepath(path):
    return decompress_pickle(path)


def sort_nicely(l):
    """ Sort the given list alphanumeric (sort strings like that 1,2,...,11 not 1,11,...,2). https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)


def list_trained_data_paths(dir):
    filepaths = []
    filenames = []
    for root, dirs, files in os.walk(dir):
        # sort, because walk returns otherwise in arbitrary order
        sort_nicely(dirs)
        sort_nicely(files)
        for name in files:
            if re.match(r'\d+.pickle.pbz2', name):
                filepath = os.path.join(root, name)
                filepaths.append(filepath)
                filenames.append(name)
    return filepaths, filenames

# path = 'D:/Desktop/MA/download_server/finished/own/utd_mhad/train/NATS-Bench-topology/NATS-tss-v1_0-7ce23-simple'
# path = 'D:/Desktop/MA/download_server/finished/own/ntu/NATS-Bench-topology/NATS-tss-v1_0-03485-simple'
path = 'C:/Users/Brozm/Desktop/utd_mhad/train/NATS-Bench-topology/NATS-tss-v1_0-7ce23-simple'

paths, names = list_trained_data_paths(path)

# safe all data in a list
all_data = []
for filename in paths:
    data = decompress_pickle(filename)
    data_dict = get_dict_for_one_architecture(data)
    all_data.append(data_dict)

with open('utdmhad_preprocessed_data.json', 'w') as f:
    json.dump(all_data, f, cls=NumpyEncoder)

if "utdmhad" in path:
    with open('utdmhad_preprocessed_data.json', 'w') as f:
        json.dump(all_data, f, cls=NumpyEncoder)
elif "ntu" in path:
    with open('ntu_preprocessed_data.json', 'w') as f:
        json.dump(all_data, f, cls=NumpyEncoder)
else:
    with open('preprocessed_data.json', 'w') as f:
        json.dump(all_data, f, cls=NumpyEncoder)

# open for example with
# open('utdmhad_preprocessed_data.json', 'r') as f:
#     all_data = json.load(f)