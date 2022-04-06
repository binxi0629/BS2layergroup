import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.data import Dataset


class AnyDataset(Dataset):

    def __init__(self, in_list_path, json2inputlabel, numClasses, bands_type="spin up", training=False):
        self.data_inputs = []
        self.data_labels = []
        desc = "train set" if training else "test set"

        if os.stat(in_list_path).st_size == 0:
            raise OSError("list is empty")

        file_names = np.loadtxt(in_list_path, "U120", ndmin=1)

        for i in tqdm(range(len(file_names)), desc=f"Loading {desc}"):
            file_name = file_names[i]
            with open(file_name, "r") as file:
                data_json = json.load(file)

            data_input_np, data_label_np = json2inputlabel(data_json, bands_type)
            # data_label_np_onehot = np.array(tf.one_hot(data_label_np, numClasses)[0])
            self.data_inputs.append(data_input_np)
            self.data_labels.append(data_label_np)
            # self.data_labels.append(data_label_np_onehot)
            # print("\r\tload: {}/{}".format(i, len(file_names)), end="")
        # print("\rload: {}".format(len(file_names)))

        # uncomment
        # self.data_labels = self.data_labels.squeeze()
        # self.show_number_labels()
        self.len = len(self.data_inputs)

    def _inputs(self):
        # FIXME: abstract method from tf.data.Dataset, rewrite it if necessary
        pass

    def element_spec(self):
        # FIXME: abstract method from tf.data.Dataset, rewrite it if necessary
        pass

    def np_one_hot(self, numClasses):
        return np.array(tf.one_hot(self, numClasses)[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_inputs[index], self.data_labels[index]

    def show_number_labels(self):
        tmp = []
        total_size = len(self.data_labels)
        for i in self.data_labels:
            if i in tmp:
                pass
            else:
                tmp.append(i)

        print(len(tmp))


#
# def get_validate_train_loader(dataset, batch_size, validate_size):
#     num_train = len(dataset)
#     indices = list(range(num_train))
#     split = int(validate_size * num_train)
#
#     validate_sampler = SubsetRandomSampler(indices[:split])
#     train_sampler = SubsetRandomSampler(indices[split:])
#
#     validate_loader = DataLoader(dataset, batch_size=batch_size, sampler=validate_sampler)
#     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#
#     return validate_loader, train_loader


# def test():
#     def json2inputlabel(data_json, bands_type="spin up"):
#         _bands_type = {"spin up": "spin_up_bands",
#                        "spin down": "spin_down_bands",
#                        "soc": "soc_bands"}
#         data_input_np = np.array(data_json[_bands_type[bands_type]]).flatten().T  # 4000 x 1
#         data_label_np = np.array([data_json["new_label"]])
#         return data_input_np, data_label_np

#     file_name = "test.json"
#     with open(file_name, 'r') as f:
#         data = json.load(f)
#     bands, label = json2inputlabel(data)
#     print(bands.shape)
#     print(label)

# test()
