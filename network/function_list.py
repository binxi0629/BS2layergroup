import os
import json

import numpy as np
import tqdm

import crystal

def create_valid_list_file(num_bands, in_data_dir, out_list_path,
                           seed=None, num_upper_bound=500, num_lower_bound=0,
                           spin="spin up", valid_classes=20):

    _bands_type={"spin up":"spin_up_bands",
                 "spin down": "spin_down_bands",
                 "soc": "soc_bands"}

    bands_loc = _bands_type[spin]

    print("\tcreate valid list:", end="")
    valid_file_names = []
    count_list = []
    for count in range(valid_classes):
        count_list.append(0)

    IndexErrorCount = 0
    for root, dirs, file_names in os.walk(in_data_dir):  # loop through file names in a directory
        for i, file_name in enumerate(file_names):

            if ".json" in file_name:
                with open(os.path.join(in_data_dir,file_name), "r") as file:
                    data_json = json.load(file)

                # (No.bands x No.kps) accept only data with certain number of bands
                if np.array(data_json[bands_loc]).shape[0] != num_bands:
                    continue

                label = data_json["new_label"]
                # Set uppper and lower bound to load data
                try:
                    # count_list[label-2] += 1
                    count_list[label - 1] += 1
                except IndexError:
                    print(f"IndexError {file_name}")
                    IndexErrorCount+=1
                    continue

                if count_list[label - 1] > num_upper_bound:
                # if count_list[label - 2] > num_upper_bound:
                    # print (f"count_list[label - 1] > num_upper_bound    file_name: {file_name}")
                    continue
                if count_list[label - 1] < num_lower_bound:
                # if count_list[label - 2] < num_lower_bound:
                    continue

                valid_file_names.append(file_name)
                print("\r\tcreate valid list: {}/{}".format(i, len(file_names)), end="")

    print(f"\nIndexErrorCount {IndexErrorCount}")

    # random shuffle dataset
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(valid_file_names)  # randomize order of data
    with open(out_list_path, "w") as file_out:
        for file_name in valid_file_names:
            file_out.write(in_data_dir + file_name + "\n")  # write data_file_paths
    print("\rcreate valid list: {}".format(len(open(out_list_path).readlines())))

# def create_valid_list_file(num_bands, in_data_dir, out_list_path,
#                            seed=None, num_upper_bound=500, num_lower_bound=0,
#                            spin="spin up", valid_classes=20):

#     _bands_type={"spin up":"spin_up_bands",
#                  "spin down": "spin_down_bands",
#                  "soc": "soc_bands"}

#     bands_loc = _bands_type[spin]

#     print("\tcreate valid list:", end="")
#     valid_file_names = []
#     count_list = []
#     for count in range(valid_classes):
#         count_list.append(0)

#     IndexErrorCount = 0
#     for root, dirs, file_names in os.walk(in_data_dir):  # loop through file names in a directory
#         for i, file_name in enumerate(file_names):

#             if ".json" in file_name:
#                 with open(os.path.join(in_data_dir,file_name), "r") as file:
#                     data_json = json.load(file)

#                 # (No.bands x No.kps) accept only data with certain number of bands
#                 if np.array(data_json[bands_loc]).shape[0] != num_bands:
#                     continue

#                 label = data_json["new_label"]
#                 # Set uppper and lower bound to load data
#                 try:
#                     count_list[label-2] += 1
#                     # count_list[label - 1] += 1
#                 except IndexError:
#                     print(f"IndexError {file_name}")
#                     IndexErrorCount+=1
#                     continue

#                 # if count_list[label - 1] > num_upper_bound:
#                 if count_list[label-2] > num_upper_bound:
#                     # print (f"count_list[label - 1] > num_upper_bound    file_name: {file_name}")
#                     continue
#                 if count_list[label-2] < num_lower_bound:
#                 # if count_list[label - 1] < num_lower_bound:
#                     continue

#                 valid_file_names.append(file_name)
#                 print("\r\tcreate valid list: {}/{}".format(i, len(file_names)), end="")

#     print(f"\nIndexErrorCount {IndexErrorCount}")

#     # random shuffle dataset
#     if seed is not None:
#         np.random.seed(seed)
#     np.random.shuffle(valid_file_names)  # randomize order of data
#     with open(out_list_path, "w") as file_out:
#         for file_name in valid_file_names:
#             file_out.write(in_data_dir + file_name + "\n")  # write data_file_paths
#     print("\rcreate valid list: {}".format(len(open(out_list_path).readlines())))


def create_empty_list_files(out_num_group, out_list_path_format):
    for i in range(out_num_group):
        open(out_list_path_format.format(i), "w").close()


def create_actual_anygroup_list_files(in_list_path, out_list_path_format,num_classes:int, seed=None):

    create_empty_list_files(num_classes, out_list_path_format)  # empty files for appending
    file_paths = np.loadtxt(in_list_path, "U120") # characters limit

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(file_paths)  # randomize order of data

    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            data_json = json.load(file)
        
        # label = data_json["layers_num"]
        label = data_json["new_label"]

        with open(out_list_path_format.format(label), "a") as file_out:
            file_out.write(file_path + "\n")
        print("\r\tcreate actual list: {}/{}".format(i, len(file_paths)), end="")

    print("\rcreate actual list: {}".format(len(file_paths)))


# 7 crystal systems
def create_actual_crystal_list_files(in_list_path, out_list_path_format, seed=None):
    create_empty_list_files(7, out_list_path_format)
    file_paths = np.loadtxt(in_list_path, "U120")
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(file_paths)  # randomize order of data
    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            data_json = json.load(file)
        csnum = crystal.crystal_number(data_json["number"])
        with open(out_list_path_format.format(csnum), "a") as file_out:
            file_out.write(file_path + "\n")
        print("\r\tcreate actual list: {}/{}".format(i, len(file_paths)), end="")
    print("\rcreate actual list: {}".format(len(file_paths)))


def create_train_and_test_anygroup_file(in_list_path, out_training_path="list/actual/train_set.txt",
                                        out_test_path="list/actual/test_set.txt", seed=None, validate_size=0.1):

    # create required files
    open(out_training_path, "w").close()
    open(out_test_path, "w").close()

    file_paths = in_list_path
    # file_paths = np.loadtxt(in_list_path, "U120")
    if seed is not None:
        np.random.seed(seed)

    train_list = []
    test_list = []
    for i, file_path in enumerate(file_paths):
        this_list = np.loadtxt(file_path, "U120")
        try:
            length = len(this_list)
        except TypeError:
            this_list=[this_list]

        np.random.shuffle(this_list)  # randomize order of data
        tmp_train_set = this_list[:int((1-validate_size)*length)]
        tmp_test_set = this_list[int((1-validate_size)*length):]

        for each_training in tmp_train_set:
            train_list.append(each_training)

        for each_testing in tmp_test_set:
            test_list.append(each_testing)

    # print(len(training_list))
    # print(testing_list)
    # print(len(testing_list))

    with open(out_training_path, "w") as file_out:
        for file_name in train_list:
            file_out.write("%s\n" % file_name)  # write data_file_paths
    print("\rcreate valid train list: {}".format(len(open(out_training_path).readlines())))

    with open(out_test_path, "w") as file_out:
        for file_name in test_list:
            file_out.write("%s\n" % file_name)  # write data_file_paths
    print("\rcreate valid test list: {}".format(len(open(out_test_path).readlines())))


# def create_guess_list_files(device, model, hs_indices, num_group, split, in_list_path, out_list_path_format):
#     create_empty_list_files(num_group, out_list_path_format)
#     file_paths = np.loadtxt(in_list_path, "U90")[:split]
#     for i, file_path in enumerate(file_paths):
#         with open(file_path, "r") as file:
#             data_json = json.load(file)
#         data_input_np = np.array(data_json["bands"])
#         #TODO: FCNN
#         data_input_np = data_input_np[:, hs_indices].flatten().T
#         # data_input_np = data_input_np[None, None,:]
#         data_input = torch.from_numpy(data_input_np).float()
#         output = model(data_input.to(device))  # feed through the neural network
#         sgnum = torch.max(output, 0)[1].item() + 1  # predicted with the most confidence
#         with open(out_list_path_format.format(sgnum), "a") as file_out:
#             file_out.write(file_path + "\n")
#         print("\r\tcreate guess list: {}/{}".format(i, len(file_paths)), end="")
#     print("\rcreate guess list: {}".format(len(file_paths)))

# def append_guess_spacegroup_in_crystal_list_files(device, model, csnum, hs_indices, split,
#                                                   in_list_path, out_list_path_format):
#     if os.stat(in_list_path).st_size == 0:
#         return
#     file_paths = np.loadtxt(in_list_path, "U60")[:split]
#     for i, file_path in enumerate(file_paths):
#         with open(file_path, "r") as file:
#             data_json = json.load(file)
#         data_input_np = np.array(data_json["bands"])
#         #TODO:FCNN
#         # data_input_np = data_input_np[None, None, :]
#         data_input_np = data_input_np[:, hs_indices].flatten().T
#         data_input = torch.from_numpy(data_input_np).float()
#         output = model(data_input.to(device))
#         sgnum = torch.max(output, 0)[1].item() + 1 + crystal.spacegroup_index_lower(csnum)  # sgnum = output + lower(cs)
#         if sgnum not in crystal.spacegroup_number_range(csnum):
#             print("\r\tcreate guess list: {}/{}".format(i, len(file_paths)), end="")
#             continue
#         with open(out_list_path_format.format(sgnum), "a") as file_out:
#             file_out.write(file_path + "\n")
#         print("\r\tcreate guess list: {}/{}".format(i, len(file_paths)), end="")
#     print("\rcreate guess list: {}".format(len(file_paths)))