import os
from tqdm import tqdm

import function_list


def prepare_inputs(numBands: int, load_from_dir: str, group_type:str, seed=None, num_upper_bound=500,
                   num_lower_bound=0, spin="spin up", numClasses=20):

    for required_dir in ["list/", "list/actual/", "list/guess/", "state_dicts/"]:
        if not os.path.exists(required_dir):
            os.mkdir(required_dir)
            print(f"made dir \"{required_dir}\"")
        else:
            print(f"dir \"{required_dir}\"")

    # prepare input data # (Do this every time dataset is changed)
    function_list.create_valid_list_file(
        num_bands=numBands,
        # TODO: modify to data path
        in_data_dir=load_from_dir,
        out_list_path="list/actual/valid_list.txt",
        num_upper_bound=num_upper_bound,
        num_lower_bound=num_lower_bound,
        spin=spin,
        valid_classes=numClasses
    )

    # prepare actual data # (Do this every time dataset is changed)
    function_list.create_actual_anygroup_list_files(
        in_list_path="list/actual/valid_list.txt",
        out_list_path_format="list/actual/{}".format(group_type)+"_list_{}.txt",
        num_classes=numClasses
    )

    # function_list.create_actual_crystal_list_files(
    #     in_list_path="list/actual/valid_list.txt",
    #     out_list_path_format="list/actual/crystal_list_{}.txt"
    # )

    # prepare training dataset and testing dataset based on each class
    in_list_path = []
    # Cubic
    # [in_list_path.append(f"list/actual/{group_type}_list_{i}.txt") for i in [3,4,5,6,7]
    [in_list_path.append(f"list/actual/{group_type}_list_{i}.txt") for i in range(numClasses)]
    function_list.create_train_and_test_anygroup_file(in_list_path=in_list_path,
                                                      out_training_path="list/actual/train_set.txt",
                                                      out_test_path="list/actual/test_set.txt")


def test():
    prepare_inputs(numBands=60,
                   load_from_dir="../../c2db_database02_output_degeneracy/",
                #    load_from_dir="../../c2db_database02_output/",
                   group_type="layers_num",
                   num_upper_bound=1000,
                   num_lower_bound=0,
                   numClasses=20,
                   seed=None)
# test()


if __name__ == '__main__':

    # TODO: load parameters from config.py, modify parameters there for your own use

    from config import args
    if args["load"]["start"]:
        cfg = args["load"]
        prepare_inputs(numBands=cfg["numBands"],
                       load_from_dir=cfg["load_from_dir"],
                       group_type=cfg["group_type"],
                       numClasses=cfg["numClasses"],
                       num_upper_bound=cfg["num_upper_bound"],
                       num_lower_bound=cfg["num_lower_bound"],
                       seed=cfg["seed"])

    # test()