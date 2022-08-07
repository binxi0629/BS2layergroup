import random
import re, copy

import numpy as np
import json, os


def load_any_json(file_path) -> dict:
    with open(file_path, 'r') as jf:
        json_data = json.load(jf)

    return json_data


def data_statistic(json_data):

    # load formula
    formula = json_data["structure"]["formula"]

    # load uid
    uid = json_data['uid']

    # load space group
    spacegroup = json_data["structure"]["spacegroup"]
    spacegroup_num = json_data["structure"]["spacegroup_number"]

    # load atoms positions
    positions = np.array(json_data["structure"]["position"])

    # load atom types
    atom_types = json_data["structure"]["atom_types"]

    # load nonsoc bands
    is_sponpolarized = json_data["bands"]["nonsoc_energies"]["is_spinpolarized"] #0 or 1
    nonsoc_energies = np.array(json_data["bands"]["nonsoc_energies"]["bands"])
    kps_labels = json_data["bands"]["nonsoc_energies"]["koints_labels"]
    special_kps = json_data["bands"]["nonsoc_energies"]["special_points"]
    all_kps = np.array(json_data["bands"]["nonsoc_energies"]["kpath"]["kpoints"])

    # load soc bands
    soc_energies = np.array(json_data["bands"]["soc_energies"]["bands"])

    up_or_down, num_kps1, num_bands1,  = nonsoc_energies.shape
    num_kps, _ = all_kps.shape
    hkps = special_kps.keys()
    num_bands2, num_kps2 = soc_energies.shape

    print("spin up or down:", up_or_down)
    print("No. bands:", num_bands1)
    print("Spin polarized:", is_sponpolarized)
    print('No. kps in bands:', num_kps1)
    print("No. kps:", num_kps)
    print("No. high symmetry points:", hkps)
    print("No. soc bands:", num_bands2)
    print("No. soc kps:", num_kps2)


def get_num_kps(json_data):
    all_kps = np.array(json_data["bands"]["nonsoc_energies"]["kpath"]["kpoints"])
    num_kps, _ = all_kps.shape
    return num_kps


def get_is_spin_polarized(json_data):
    is_sponpolarized = json_data["bands"]["nonsoc_energies"]["is_spinpolarized"]
    return  is_sponpolarized


def test01():

    test_path = "../c2db_database_test/"
    count = 0
    for dirs, subdirs, files in os.walk(test_path):
        for file in files:
            json_data = load_any_json(file_path=os.path.join(test_path,file))
            count +=1
            print(f"File {file}:")
            data_statistic(json_data)
            print("----------------------------------------")


def count_kps_num():
    test_path = "../../database/c2db_database/"
    count = 0
    invalid_count = 0
    for dirs, subdirs, files in os.walk(test_path):
        total = len(files)
        for file in files:
            json_data = load_any_json(file_path=os.path.join(test_path, file))
            count += 1

            # num_kps=get_num_kps(json_data)
            is_spin_polarized = get_is_spin_polarized(json_data)
            if is_spin_polarized !=0 :
                # print(file)
                invalid_count +=1
            print(f"\r\tFinished: {count}| invalid count {invalid_count}|Total {total}", end='')

# test01()
# count_kps_num()


def test02():
    test_path = "../c2db_database_test/c2db_Ag2Br6-8e63ef562431.json"
    with open(test_path, 'r') as testf:
        json_data = json.load(testf)

    data_statistic(json_data)


# test02()
def get_json_from_path(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def labes2labe_list(labels:str):
    # labels = "GYHCH1XG"
    label_list = re.findall("\w\d|\w", labels)
    print(label_list)
    return label_list


def test03(file_name="test02.json"):
    json_data = get_json_from_path(file_name)

    from format_data_layergroup import LayerBands
    tmp_example = LayerBands(json_data,"t1")
    bands = tmp_example.spinup_bands # 400 x no. bands
    print(">>>",bands.shape)
    print(tmp_example.labels_idx)

    a = tmp_example.special_kps_separation(100)
    print(a)
    b = copy.deepcopy(a)

    LayerBands.kpath_shuffle(b)
    print(b)
    for i in range(len(a)):
        print(len(a[i]))

    paths = LayerBands.produceNewKpsIdxFromShuffledKpath(b, 3, shiftting_rate=0.3)
    for i in range(len(paths)):
        print("before shuffle:", paths[i])
        LayerBands.kpath_shuffle(paths[i])
        print("after shufffle:", paths[i])
        kps_list = LayerBands.unrollKpath(paths[i])
        print("len:", len(kps_list))
        extraced_bands = LayerBands.extractBandsByKpsIdx(bands.T,kps_list)
        print(extraced_bands.shape)

test03("test.json")
test03()


