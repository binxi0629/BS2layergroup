import layergroup

import numpy as np
import os, json, re
import random

from copy import copy,deepcopy

class LayerBands:

    _root_save_path = "../../input_data/energy_separation01/"

    def __init__(self, jsonbands, file_path):

        if jsonbands is not None:
            self.json_data = jsonbands
        else:
            raise ValueError("Empty bands")

        self.uid = self.json_data['uid']
        self.path = file_path
        self.agumented_num=0

        self.formula = self.json_data["structure"]["formula"]
        self.atoms_type = self.json_data["structure"]["atom_types"]
        self.positions = np.array(self.json_data["structure"]["position"])

        # need test
        # z_position = np.array([atom_position[2] for atom_position in self.positions])
        # z_position = z_position.round(decimals=2)
        z_position = np.array([atom_position[2] for atom_position in self.positions]).round(decimals=2)
        z_counts = len(np.unique(z_position, return_counts=True)[0])

        self.layers_num = z_counts

        # x_position = np.array([atom_position[0] for atom_position in self.positions]).round(decimals=2)
        # x_counts = len(np.unique(x_position, return_counts=True)[0])
        # y_position = np.array([atom_position[1] for atom_position in self.positions]).round(decimals=2)
        # y_counts = len(np.unique(y_position, return_counts=True)[0])
        # z_position = np.array([atom_position[2] for atom_position in self.positions]).round(decimals=2)
        # z_counts = len(np.unique(z_position, return_counts=True)[0])
        # layers_num = min(x_counts, y_counts, z_counts)
        # layer_direction = "NA"
        # if x_counts == layers_num:
        #     layer_direction = "a"
        # if y_counts == layers_num:
        #     layer_direction = "b"
        # if z_counts == layers_num:
        #     layer_direction = "c"

        # self.z_layers_num = z_counts
        # self.layers_num = layers_num
        # self.layer_direction = layer_direction

        #######

        self.spacegroup = self.json_data["structure"]["spacegroup"]
        self.spacegroup_num = self.json_data["structure"]["spacegroup_number"]
        self.lattice = self.json_data["structure"]["lattice"]
        self.layergroup_num = layergroup.exhasutive_search_layergroup(self.positions, self.atoms_type)

        self.kps = np.array(self.json_data["bands"]["nonsoc_energies"]["kpath"]["kpoints"])
        self.num_all_kps, _ = self.kps.shape
        self.kps_labels = self.json_data["bands"]["nonsoc_energies"]["koints_labels"]
        self.special_kps_list = self.json_data["bands"]["nonsoc_energies"]["special_points"]
        self.labels_idx = self.position_high_symmetry_kps()

        self.is_spin_polarized = self.json_data["bands"]["nonsoc_energies"]["is_spinpolarized"]
        self.nonsoc_energies = np.array(self.json_data["bands"]["nonsoc_energies"]["bands"])
        self.spinup_bands, self.spindown_bands = self.get_bands()

        self.soc_energies = np.array(self.json_data["bands"]["soc_energies"]["bands"])

    def get_bands(self):
        """
        For non-spin polarized calculation (is_spin_polarized = 1), it's assumed that spin up and down are indistinguishable, but because there
        are spin polarized calculations, so spin up bands and spin down bands are all nonsoc_energies

        For spin polarized calculation (is_spin_polarized = 1), just return spin up bands and spin down bands
        :return:  spin up and spin down
        """
        spin_up_bands = self.nonsoc_energies[0, :, :]
        if self.is_spin_polarized == 1:
            spin_down_bands = self.nonsoc_energies[1,:,:]
        else:
            spin_down_bands = spin_up_bands
        return spin_up_bands, spin_down_bands

    # TODO: Set different number of kps here
    def get_num_kps(self):
        return self.num_all_kps

    def position_high_symmetry_kps(self):

        torlence = 1e-6
        labels_idx = []

        def labels2label_list(labels:str):
            label_list = re.findall("\w\d|\w", labels)
            return label_list

        label_list = labels2label_list(self.kps_labels)

        for each in label_list:
            this_label = [each]
            k_pos = np.array([self.special_kps_list[each]["__ndarray__"][2]])

            tmp_list = np.array(np.absolute([self.kps - k_pos]) <= torlence)
            tmp_list = np.reshape(tmp_list, (400, 3))
            tmp_list = np.sum(tmp_list, axis=1, keepdims=False)

            for idx, i in enumerate(tmp_list):
                if i == 3:
                    this_label.append(idx)

            labels_idx.append(this_label)
            del this_label

        start_idx = 0
        paths = []
        for i in range(len(labels_idx)):
            for j in range(len(labels_idx[i])):
                if j != 0:
                    if labels_idx[i][j] >= start_idx:
                        start_idx = labels_idx[i][j]
                        paths.append([labels_idx[i][0], start_idx])
                        break

        return paths

    def special_kps_separation(self, num_kps, strategy=None):
        """
            Shrink k points dimension from 400 to num_kps
        """
        num_paths = len(self.labels_idx) - 1

        def equalInervalIdxSplitting(start_idx:int, end_idx:int,num_idx:int):
            """
                Equal interval shrinking: int: (end_idx-start_idx-1)/num_idx
            """
            interval = end_idx-start_idx-1
            idx_list = []

            if interval<num_idx-2:
                raise IndexError(f"Index received max number: {interval+1}, but got {num_idx}")
            elif 0 == interval:
                return [start_idx]
            else:
                itv = int(interval/num_idx)
                for i in range(num_idx-1):
                    idx_list.append(start_idx+itv*i)
                idx_list.append(end_idx)

            return idx_list

        def default_strategy(required_num, total=400):
            """
                Weighted shrinking for each path
            """
            tmp = []
            tmp_sum = 0
            no_path_cases = 0

            for k in range(num_paths - 1):
                tmp_val = int(required_num * (self.labels_idx[k + 1][1] - self.labels_idx[k][1]) / total)
                if 0 == tmp_val:
                    no_path_cases += 1
                tmp.append(tmp_val)
                tmp_sum += tmp_val

            tmp.append(required_num - tmp_sum)

            for _ in range(no_path_cases):
                tmp[tmp.index(max(tmp))] -= 1

            return tmp

        kps_idx = []

        num_idx_list=default_strategy(num_kps)

        for j in range(num_paths):
            kps_idx.append(equalInervalIdxSplitting(start_idx=self.labels_idx[j][1], end_idx=self.labels_idx[j+1][1], num_idx=num_idx_list[j]))

        # include the last index
        # kps_idx+=[self.labels_idx[num_paths][1]]

        return kps_idx

    @staticmethod
    def kpath_shuffle(kpath):
        """
            Need deepcopy first
        """
        return random.shuffle(kpath)

    # FIXME: data augmentation, still under testing
    @staticmethod
    def genShifttingChoice(kpaths, shiftting_rate):
        num_paths = len(kpaths)
        criteria_list = []

        for i in range(num_paths):
            num = int(shiftting_rate *len(kpaths[i]))
            # print("nunm:",num)
            criteria_list.append([-num, num])

        def genNrandomInts(num_rands, crteria):
            choice = []
            for i in range(num_rands):
                choice.append(random.randint(crteria[i][0], crteria[i][1]))
            return choice

        choice = genNrandomInts(num_paths - 1, criteria_list)

        target_list = []

        for i in range(criteria_list[-1][0], criteria_list[-1][1]):
            target_list.append(i)

        while -1 * sum(choice) not in target_list:
            choice = genNrandomInts(num_paths - 1, criteria_list)

        choice += [-1 * sum(choice)]
        return choice

    @staticmethod
    def appendRemoveList(target_list, num,add=False):
        list_temp = deepcopy(target_list)
        max_idx = len(list_temp)
        # print(">>>",max_idx)
        count = 0
        if add:
            while count != num:
                idx = random.randint(1,max_idx-2)
                if list_temp[idx]-list_temp[idx-1] >=2:
                    val = list_temp[idx-1]+int((list_temp[idx+1]-list_temp[idx])/2)
                    list_temp.insert(idx,val)
                    max_idx += 1
                    count += 1
        else:
            while count != num:
                if max_idx > 1:
                    list_temp.pop(random.randint(1,max_idx-2))
                    max_idx -= 1
                    count += 1
        # print(list_temp)
        return list_temp

    @staticmethod
    def produceNewKpsIdxFromShuffledKpath(kpaths, num_examples, shiftting_rate=0.4):
        n_paths = []
        for each in range(num_examples):
            list_a = deepcopy(kpaths)
            new_kpaths = []
            choice = LayerBands.genShifttingChoice(kpaths, shiftting_rate)
            # print(choice)
            for idx, i in enumerate(choice):
                if 0 == i:
                    new_kpaths.append(list_a[idx])
                    # print(0)
                elif i < 0:
                    # print(list_a[idx][-1]-list_a[idx][0])
                    new_kpaths.append(LayerBands.appendRemoveList(list_a[idx], -i, add=False))
                    # print(-1)
                else:
                    new_kpaths.append(LayerBands.appendRemoveList(list_a[idx], i, add=True))
                    # print(1)
            n_paths.append(new_kpaths)
            del list_a, new_kpaths

        return n_paths

    @staticmethod
    def unrollKpath(kpaths):
        flat_list = [item for sublist in kpaths for item in sublist]
        return flat_list

    @staticmethod
    def extractBandsByKpsIdx(formatted_bands,kps_list):
        formatted_bands=np.array(formatted_bands)
        extracted_bands = formatted_bands[:,kps_list]
        return extracted_bands


    @staticmethod
    def energy_separation(formatted_bands, padded_number=0):
        tmp = formatted_bands
        size = np.shape(tmp)
        # print(size)
        separation_bands = np.zeros((size[0] - 1, size[1])) + padded_number  # -1: difference

        for i in range(size[1]):
            each_column = []
            for j in range(size[0] - 1):
                energy = tmp[j + 1][i] - tmp[j][i]
                each_column.append(energy)

            separation_bands[:, i] = np.array(each_column)

        return separation_bands

    @staticmethod
    def degen_translate(formatted_bands, en_tolerance=0.001):

        """
            This method is to represent the bands matrix into a degeneracy form
        :param formatted_bands: one of the output from format_data() method
        :param en_tolerance: energy tolerance, default 0.01eV
        :return: the formatted degenerate bands

        """
        tmp = formatted_bands
        size = np.shape(tmp)

        # Here we have two assumptions:
        # Assumption 1: missing data are padded zero, which we assume they are not existed or missed
        # Assumption 2: missing data are padded one, which we assume they are non-degenerate
        # Comment: assumption 2 is more physical, but assumption 1 gives a better prediction result
        # Why this: because bands structure are calculated based on a high symmetry path, the bands missing at certain
        #           k points don't mean these bands are not existed

        # TODO: if you want assumption 2, comment the assumption 1, and uncomment assumption 2
        degen_bands = np.zeros(size)  # assumption 1: missing data are assumed null, labeled by 0
        # degen_bands = np.zeros(size)+1  # assumption 2: missing data are assumed non-degenerate, labeled by 1

        # Degeneracy analysis
        for i in range(size[1]):
            each_column = []
            count = 1
            for j in range(size[0] - 1):
                if tmp[j][i] == 0:
                    count = 0
                    break
                else:
                    if np.absolute(tmp[j + 1][i] - tmp[j][i]) <= en_tolerance:
                        count += 1
                    else:
                        for k in range(count):
                            each_column.append(count)
                        count = 1

            if count == 0:
                pass
            else:
                for k in range(count):
                    each_column.append(count)
                degen_bands[:, i] = np.array(each_column)

        return degen_bands

    @staticmethod
    def find_fermi_index(formatted_bands):
        tmp =formatted_bands
        # count bands number
        bands_num = np.shape(tmp)[0]

        for j in range(bands_num-1):
            if (tmp[j][0]) * (tmp[j + 1][0]) <= 0:
                fermi_index = j
                # print(j)
                return fermi_index
        return bands_num

    def vb_count(self, formatted_bands, fermi_index: int):
        return fermi_index + 1

    def cb_count(self, formatted_bands, fermi_index: int):
        tmp = formatted_bands
        bands_num = np.shape(tmp)[0]
        return bands_num-fermi_index-1

    def padding_judgement(self,
                          conduction_num,
                          valence_num,
                          num_bands,
                          bands_below_fermi_limit):

        if valence_num < bands_below_fermi_limit:
            padding_btm = True
        else:
            padding_btm = False

        if conduction_num < num_bands-bands_below_fermi_limit:
            padding_top = True
        else:
            padding_top = False

        return padding_btm, padding_top

    def padding_around_fermi(self,
                             formatted_bands,
                             num_of_bands,
                             bands_below_fermi_limit,
                             padding_num=0,
                             fermi_index=0):

        # bands padding around fermi level
        tmp = formatted_bands
        bands_num = np.shape(tmp)[0]
        row_dim = np.shape(tmp)[1]

        padding_vector = []
        [padding_vector.append(padding_num) for num in range(row_dim)]

        conduction_bands_num = self.cb_count(formatted_bands, fermi_index=fermi_index)
        # print(conduction_bands_num)
        valence_bands_num = self.vb_count(formatted_bands, fermi_index=fermi_index)
        # print(valence_bands_num)
        padding_btm, padding_top = self.padding_judgement(conduction_bands_num,
                                                          valence_bands_num,
                                                          num_of_bands,
                                                          bands_below_fermi_limit=bands_below_fermi_limit)

        valence_bands = tmp[0:fermi_index+1, :]
        if not padding_btm:
            btm_bands = tmp[fermi_index+1-bands_below_fermi_limit:fermi_index+1, :]
            # print(np.shape(btm_bands))
        else:
            padded_btm_bands = []
            btm_num = bands_below_fermi_limit-valence_bands_num
            [padded_btm_bands.append(padding_vector) for num in range(btm_num)]
            padded_btm_bands = np.array(padded_btm_bands)
            btm_bands = np.concatenate((padded_btm_bands, valence_bands), axis=0)
            # print('btm_bands_num', len(btm_bands))
            # print('padded_bands_num', len(padded_btm_bands))
            # print('vb_num', len(valence_bands))

        conduction_bands = tmp[fermi_index+1:, :]
        if not padding_top:
            top_bands = tmp[fermi_index+1:fermi_index+num_of_bands-bands_below_fermi_limit+1, :]
            # print(np.shape(top_bands))
        else:
            padded_top_bands = []
            top_num = num_of_bands-bands_below_fermi_limit-conduction_bands_num
            [padded_top_bands.append(padding_vector) for num in range(top_num)]
            padded_top_bands = np.array(padded_top_bands)
            top_bands = np.concatenate((conduction_bands, padded_top_bands), axis=0)
            # print('top_bands_num', len(top_bands))
            # print('padded_bands_num', len(padded_top_bands))
            # print('cb_num', len(conduction_bands))

        padded_bands = np.concatenate((btm_bands, top_bands), axis=0)

        # print(len(padded_bands))
        if len(padded_bands) != num_of_bands:
            raise Exception(f"Bands number not {num_of_bands}")

        return padded_bands


class NumpyEncoder(json.JSONEncoder):
    """
        to solve Error: NumPy array is not JSON serializable
        see: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
