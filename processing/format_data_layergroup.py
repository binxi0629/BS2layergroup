import layergroup

import numpy as np
import os, json



class LayerBands:

    _root_save_path = "../../input_data/energy_separation01/"
    def __init__(self, jsonbands, file_path):


        if jsonbands is not None:
            self.json_data = jsonbands
        else:
            raise ValueError("Empty bands")

        self.uid = self.json_data['uid']
        self.path = file_path

        self.formula = self.json_data["structure"]["formula"]
        self.atoms_type = self.json_data["structure"]["atom_types"]
        self.positions = np.array(self.json_data["structure"]["position"])
        self.spacegroup = self.json_data["structure"]["spacegroup"]
        self.spacegroup_num = self.json_data["structure"]["spacegroup_number"]
        self.lattice = self.json_data["structure"]["lattice"]
        self.layergroup_num = layergroup.exhasutive_search_layergroup(self.positions, self.atoms_type)

        self.kps = np.array(self.json_data["bands"]["nonsoc_energies"]["kpath"]["kpoints"])
        self.num_all_kps, _ = self.kps.shape
        self.kps_labels = self.json_data["bands"]["nonsoc_energies"]["koints_labels"]

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

    #TODO: Set different number of kps here
    def get_num_kps(self):
        return self.num_all_kps

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
            else:
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
