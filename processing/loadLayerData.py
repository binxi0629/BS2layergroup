import format_data_layergroup
import os, re, json
import numpy as np

from tqdm import tqdm


def layergroup_filter(layergroup_lower_bound=-1):
    # TODO: hard code the counting result of layergroup
    with open("lg_distribution.json", 'r') as f:
        dis=json.load(f)
    layergroup_stat = dis["lg_distribution"]

    layergroup_list=[]
    for index, i in enumerate(layergroup_stat):
        if i >= layergroup_lower_bound:
            layergroup_list.append(index+1)

    return layergroup_list


def layerNorm(bands:np.array, energy_scale=10., shift=1.) -> np.array:

    norm_v = np.linalg.norm(bands, axis=0)
    bands = ((bands / norm_v)+shift)*energy_scale

    return bands


def processing(save_dir="../../input_data/energy_separation01/",
               raw_data_dir="../../database/c2db_database/",
               num_of_bands=100,
               degeneracy=False,
               energy_separation=False,
               en_tolerance=0.001,
               padding_around_fermi=False,
               padding_vb_only=False,
               padding_num=0,
               is_soc=False,
               bands_below_fermi_limit=50,
               layer_norm=False,
               layergroup_lower_bound=-1,
               energy_scale=10.,
               shift=1.,
               norm_before_padding=False):  # energy scale used for normalization

    if layergroup_lower_bound > 0:
        layergroup_list = layergroup_filter(layergroup_lower_bound)
    else:
        layergroup_list = [i for i in range(1,81)]

    norm_after_padding = False
    count = 0
    valid_count = 0
    layer_data = {}

    for dirs, subdirs, files in os.walk(raw_data_dir):
        total_count = len(files)
        for i in tqdm(range(total_count), desc="Processing data"):
            file = files[i]
            layer_data = {}

            with open(os.path.join(raw_data_dir, file), 'r') as jsonf:
                json_data = json.load(jsonf)

            this_datum = format_data_layergroup.LayerBands(jsonbands=json_data, file_path=save_dir)

            # >>>step 1: Check the population
            if this_datum.layergroup_num in layergroup_list:
                new_label = layergroup_list.index(this_datum.layergroup_num)

                if not is_soc:
                    spinup_bands = this_datum.spinup_bands.T
                    spindown_bands = this_datum.spindown_bands.T
                    is_spin_polarized = this_datum.is_spin_polarized

                    # >>>step 2: Find fermi level
                    try:
                        fermi_level_spinup = format_data_layergroup.LayerBands.find_fermi_index(spinup_bands)
                        fermi_level_spindown = format_data_layergroup.LayerBands.find_fermi_index(spindown_bands)
                    except IndexError as e:
                        print(this_datum.uid)
                        raise e
                    # >>>step 3:  Process by degeneracy or energy separation
                    if degeneracy:
                        tranformed_spinup_bands = format_data_layergroup.LayerBands.degen_translate(spinup_bands, en_tolerance=en_tolerance)
                        tranformed_spindown_bands = format_data_layergroup.LayerBands.degen_translate(spindown_bands, en_tolerance=en_tolerance)
                    elif energy_separation:

                        # >>>step 3.a: normalization
                        if layer_norm:
                            if norm_before_padding:
                                # normalize the energy difference to a scale
                                spinup_bands = layerNorm(spinup_bands, energy_scale, shift=shift)
                                spindown_bands = layerNorm(spindown_bands, energy_scale, shift=shift)
                            else:
                                norm_after_padding = True

                        tranformed_spinup_bands = format_data_layergroup.LayerBands.energy_separation(spinup_bands, padded_number=padding_num)
                        tranformed_spindown_bands = format_data_layergroup.LayerBands.energy_separation(spindown_bands, padded_number=padding_num)
                    else:
                        #TODO: new function to be developed here
                        tranformed_spinup_bands = None
                        tranformed_spindown_bands = None

                    if tranformed_spinup_bands is None:
                        raise Exception("Neither gegeneracy nor energy separation is opted")

                    padded_spinup_bands = this_datum.padding_around_fermi(formatted_bands=tranformed_spinup_bands,
                                                                          num_of_bands=num_of_bands,
                                                                          bands_below_fermi_limit=bands_below_fermi_limit,
                                                                          padding_num=padding_num,
                                                                          fermi_index=fermi_level_spinup)

                    padded_spindown_bands = this_datum.padding_around_fermi(formatted_bands=tranformed_spindown_bands,
                                                                            num_of_bands=num_of_bands,
                                                                            bands_below_fermi_limit=bands_below_fermi_limit,
                                                                            padding_num=padding_num,
                                                                            fermi_index=fermi_level_spindown)

                    # >>>step 3.a: normalization
                    if layer_norm:
                        if norm_after_padding:
                            # normalize the energy difference to a scale
                            padded_spinup_bands = layerNorm(padded_spinup_bands, energy_scale, shift=shift)
                            padded_spindown_bands = layerNorm(padded_spindown_bands, energy_scale, shift=shift)

                    # #FIXME: debugging use
                    # print(file)
                    # print("Spin-up bands: ",padded_spinup_bands.shape)
                    # print("Spin-down bands: ", padded_spindown_bands.shape)
                    # print("Layer group:",this_datum.layergroup_num)
                    #
                    # print("Spin-up bands")
                    # print(padded_spinup_bands)
                    # print("----------------------------------------------------------")

                    # >>>step 4: save
                    # monolayer structure
                    layer_data["uid"] = this_datum.uid
                    layer_data["formula"] = this_datum.formula

                    # structure
                    layer_data["atoms_type"] = this_datum.atoms_type
                    layer_data["lattice"] = this_datum.lattice
                    layer_data["positions"] = this_datum.positions
                    layer_data["layergroup_number"] = this_datum.layergroup_num
                    layer_data["new_label"] = new_label
                    # bands and k points
                    layer_data["is_spin_polarized"] = this_datum.is_spin_polarized
                    layer_data["spin_up_bands"] = padded_spinup_bands
                    layer_data["spin_down_bands"] = padded_spindown_bands
                    # layer_data["soc_bands"]
                    layer_data["kpoints"] = this_datum.kps
                    layer_data["k_labels"] = this_datum.kps_labels

                    valid_count += 1

                    with open(os.path.join(save_dir, file), 'w') as jf:
                        json.dump(layer_data, jf, cls=format_data_layergroup.NumpyEncoder, indent=2)

                    # print(f"\r\t FINISHED: COUNT: {count}|VALID: {valid_count}|TOTAL: {total_count}", end=' ')

                    del new_label

            del layer_data
            count += 1

        print(f"\n Valid: {valid_count} | Total: {total_count}")
        print("Valid Layer groups:",layergroup_list)
        print(f"Successfully saved into {save_dir} ")


def test():
    processing(save_dir="../c2db_database_test/",
               raw_data_dir="../c2db_database_test/",
               num_of_bands=60,
               degeneracy=False,
               energy_separation=True,
               en_tolerance=0.0005,
               padding_around_fermi=False,
               padding_vb_only=False,
               padding_num=0,
               is_soc=False,
               bands_below_fermi_limit=40,
               layer_norm=True,
               layergroup_lower_bound=-1,
               energy_scale=10.,
               shift=1.,
               norm_before_padding=True)

# test()


if __name__ == "__main__":

    # load parameters from config.py
    # TODO: The parameters are loaded from config.py, please change the parameters there

    print("Loading...")

    from config import args
    if args['process']['start']:
        cfg = args['process']['parameters']
        processing(save_dir=cfg["save_to_directory"],
                   raw_data_dir=cfg["raw_data_directory"],
                   degeneracy=cfg['degeneracy'],
                   energy_separation=cfg["energy_separation"],
                   en_tolerance=cfg['en_tolerance'],
                   padding_around_fermi=cfg['padding_around_fermi'],
                   padding_vb_only=cfg['padding_vb_only'],
                   padding_num=cfg["padding_num"],
                   is_soc=cfg["is_soc"],
                   num_of_bands=cfg['num_of_bands'],
                   bands_below_fermi_limit=cfg['bands_below_fermi_limit'],
                   layer_norm=cfg["layer_norm"],
                   layergroup_lower_bound=cfg["layergroup_lower_bound"],
                   energy_scale=cfg["energy_scale"],
                   shift =cfg["shift"],
                   norm_before_padding=cfg["norm_before_padding"])
