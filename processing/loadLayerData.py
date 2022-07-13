import format_data_layergroup
import os, re, json, math
import numpy as np

from copy import deepcopy, copy
from tqdm import tqdm


def layergroup_filter(layergroup_lower_bound=-1):
    # TODO: hard code the counting result of layergroup
    with open("lg_distribution.json", 'r') as f:
        dis=json.load(f)
    layergroup_stat = dis["lg_distribution"]

    layergroup_list=[]
    if layergroup_lower_bound > 0:
        for index, i in enumerate(layergroup_stat):
            if i >= layergroup_lower_bound:
                layergroup_list.append(index+1)
    else:
        layergroup_list = [i for i in range(1, 81)]

    return layergroup_list, layergroup_stat

def layernumber_filter(layernumber_lower_bound=-1):
    # TODO: hard code the counting result of layer numbers
    # only layers with enough population are counted into the dataset
    # layernumber_lower_bound = 150
    with open("layernumber_distribution.json", 'r') as f:
        dis=json.load(f)
    layernumber_stat = dis["layer_population"]

    layernumber_list=[]
    if layernumber_lower_bound > 0:
        for index, i in enumerate(layernumber_stat):
            if i >= layernumber_lower_bound:
                layernumber_list.append(index+1)
    else:
        layernumber_list = [i for i in range(1, 18)]

    return layernumber_list, layernumber_stat


def layerNorm(bands:np.array, energy_scale=10., shift=1.) -> np.array:

    norm_v = np.linalg.norm(bands, axis=0)
    bands = ((bands / norm_v)+shift)*energy_scale

    return bands


def processing(save_dir="../../input_data/energy_separation01/",
               raw_data_dir="../../database/c2db_database/",
               num_of_bands=100,
               num_of_kps=100,  # TODO: Newly added
               degeneracy=False,
               energy_separation=False,
               eigenvalue=False,
               en_tolerance=0.001,
               padding_around_fermi=False,
               padding_vb_only=False,
               padding_num=0,
               is_soc=False,
               bands_below_fermi_limit=50,
               layer_norm=False,
               layergroup_lower_bound=-1,
               layernumber_lower_bound=-1,
               energy_scale=10.,  # energy scale used for normalization
               shift=1.,
               norm_before_padding=False,
               kpaths_shuffle=False,  # TODO: shuffle kpaths
               do_agumentation=False,  # TODO: Data agumentation
               agmentation_class_limit=50, # TODO: No. data to generate
               ag_shiftting_rate=0.4, # TODO: shiftting rate of kps
               debug=False):

    # FIXME: When not debugging, set debug to False
    # debug = False

    # layergroup_list, lg_population_list = layergroup_filter(layergroup_lower_bound)

    # need edit name
    layernumber_list, layernumber_population_list = layernumber_filter(layernumber_lower_bound)
    # print (f"layernumber_list {layernumber_list}\n")
    # print (f"layernumber_list {layernumber_list.index(3)}\n")

    # if debug:
    #     print("No. Valid layer groups: {}".format(len(layergroup_list)))

    if debug:
        print("No. Valid layer groups: {}".format(len(layernumber_list)))

    norm_after_padding = False
    count = 0
    valid_count = 0
    ag_data_count =0
    ZeroDivisionError_count = 0

    for dirs, subdirs, files in os.walk(raw_data_dir):
        total_count = len(files)
        for i in tqdm(range(total_count), desc="Processing data"):
            file = files[i]
            layer_data = {}

            try:

                with open(os.path.join(raw_data_dir, file), 'r') as jsonf:
                    # print(f"file name {file}")
                    json_data = json.load(jsonf)

                this_datum = format_data_layergroup.LayerBands(jsonbands=json_data, file_path=save_dir)

                # TODO: step 1: Check the population
                # if this_datum.layergroup_num in layergroup_list:
                if this_datum.layers_num in layernumber_list:
                    new_label = layernumber_list.index(this_datum.layers_num)
                    # new_label = layergroup_list.index(this_datum.layergroup_num)

                    if not is_soc:
                        spinup_bands = this_datum.spinup_bands.T  # FIXME  No. bands X No. kps
                        spindown_bands = this_datum.spindown_bands.T

                        # FIXME: to be developed
                        is_spin_polarized = this_datum.is_spin_polarized

                        # TODO: step 2: Find fermi level
                        try:
                            fermi_level_spinup = format_data_layergroup.LayerBands.find_fermi_index(spinup_bands)
                            fermi_level_spindown = format_data_layergroup.LayerBands.find_fermi_index(spindown_bands)

                            if debug:
                                print("Shape of raw spin-up bands: {}".format(spinup_bands.shape))
                                print("Shape of raw spin-down bands: {}".format(spindown_bands.shape))
                                print("Fermi level of Spin-up bands is at: {}".format(fermi_level_spinup))
                                print("Fermi level of Spin-down bands is at: {}".format(fermi_level_spindown))

                        except IndexError as e:
                            print("Fermi level not found for",this_datum.uid)
                            raise e

                        # TODO: step 3:  Process by degeneracy or energy separation
                        if degeneracy:
                            tranformed_spinup_bands = format_data_layergroup.LayerBands.degen_translate(spinup_bands, en_tolerance=en_tolerance)
                            tranformed_spindown_bands = format_data_layergroup.LayerBands.degen_translate(spindown_bands, en_tolerance=en_tolerance)

                        elif energy_separation:

                            # TODO: step 3.a: normalization
                            if layer_norm:
                                if norm_before_padding:
                                    # normalize the energy difference to a scale
                                    spinup_bands = layerNorm(spinup_bands, energy_scale, shift=shift)
                                    spindown_bands = layerNorm(spindown_bands, energy_scale, shift=shift)
                                else:
                                    norm_after_padding = True
                            tranformed_spinup_bands = format_data_layergroup.LayerBands.energy_separation(spinup_bands, padded_number=padding_num)
                            tranformed_spindown_bands = format_data_layergroup.LayerBands.energy_separation(spindown_bands, padded_number=padding_num)

                        elif eigenvalue:
                            #TODO:
                            if layer_norm:
                                if norm_before_padding:
                                    # normalize the energy difference to a scale
                                    spinup_bands = layerNorm(spinup_bands, energy_scale, shift=shift)
                                    spindown_bands = layerNorm(spindown_bands, energy_scale, shift=shift)
                                else:
                                    norm_after_padding = True
                            tranformed_spinup_bands = format_data_layergroup.LayerBands.eigenvalue(spinup_bands, padded_number=padding_num)
                            tranformed_spindown_bands = format_data_layergroup.LayerBands.eigenvalue(spinup_bands, padded_number=padding_num)

                        else:
                            #TODO: new function to be developed here for Spin-coupling effect
                            tranformed_spinup_bands = None
                            tranformed_spindown_bands = None

                        if tranformed_spinup_bands is None:
                            raise Exception("Neither degeneracy nor energy separation is opted")

                        if debug:
                            print("Shape of spin-up bands after degeneracy or energy separation: {}".format(tranformed_spinup_bands.shape))
                            print("Shape of spin-down bands after degeneracy or energy separation: {}".format(tranformed_spindown_bands.shape))

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

                        if debug:
                            print("Shape of Spin-up bands aftering padding: {}".format(padded_spinup_bands.shape))
                            print("Shape of Spin-down bands aftering padding: {}".format(padded_spindown_bands.shape))

                        # TODO: step 3.a: normalization
                        if layer_norm:
                            if norm_after_padding:
                                # normalize the energy difference to a scale
                                padded_spinup_bands = layerNorm(padded_spinup_bands, energy_scale, shift=shift)
                                padded_spindown_bands = layerNorm(padded_spindown_bands, energy_scale, shift=shift)

                        # TODO: step 4. shrink the dimension of kps
                        kpaths = this_datum.special_kps_separation(num_of_kps)

                        # TODO: step 5. Data augmentation
                        if do_agumentation:
                            kpaths_backup = deepcopy(kpaths)

                            # TODO: 5.a if need augmenattion
                            # print (f"this_datum.layergroup_num - 1 :{this_datum.layergroup_num - 1}")
                            # print (f"lg_population_list: {np.array(lg_population_list).shape}")
                            # print (lg_population_list)


                            # if lg_population_list[this_datum.layergroup_num - 1] < agmentation_class_limit:
                            #     num_this_lg = lg_population_list[this_datum.layergroup_num - 1]
                            if layernumber_population_list[this_datum.layers_num - 1] < agmentation_class_limit:
                                num_this_lg = layernumber_population_list[this_datum.layers_num - 1]

                                num_agumentation = math.ceil((agmentation_class_limit -num_this_lg)/num_this_lg)

                                this_datum.agumented_num = num_agumentation
                                temp_kps_list_list = []
                                agumented_spinup_bands = []
                                agumented_spindown_bands = []

                                # TODO: 5.b generate new paths
                                new_paths_list = format_data_layergroup.LayerBands.produceNewKpsIdxFromShuffledKpath(
                                    kpaths_backup,
                                    num_agumentation,
                                    shiftting_rate=ag_shiftting_rate)

                                for ag_count in range(len(new_paths_list)):
                                    # shuffle ordering of new kpaths
                                    format_data_layergroup.LayerBands.kpath_shuffle(new_paths_list[ag_count])
                                    # unroll each kpath
                                    temp_kps_list = format_data_layergroup.LayerBands.unrollKpath(new_paths_list[ag_count])
                                    temp_kps_list_list.append(temp_kps_list)

                                    # extract bands and record
                                    agumented_spinup_bands.append(format_data_layergroup.LayerBands.extractBandsByKpsIdx(padded_spinup_bands, temp_kps_list))
                                    agumented_spindown_bands.append(format_data_layergroup.LayerBands.extractBandsByKpsIdx(padded_spindown_bands,
                                                                                            temp_kps_list))

                        if kpaths_shuffle:
                            # shuffle ordering of kpaths
                            format_data_layergroup.LayerBands.kpath_shuffle(kpaths)
                            kps_list = format_data_layergroup.LayerBands.unrollKpath(kpaths)

                            if debug:
                                print("No. Kps:", len(kps_list))
                                print("kps:", kps_list)
                        else:
                            kps_list = format_data_layergroup.LayerBands.unrollKpath(kpaths)

                        padded_spinup_bands = format_data_layergroup.LayerBands.extractBandsByKpsIdx(padded_spinup_bands, kps_list)
                        padded_spindown_bands = format_data_layergroup.LayerBands.extractBandsByKpsIdx(padded_spindown_bands, kps_list)

                        if debug:
                            print("Spin-up bands shape after kpaths shuffling:", padded_spinup_bands.shape)
                            print("Spin-down bands shape after kpaths shuffling:",padded_spindown_bands.shape)

                        # TODO: step 4: save
                        # monolayer structure
                        layer_data["uid"] = this_datum.uid
                        layer_data["formula"] = this_datum.formula

                        # structure
                        layer_data["atoms_type"] = this_datum.atoms_type
                        layer_data["lattice"] = this_datum.lattice
                        layer_data["positions"] = this_datum.positions
                        # layer_data["z_layers_num"] = this_datum.z_layers_num
                        layer_data["layers_num"] = this_datum.layers_num
                        # layer_data["layer_direction"] = this_datum.layer_direction
                        
                        
                        layer_data["layergroup_number"] = this_datum.layergroup_num
                        
                        layer_data["spacegroup"] = this_datum.spacegroup
                        layer_data["spacegroup_num"] = this_datum.spacegroup_num

                        layer_data["new_label"] = new_label
                        layer_data["kpoints"] = this_datum.kps
                        # layer_data["kpoints_number"] = this_datum.kps.shape
                        layer_data["k_labels"] = this_datum.labels_idx

                        # bands and k points
                        layer_data["is_spin_polarized"] = this_datum.is_spin_polarized

                        if this_datum.agumented_num != 0:
                            ag_data_count += this_datum.agumented_num

                            if debug:
                                print("Augmentation: {}".format(this_datum.agumented_num))

                            for i in range(this_datum.agumented_num):
                                layer_data["k_idx"] = temp_kps_list_list[i]
                                layer_data["shrinked_kpoint_numbers"] = np.array(layer_data["k_idx"]).shape[0]
                                layer_data["spin_up_bands"] =agumented_spinup_bands[i]
                                layer_data["spin_down_bands"] = agumented_spindown_bands[i]
                                file_name ="c2db_"+this_datum.uid+f'_{i+1}.json'
                                with open(os.path.join(save_dir, file_name), 'w') as jf:
                                    json.dump(layer_data, jf, cls=format_data_layergroup.NumpyEncoder, indent=2)

                                if debug:
                                    print("\t---------------------------------------------------------")
                                    print(f"\t the {i+1} Augmented spin-up bands shape: {agumented_spinup_bands[i].shape}")
                                    print(f"\t the {i+1} Augmented spin-down bands shape: {agumented_spindown_bands[i].shape}")
                                    print("\t---------------------------------------------------------")

                        layer_data["spin_up_bands"] = padded_spinup_bands
                        layer_data["spin_down_bands"] = padded_spindown_bands

                        # layer_data["soc_bands"]

                        layer_data["k_idx"] = kps_list
                        layer_data["shrinked_kpoint_numbers"] = np.array(layer_data["k_idx"]).shape[0]
                        valid_count += 1

                        with open(os.path.join(save_dir, file), 'w') as jf:
                            json.dump(layer_data, jf, cls=format_data_layergroup.NumpyEncoder, indent=2)
                        # print(f"\r\t FINISHED: COUNT: {count}|VALID: {valid_count}|TOTAL: {total_count}", end=' ')

                        del new_label

                del layer_data
                count += 1

            except ZeroDivisionError:
                if debug:
                    print (f"ZeroDivisionError file name: {file}")
                ZeroDivisionError_count += 1

        print(f"\n Valid: {valid_count} | Augmented: {ag_data_count} |Total: {total_count}")
        # print("Valid Layer groups:",layergroup_list)
        print("Valid Layer numbers:",layernumber_list)
        print(f"Successfully saved into {save_dir} ")
        print (f"ZeroDivisionError_count = {ZeroDivisionError_count}")


def test():
    processing(
               save_dir="../../c2db_database02_output_eigenvalue_norm/",
               raw_data_dir="../../c2db_database02/",
            #    save_dir="../../c2db_database_test_output/",
            #    raw_data_dir="../../c2db_database_test_input/",
               num_of_bands=60,  # <<<
               num_of_kps=100,  # <<<
               degeneracy=False,  # <<<
               energy_separation=False, # <<<
               eigenvalue=True,
               en_tolerance=0.0005, # <<<
               padding_around_fermi=True,
               padding_vb_only=False,
               padding_num=-100,  # <<<
               is_soc=False,
               bands_below_fermi_limit=40,  # <<<
               layer_norm=True,  # <<<
               layergroup_lower_bound=10,  # <<<
               layernumber_lower_bound=200,
               energy_scale=10.,  # <<<
               shift=1.,  # <<<
               kpaths_shuffle=True,  # <<<
               do_agumentation=True,  # <<<
               agmentation_class_limit=20, # <<<
               ag_shiftting_rate=0.4,
               norm_before_padding=True,
               debug=False)


if __name__ == "__main__":
#
#     # load parameters from config.py
#     # TODO: The parameters are loaded from config.py, please change the parameters there
#
#     print("Loading...")
#
#     from config import args
#     if args['process']['start']:
#         cfg = args['process']['parameters']
#         processing(save_dir=cfg["save_to_directory"],
#                    raw_data_dir=cfg["raw_data_directory"],
#                    degeneracy=cfg['degeneracy'],
#                    energy_separation=cfg["energy_separation"],
#                    en_tolerance=cfg['en_tolerance'],
#                    padding_around_fermi=cfg['padding_around_fermi'],
#                    padding_vb_only=cfg['padding_vb_only'],
#                    padding_num=cfg["padding_num"],
#                    is_soc=cfg["is_soc"],
#                    num_of_bands=cfg['num_of_bands'],
#                    bands_below_fermi_limit=cfg['bands_below_fermi_limit'],
#                    layer_norm=cfg["layer_norm"],
#                    num_of_kps=cfg["num_of_kps"],
#                    layergroup_lower_bound=cfg["layergroup_lower_bound"],
#                    energy_scale=cfg["energy_scale"],
#                    shift =cfg["shift"],
#                    norm_before_padding=cfg["norm_before_padding"],
#                    kpaths_shuffle=cfg["kpaths_shuffle"],
#                    do_agumentation=cfg["do_agumentation"],
#                    agmentation_class_limit=cfg["agmentation_class_limit"],
#                    ag_shiftting_rate=cfg["ag_shiftting_rate"])

    test()
