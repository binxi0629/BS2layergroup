args={
    "process": {
        "start": True,
        "parameters": {
            "save_to_directory": "../../input_data/degeneracy_04_auged/",
            "raw_data_directory": "../../database/c2db_database/",
            "degeneracy": True,  # TODO: change the tag
            "energy_separation": False,  # TODO: change the tag
            "en_tolerance": 0.0005,
            "padding_around_fermi": True,
            "padding_vb_only": False,
            "padding_num": 0,  # TODO: change the tag
            "is_soc": False,
            "num_of_bands": 80,
            "bands_below_fermi_limit": 50,
            "layer_norm": False,  # TODO: change the tag
            "num_of_kps":100,  # TODO: change the tag
            "layergroup_lower_bound": 10,
            "energy_scale": 10,
            "shift": 1.,
            "norm_before_padding": False,  # TODO: change the tag
            "kpaths_shuffle":True,  # TODO: change the tag
            "do_agumentation":True,  # TODO: change the tag
            "agmentation_class_limit":300,
            "ag_shiftting_rate":0.6
        }
    }
}
