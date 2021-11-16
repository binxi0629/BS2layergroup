args={
    "process": {
        "start": True,
        "parameters": {
            "save_to_directory": "../../input_data/degeneracy_02/",
            "raw_data_directory": "../../database/c2db_database/",
            "degeneracy": True,
            "energy_separation": False,
            "en_tolerance": 0.0005,
            "padding_around_fermi": True,
            "padding_vb_only": False,
            "padding_num": 0,
            "is_soc": False,
            "num_of_bands": 60,
            "bands_below_fermi_limit": 40,
            "layer_norm": False,
            "layergroup_lower_bound": 10,
            "energy_scale": 10,
            "shift": 1.,
            "norm_before_padding": True
        }
    }
}
