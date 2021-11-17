# Processing data
## Description 
### This is a deep learning project, aimming to give predictions of [layer groups](https://en.wikipedia.org/wiki/Layer_group) by inputs of band structures of layered materials. 

Database: all layered structures come from [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html), that contains valid 4046 structures performed with either spin-polarized or 
nonspin-polarized band structure calculation. Also effect of [spin-orbit coupling](https://en.wikipedia.org/wiki/Spin%E2%80%93orbit_interaction) is considered 
(which is out of the scope of this project at the current step:).

### Below are the usage of scripts

- `layergroup.py`: Symmetry analysis of 80 layer groups based on fractional coordinates of atoms and atom species.
- `config.py`: Parameter settings of input data.
- `fromat_data_layergroup.py`: Core class of a json file that contains e.g. system name, atom postions, lattice constant, bands, kpoints..., methods are provided to process raw bands 
into desired input band format.
- `loadLayerData.py`: Main script of data processing, it will load the parameters in `config.py`, the desired input format will stored in json file in given path.
- `lg_distribution.json`: Layer group population list.

### Design your own input format
1. Add your customized methods into `LayerBands` class of `format_data_layergroup.py`.
2. Call the function inside `processing(args)` of `loadLayerData.py`


