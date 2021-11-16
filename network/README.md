# Deep learning neural networks
## Descriptions:
### This is the deep learning project of giving layer group preditons based on electronic bands of layered structures. This directory mainly stored the code of deep learning model training.

## Usage of the scripts

- `config.py`: Put all parameters for input data preparation, and model architectures as well as hyperparamters.
- `main_preparation.py`: Prepare input data, basically it splits the total dataset into training and test set (dev set) randomly. The train set and test set follow similar 
distribution. It will generates valid data list that is convenient for data tracing. You need set proper data path to load input data after [data processing step](https://github.com/binxi0629/BS2layergroup/new/main/processing).
- `main_bs2lg.py`: Main script for model training, it will load both model architecture and hyperparameters from `config.py`, so give a proper setting there.
- `function_list`: Functions stored for data preparation use.
- `function_training`: Functions used for training and test model.
- `dataLoader.py`: A class that inherits tensorflow.data.Dataset class, added new functionality and attributes for our purpose.
-  `crytsal.py`: Space group classification based on 7 crystal systems.
