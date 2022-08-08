import json
import numpy
import numpy as np
from matplotlib import pyplot
import pandas
import seaborn
import random, time, re,os


def v_from_dir(dir_name = 'data_augmentationExamples/'):
    count=0
    for dirs, subdirs, files in os.walk(dir_name):
        for i in range(len(files)):
            file = files[i]
            with open(os.path.join(dir_name,file), 'r') as f:
                dataj = json.load(f)
            bands = np.array(dataj["spin_up_bands"])
            bands = np.flip(bands,axis=0)
            fig = pyplot.figure()
            dataframe_colormap = pandas.DataFrame(bands)
            seaborn.heatmap(dataframe_colormap,
                            cmap="YlGnBu",
                            vmax=5,
                            vmin=0
                            )
            count+=1
            fig.savefig(fname='../../results/dataAugmentationExamples/d/' + f"sample_{count}")


def v_random_from_all(dir_name = '../../input_data/energy_separation04_auged/', layergroup_num=80, num_example=10):
    count = 0
    for dirs, subdirs, files in os.walk(dir_name):
        np.random.shuffle(files)
        for i in range(len(files)):

            if count >= num_example:
                break

            file = files[i]
            with open(os.path.join(dir_name,file), 'r') as f:
                dataj = json.load(f)

            lg = dataj["layergroup_number"]
            if lg != layergroup_num:
                continue
            count +=1
            bands = np.array(dataj["spin_up_bands"])
            bands = np.flip(bands,axis=0)
            fig = pyplot.figure()
            dataframe_colormap = pandas.DataFrame(bands)
            seaborn.heatmap(dataframe_colormap,
                            cmap="YlGnBu",
                            vmin=-0.1,
                            vmax=0.5)
            pyplot.title(f"{file} | layergroup number: {lg}")
            fig.savefig(fname='../../results/dataAugmentationExamples/d/' + f"example{count}")


v_from_dir()

# v_random_from_all(layergroup_num=78, num_example=20)
