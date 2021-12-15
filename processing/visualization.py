import json
import numpy
import numpy as np
from matplotlib import pyplot
import pandas
import seaborn
import random, time, re,os


def v_from_dir(dir_name='data_augmentationExamples/',
               save_to_path='../../results/dataAugmentationExamples/d/'):
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
                            vmin=0)
            count += 1
            file_name = os.path.join(save_to_path,f"sample_{count}")
            fig.savefig(fname=file_name)


def v_random_from_all(dir_name='../../input_data/energy_separation04_auged/',
                      save_to_path='../../results/dataAugmentationExamples/d/',
                      layergroup_num=80, num_example=10):
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
            file_name = os.path.join(save_to_path, f"example{count}" )
            fig.savefig(fname=file_name)


def main():
    # FIXME: set proper path when calling it
    v_from_dir()
    # v_random_from_all(layergroup_num=78, num_example=20)


if __name__ == '__main__':
    main()