import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal as sig
from scipy.interpolate import interp1d
ROOT = os.environ['HOME'] + '/python/'
idx = pd.IndexSlice


#converts input array to smooth array of length standard_length
def standarize(ar, standard_length = 10, window_size = 3):
    f = interp1d(np.linspace(0, standard_length, len(ar)), ar)
    return sig.savgol_filter(f(range(standard_length)), window_size, 2)

#array must be subjects x datapoints
def prepare_line(array):
    avg = np.nanmean(array, axis = 0)
    sem = (np.nanstd(array, axis = 0) / np.sqrt(len(array)))
    return avg, sem

def fancy_line(avg, sem, ax, colorOne):
    ax.fill_between(range(len(avg)), avg - sem, avg + sem, color = colorOne)
    ax.plot(range(len(avg)), avg, color = "white", lw=2)

def training_plot(RWScores, title, ylim = [0.45, 0.8],
                  noFields = 9, savefig = 0):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim([0,8])
    ax.set_ylim(ylim)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(range(noFields))
    ax.set_xticklabels(['Naive','','','','Mid-Training','','','','Criteria'],
                        fontsize = 14)

    colors = ["#3F5D7D", "#559e83", "#7d3f5d", "#5d7d3f", "#ae5a41" ]
    P = []
    noRats = RWScores.index.levels[0].shape[0]


    vars = ['alpha','beta','score']
    for param_index, param in enumerate(vars):
        array = np.zeros((noRats, noFields))
        for index, (rat_label, rat_data) in enumerate(RWScores.groupby('rat')):
            array[index, :] = \
                    standarize(rat_data[param], standard_length = noFields)
        avg, sem = prepare_line(array)

        fancy_line(avg, sem, ax, colors[param_index])
        P.append(Rectangle((0, 0), 1, 1, fc = colors[param_index]))

    plt.legend(P, vars, fontsize=14, loc = 'best')
    plt.title(title, fontsize = 20)
    if savefig: plt.savefig(title.replace(' ','_') + '.jpg',
                            bbox_inches = 'tight',
                            dpi= 600)
    plt.show()

if __name__ == '__main__':
    source = ROOT + 'RLmodule/Results/RW_DSR.p'
    RWScores = pickle.load(open(source, 'rb'))
    training_plot(RWScores, title='Training DSR - RW Model', ylim=[0, 1])
