from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from plotting import *
import os
from tools.basics import create_folder_structure, find_files
import multiprocessing
import traceback
from utils import check_aperiodic
import multiprocessing as mp
from graph_tool.all import *
import datetime

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
matplotlib.rcParams.update({'font.size': 20})


def plot_df(df, bias_strength, filename):
    plt.scatter(df['com-size'], df['stat_dist'], c=df['in_neighbours_in_deg'], lw=0, alpha=0.7, cmap='coolwarm')
    cbar = plt.colorbar()
    plt.xlabel('com-size')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('$\\sum \\pi$')
    cbar.set_label('sum in neighbours in deg')
    plt.title('Bias Strength: ' + str(int(bias_strength)))
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close('all')


def main():
    base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
    out_dir = base_dir + 'plots/'
    create_folder_structure(out_dir)

    result_files = find_files(base_dir, '.df')
    print result_files

    for i in result_files:
        print 'plot:', i.rsplit('/', 1)[-1]
        df = pd.read_pickle(i)
        bias_strength = int(i.split('_bs')[-1].split('.')[0])
        out_fn = i[:-3] + '.png'
        plot_df(df, bias_strength, out_fn)


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

