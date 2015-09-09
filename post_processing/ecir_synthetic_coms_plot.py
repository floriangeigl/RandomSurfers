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
    gb = df[['com-size', 'stat_dist']].groupby('com-size')
    trans_lambda = lambda x: (x-x.mean()) / x.std()
    gb = gb.transform(trans_lambda)
    # print gb
    stat_dist = np.array(gb['stat_dist']).astype('float')
    stat_dist[np.invert(np.isfinite(stat_dist))] = np.nan
    df['stat_dist_normed'] = stat_dist
    df.dropna(axis=0, how='any', inplace=True)
    # print df
    for i in [True, False]:
        y = df['in_neighbours_in_deg']
        x = df['com-size']
        c = df['stat_dist'] if i else df['stat_dist_normed']
        plt.scatter(x, y, c=c, lw=0, alpha=0.7, cmap='coolwarm')
        cbar = plt.colorbar()
        plt.xlabel('category size')
        plt.xlim([0, 1])
        plt.ylim([df['in_neighbours_in_deg'].min(), df['in_neighbours_in_deg'].max()])
        plt.ylabel('in neighbors in degree')
        cbar.set_label('$\\sum \\pi$')

        plt.title('Bias Strength: ' + str(int(bias_strength)))
        plt.tight_layout()
        out_f = filename if i else filename[:-4] + '_normed.png'
        plt.savefig(out_f, dpi=150)
        plt.close('all')
    cor_df = pd.DataFrame()
    cor_df['stat_dist'] = df['stat_dist'].astype('float')
    cor_df['X'] = (df['in_neighbours_in_deg']).astype('float')
    print cor_df.corr()

    return cor_df.corr().iat[0, 1]


def main():
    base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
    out_dir = base_dir + 'plots/'
    create_folder_structure(out_dir)

    result_files = find_files(base_dir, '.df')
    print result_files
    cors = list()
    for i in sorted(result_files):
        print 'plot:', i.rsplit('/', 1)[-1]
        df = pd.read_pickle(i)
        bias_strength = int(i.split('_bs')[-1].split('.')[0])
        out_fn = out_dir + i.rsplit('/', 1)[-1][:-3] + '.png'
        cors.append(plot_df(df, bias_strength, out_fn))
    cors = np.array(cors)
    print 'average corr:', cors.mean()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

