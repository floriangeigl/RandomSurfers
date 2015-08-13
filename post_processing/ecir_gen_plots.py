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

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)


def get_stat_dist_bias_sum(df, col_name, cat_name, category_col_name):
    filt_df = df[[col_name, category_col_name]]
    assert np.isclose(df[col_name].sum(), 1)
    bias_sum = filt_df[col_name][filt_df[category_col_name] == cat_name].sum()
    return bias_sum


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)

stat_dist_files = find_files(base_dir, 'stat_dists.df')
print stat_dist_files

for stat_dist_fn in stat_dist_files:
    ds_name = stat_dist_fn.rsplit('/', 1)[-1].split('_stat_dist')[0]
    print ds_name
    res_df = pd.DataFrame(index=[1.])
    df = pd.read_pickle(stat_dist_fn)
    bias_base_names = set(map(lambda x: x.split('_cs')[0], filter(lambda x: '_bs' and '_cs' in x, df.columns)))
    print bias_base_names
    for bias_name in bias_base_names:
        bias_columns = filter(lambda x: x.startswith(bias_name), df.columns)
        bias_label = bias_name + ' (' + str(
            "%.2f" % (float(bias_columns[0].split('_cs')[-1].split('_bs')[0]) / len(df) * 100)) + '%)'
        print bias_columns
        # unbiased
        res_df.at[1., bias_label] = get_stat_dist_bias_sum(df, 'adjacency', bias_name, 'category')
        for bc in bias_columns:
            bs = float(bc.split('_bs')[-1])
            res_df.at[bs, bias_label] = get_stat_dist_bias_sum(df, bc, bias_name, 'category')
        res_df.sort(inplace=True)
    print res_df
    # res_df /= res_df.max()
    res_df.plot(lw=3, style='-*')
    plt.xlabel('bias strength')
    plt.ylabel('sum of stationary values')
    plt.tight_layout()
    plt.savefig(out_dir + ds_name + '_bias_influence.pdf')

    res_df /= res_df.min()
    res_df.plot(lw=3, style='-*')
    plt.xlabel('bias strength')
    plt.ylabel('fraction of unbiased sum')
    plt.ylim([1., res_df.max().max()])
    plt.tight_layout()
    plt.savefig(out_dir + ds_name + '_bias_influence_norm.pdf')
