from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from plotting import *
import os
from tools.basics import create_folder_structure

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)

def find_files(base_dir,file_ending):
    res = list()
    for root, dirs, files in os.walk(base_dir):
        if not root.endswith('/'):
            root += '/'
        res.extend([root + i for i in filter(lambda x: x.endswith(file_ending), files)])
    return sorted(res)


base_dir = '/home/fgeigl/navigability_of_networks/output/wsdm/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)


bias_name_mapping = dict()
bias_name_mapping['adjacency'] = 'Unbiased'
bias_name_mapping['deg'] = 'Degree Bias Bias Factor'
bias_name_mapping['eigenvector'] = 'Eigenvector C. Bias Factor'
bias_name_mapping['inv_sqrt_deg'] = 'Inv. Degree Bias Factor'
bias_name_mapping['sigma'] = 'Similarity Bias Factor'
bias_name_mapping['sigma_sqrt_deg_corrected'] = 'Deg. Cor. Similarity Bias Factor'


base_line = bias_name_mapping[base_line] if base_line in bias_name_mapping else base_line

stat_dist_files = find_files(base_dir, 'stat_dists.df')
entropy_files = find_files(base_dir, 'entropy.df')
gini_files = find_files(base_dir, 'gini.df')
network_files = filter(lambda x: 'rewire' not in x, find_files(base_dir, '.gt'))
plot_degree_distributions(network_files, out_dir + 'deg_distributions/')

entropy_rates = None
for idx, fn in enumerate(entropy_files):
    entropy_rate_df = pd.read_pickle(fn)
    if entropy_rates is None:
        entropy_rates = pd.DataFrame(columns=list(entropy_rate_df.columns))
    entropy_rates.loc[fn.rsplit('/', 1)[-1].replace('_entropy.df', '')] = entropy_rate_df.loc[0]
plot_entropy_rates(entropy_rates, out_dir + 'entropy.pdf')

for fn in stat_dist_files:
    ds_name = fn.rsplit('/', 1)[-1].replace('stat_dists.df', '').strip('_')
    stat_dist_df = pd.read_pickle(fn)
    stat_dist_df.columns = [bias_name_mapping[i] if i in bias_name_mapping else i for i in stat_dist_df.columns]
    not_baseline_cols = list(filter(lambda x: x != base_line, stat_dist_df.columns))
    print stat_dist_df.head()
    bias_factors_df = create_bf_scatters_from_df(stat_dist_df, base_line, not_baseline_cols,
                                                 output_folder=out_dir + 'bf_scatter/' + ds_name + '/')
    min_y, max_y = bias_factors_df[bias_factors_df > 0].min().min(), bias_factors_df.max().max()

    bias_factors_df = create_bf_scatters_from_df(stat_dist_df, base_line, not_baseline_cols,
                                                 output_folder=out_dir + 'bf_scatter/' + ds_name + '/',
                                                 y_range=[min_y, max_y])

    create_scatters_from_df(stat_dist_df, stat_dist_df.columns, output_folder=out_dir + 'scatter/' + ds_name + '/')
    create_ginis_from_df(stat_dist_df, stat_dist_df.columns, output_folder=out_dir + 'gini/' + ds_name + '/', lw=3,
                         ms=15, font_size=15)

