from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from post_processing.plotting import *
import os
from tools.basics import create_folder_structure, find_files
import multiprocessing
import traceback
from tools.gt_tools import *
import string

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)


def create_plots(fn, colored_categories=None):
    try:
        ds_name = fn.rsplit('/', 1)[-1].replace('stat_dists.df', '').strip('_')
        stat_dist_df = pd.read_pickle(fn)
        stat_dist_orig_columns = list(stat_dist_df.columns)
        stat_dist_df.columns = [name_mapping[i] if i in name_mapping else i for i in stat_dist_orig_columns]
        print stat_dist_df.head()
        gini_df = stat_dist_df.copy()
        gini_df.columns = [i.split('Bias')[0].strip() for i in gini_df.columns]
        create_ginis_from_df(gini_df, output_folder=out_dir + 'gini/', out_fn=ds_name + '.pdf', lw=3, ms=15, font_size=15,
                             zoom=80)
        if colored_categories is None:
            colored_categories = dict()
        colored_categories[None] = None
        valid_chars = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(map(str, range(10))) + ['_', '.']
        valid_chars = set(valid_chars)
        for key, val in colored_categories.iteritems():
            stat_dist_tmp_df = stat_dist_df.copy()
            current_ds_name = ds_name
            if key is not None:
                if 'category' in stat_dist_df.columns:
                    stat_dist_tmp_df['category'] = stat_dist_tmp_df['category'].apply(
                        lambda x: x if x == val else 'Other')

                val = ''.join([i if i in valid_chars else '' for i in val])
                current_ds_name = ds_name + '/' + val
            else:
                if 'category' in stat_dist_tmp_df.columns:
                    stat_dist_tmp_df.drop('category', inplace=True, axis=1)
            stat_dist_orig_columns = list(stat_dist_tmp_df.columns)

            stat_dist_tmp_df.columns = [name_mapping[i] if i in name_mapping else i for i in stat_dist_orig_columns]
            not_baseline_cols = list(filter(lambda x: x != base_line, stat_dist_df.columns))
            print(stat_dist_tmp_df.head(2))
            create_scatters_from_df(stat_dist_tmp_df, not_baseline_cols,
                                    output_folder=out_dir + 'scatter/' + current_ds_name + '/')

            stat_dist_tmp_df.columns = [bf_name_mapping[i] if i in bf_name_mapping else i for i in stat_dist_orig_columns]
            not_baseline_cols = list(filter(lambda x: x != base_line, stat_dist_df.columns))
            print(stat_dist_tmp_df.head(2))
            bias_factors_df = create_bf_scatters_from_df(stat_dist_tmp_df, base_line, not_baseline_cols,
                                                         output_folder=out_dir + 'bf_scatter/' + current_ds_name + '/')
            min_y, max_y = bias_factors_df[bias_factors_df > 0].min().min(), bias_factors_df.max().max()

            bias_factors_df = create_bf_scatters_from_df(stat_dist_tmp_df, base_line, not_baseline_cols,
                                                         output_folder=out_dir + 'bf_scatter/' + current_ds_name + '/',
                                                         y_range=[min_y, max_y])

    except:
        print traceback.format_exc()



base_dir = '/home/fgeigl/navigability_of_networks/output/steering_rnd_surfer/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)


name_mapping = dict()
name_mapping['adjacency'] = 'Unbiased'
name_mapping['deg'] = 'Degree'
name_mapping['eigenvector'] = 'Eigenvector Centrality'
name_mapping['inv_sqrt_deg'] = 'Inverse Degree'
name_mapping['sigma'] = 'Sigma'
name_mapping['sigma_sqrt_deg_corrected'] = 'Degree Corrected Sigma'
name_mapping['getdigital'] = 'GD'
name_mapping['karate.edgelist'] = 'Toy Example'
name_mapping['milan_spiele'] = 'MS'
name_mapping['thinkgeek'] = 'TG'
name_mapping['wiki4schools'] = 'WFS'
name_mapping['bar_wiki'] = 'BW'
name_mapping['tvthek_orf'] = 'ORF'
name_mapping['daserste'] = 'DEM'

bf_name_mapping = dict()
bf_name_mapping['adjacency'] = 'Unbiased'
bf_name_mapping['deg'] = 'Degree Bias Factor'
bf_name_mapping['eigenvector'] = 'Eigenvector Centrality B.F.'
bf_name_mapping['inv_sqrt_deg'] = 'Inverse Degree Bias Factor'
bf_name_mapping['sigma'] = 'Sigma Bias Factor'
bf_name_mapping['sigma_sqrt_deg_corrected'] = 'Degree Corrected Sigma B.F.'

sorting = dict()
sorting['adjacency'] = 0
sorting['deg'] = 1
sorting['eigenvector'] = 5
sorting['inv_sqrt_deg'] = 2
sorting['sigma'] = 3
sorting['sigma_sqrt_deg_corrected'] = 4
sorting['getdigital'] = 2
sorting['karate.edgelist'] = 0
sorting['milan_spiele'] = 3
sorting['thinkgeek'] = 1
sorting['wiki4schools'] = 4
sorting['bar_wiki'] = 5
sorting['tvthek_orf'] = 6
sorting['daserste'] = 7

special_cats = defaultdict(lambda: None)
# special_cats['tvthek_orf'] = {'topic_one': 'Nationalrat', 'topic_two': 'Steiermark-heute', 'topic_three': 'Raetselburg'}

base_line = name_mapping[base_line] if base_line in name_mapping else base_line

stat_dist_files = find_files(base_dir, 'stat_dists.df')
entropy_files = find_files(base_dir, 'entropy.df')
gini_files = find_files(base_dir, 'gini.df')
network_files = filter(lambda x: 'rewire' not in x, find_files(base_dir, '.gt'))
plot_degree_distributions(network_files, out_dir + 'deg_distributions/')

#for n_fn in network_files:
#    check_aperiodic(n_fn)

entropy_rates = None
for idx, fn in enumerate(entropy_files):
    entropy_rate_df = pd.read_pickle(fn)
    if entropy_rates is None:
        entropy_rates = pd.DataFrame(columns=list(entropy_rate_df.columns))
    entropy_rates.loc[fn.rsplit('/', 1)[-1].replace('_entropy.df', '')] = entropy_rate_df.loc[0]
try:
    entropy_rates.drop('karate.edgelist', inplace=True)
except ValueError:
    pass
print entropy_rates.columns
entropy_rates = entropy_rates[sorted(entropy_rates.columns, key=lambda x: sorting[x] if x in sorting else 1000)].copy()
entropy_rates.columns = [name_mapping[i] if i in name_mapping else i for i in entropy_rates.columns]
entropy_rates.columns = [i.split('Bias')[0].strip() for i in entropy_rates.columns]
entropy_rates.index = [name_mapping[i] if i in name_mapping else i for i in entropy_rates.index]
rewired_idx = filter(lambda x: 'rewired' in x, entropy_rates.index)
entropy_rates.drop(rewired_idx, inplace=True)
sorting = {name_mapping[key]: val for key, val in sorting.iteritems()}
entropy_rates = entropy_rates.loc[sorted(entropy_rates.index, key=lambda x: sorting[x] if x in sorting else 1000)].copy()
plot_entropy_rates(entropy_rates, out_dir + 'entropy.pdf')

worker_pool = multiprocessing.Pool(processes=15)

for fn in stat_dist_files[:1]:
    ds_name = fn.rsplit('/', 1)[-1].replace('_stat_dists.df', '')
    print ds_name.center(120, '#')
    worker_pool.apply_async(create_plots, args=(fn, special_cats[ds_name],))
worker_pool.close()
worker_pool.join()
print 'ALL DONE'
