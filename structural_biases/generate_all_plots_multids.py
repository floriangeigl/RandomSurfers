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
matplotlib.rcParams['xtick.major.pad'] *= 2
matplotlib.rcParams['ytick.major.pad'] *= 2


def create_plots(fn, colored_categories=None):
    try:
        ds_name = fn.rsplit('/', 1)[-1].replace('stat_dists.df', '').strip('_')
        stat_dist_df = pd.read_pickle(fn)
        stat_dist_orig_columns = list(stat_dist_df.columns)
        stat_dist_df.columns = [name_mapping[i] if i in name_mapping else i for i in stat_dist_orig_columns]
        print stat_dist_df.head()
        gini_df = stat_dist_df.copy()
        gini_df.columns = [i.split('Bias')[0].strip() for i in gini_df.columns]
        # create_ginis_from_df(gini_df, output_folder=out_dir + 'gini/', out_fn=ds_name + '.pdf', lw=3, ms=15, font_size=15,
        #                             zoom=80)
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
            # not_baseline_cols = list(filter(lambda x: x != base_line, stat_dist_df.columns))
            # print(stat_dist_tmp_df.head(2))
            # create_scatters_from_df(stat_dist_tmp_df, not_baseline_cols,
            #                         output_folder=out_dir + 'scatter/' + current_ds_name + '/')

            stat_dist_tmp_df.columns = [bf_name_mapping[i] if i in bf_name_mapping else i for i in
                                        stat_dist_orig_columns]
            not_baseline_cols = list(
                filter(lambda x: x != base_line and x != 'category' and 'prop_' not in x, stat_dist_df.columns))
            print(stat_dist_tmp_df.head(2))
            cor_out_dir = out_dir + 'bf_cor/' + current_ds_name + '/'
            if not os.path.isdir(cor_out_dir):
                os.makedirs(cor_out_dir)

            ranges = [None, None, None, None]
            for col in filter(lambda x: 'Degree' in x, not_baseline_cols):
                x = stat_dist_tmp_df[base_line]
                y = stat_dist_tmp_df[col]
                assert np.allclose(1., np.array([x.sum(), y.sum()]))
                plt_ranges = create_bf_scatter((base_line, x), (col, y),
                                  cor_out_dir + col.replace(' ', '_') + '.pdf')
                ranges = map(lambda (idx, x): x if ranges[idx] is None else (min(ranges[idx], x) if idx % 2 == 0 else max(ranges[idx], x)), enumerate(plt_ranges))
                # stat_dist_tmp_df.plot(x=base_line, y=col, loglog=True, kind='scatter', lw=0, alpha=.6)
#                plt.tight_layout()
#                plt.savefig(cor_out_dir + col.replace(' ', '_') + '_simple.png')
#                plt.close('all')
            # replot rescaled
            ranges = [min(ranges[0], ranges[2]), max(ranges[1], ranges[3])]
            for col in filter(lambda x: 'Degree' in x, not_baseline_cols):
                x = stat_dist_tmp_df[base_line]
                y = stat_dist_tmp_df[col]

                create_bf_scatter((base_line, x), (col, y),
                                  cor_out_dir + col.replace(' ', '_') + '.pdf', min_x=ranges[0], max_x=ranges[1],
                                  min_y=ranges[0], max_y=ranges[1])

            if False:
                bias_factors_df = create_bf_scatters_from_df(stat_dist_tmp_df, base_line, not_baseline_cols,
                                                             output_folder=out_dir + 'bf_scatter/' + current_ds_name + '/')
                min_y, max_y = bias_factors_df[bias_factors_df > 0].min().min(), bias_factors_df.max().max()

                bias_factors_df = create_bf_scatters_from_df(stat_dist_tmp_df, base_line, not_baseline_cols,
                                                             output_folder=out_dir + 'bf_scatter/' + current_ds_name + '/',
                                                             y_range=[min_y, max_y])
        print('--->', ds_name, 'all done')
    except:
        print('!!! --->', ds_name, 'FAILED')
        print traceback.format_exc()

base_dir = '/home/fgeigl/navigability_of_networks/output/steering_rnd_surfer/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)

name_mapping = dict()
name_mapping['adjacency'] = 'Original'
name_mapping['deg'] = 'Degree'
name_mapping['inv_deg'] = 'Inverse Degree'
name_mapping['log_deg'] = 'Degree'
name_mapping['eigenvector'] = 'Eigenvector Centrality'
name_mapping['inv_sqrt_deg'] = 'Inverse Degree'
name_mapping['inv_log_deg'] = 'Inverse Degree'
name_mapping['sigma'] = 'Sigma'
name_mapping['sigma_sqrt_deg_corrected'] = 'Degree Corrected Sigma'
name_mapping['getdigital'] = 'GD'
name_mapping['getdigital_eu_resolved_cleaned.gt'] = 'GD'
name_mapping['getdigital_eu'] = 'GD'
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
entropy_rate_files = find_files(base_dir, 'entropy_rate.df')
stat_entropy_files = find_files(base_dir, '_stat_entropy.df')
gini_files = find_files(base_dir, 'gini.df')
network_files = filter(lambda x: 'rewire' not in x, find_files(base_dir, '.gt'))
# plot_degree_distributions(network_files, out_dir + 'deg_distributions/')

#for n_fn in network_files:
#    check_aperiodic(n_fn)

if True:
    entropy_rates = None
    for idx, fn in enumerate(entropy_rate_files):
        entropy_rate_df = pd.read_pickle(fn)
        if entropy_rates is None:
            entropy_rates = pd.DataFrame(columns=list(entropy_rate_df.columns))
        entropy_rates.loc[fn.rsplit('/', 1)[-1].replace('_entropy_rate.df', '')] = entropy_rate_df.loc[0]
    try:
        entropy_rates.drop('karate.edgelist', inplace=True)
    except ValueError:
        pass
    print entropy_rates.columns
    entropy_rates = entropy_rates[sorted(entropy_rates.columns, key=lambda x: sorting[x] if x in sorting else 1000)].copy()
    entropy_rates = entropy_rates[filter(lambda x: 'prop_' not in x, entropy_rates.columns)]
    entropy_rates.columns = [name_mapping[i] if i in name_mapping else i for i in entropy_rates.columns]
    entropy_rates.columns = [i.split('Bias')[0].strip() for i in entropy_rates.columns]
    entropy_rates.index = [name_mapping[i] if i in name_mapping else i for i in entropy_rates.index]
    rewired_idx = filter(lambda x: 'rewired' in x, entropy_rates.index)
    entropy_rates.drop(rewired_idx, inplace=True)
    sorting_conv = {name_mapping[key]: val for key, val in sorting.iteritems()}
    entropy_rates = entropy_rates.loc[sorted(entropy_rates.index, key=lambda x: sorting_conv[x] if x in sorting_conv else 1000)].copy()
    plot_bar_plot(entropy_rates, out_dir + 'entropy_rate.pdf', 'entropy rate')

if True:
    stat_entropy = None
    for idx, fn in enumerate(stat_entropy_files):
        stat_entropy_df = pd.read_pickle(fn)
        if stat_entropy is None:
            stat_entropy = pd.DataFrame(columns=list(stat_entropy_df.columns))
        stat_entropy.loc[fn.rsplit('/', 1)[-1].replace('_stat_entropy.df', '')] = stat_entropy_df.loc[0]
    try:
        stat_entropy.drop('karate.edgelist', inplace=True)
    except ValueError:
        pass
    print stat_entropy.columns
    stat_entropy = stat_entropy[sorted(stat_entropy.columns, key=lambda x: sorting[x] if x in sorting else 1000)].copy()
    stat_entropy = stat_entropy[filter(lambda x: 'prop_' not in x, stat_entropy.columns)]
    stat_entropy.columns = [name_mapping[i] if i in name_mapping else i for i in stat_entropy.columns]
    stat_entropy.columns = [i.split('Bias')[0].strip() for i in stat_entropy.columns]
    stat_entropy.index = [name_mapping[i] if i in name_mapping else i for i in stat_entropy.index]
    rewired_idx = filter(lambda x: 'rewired' in x, stat_entropy.index)
    stat_entropy.drop(rewired_idx, inplace=True)
    sorting_conv = {name_mapping[key]: val for key, val in sorting.iteritems()}
    stat_entropy = stat_entropy.loc[sorted(stat_entropy.index, key=lambda x: sorting_conv[x] if x in sorting_conv else 1000)].copy()
    plot_bar_plot(stat_entropy, out_dir + 'stat_entropy.pdf', 'stationary entropy')

worker_pool = multiprocessing.Pool(processes=15)

worker_results = list()
for fn in filter(lambda x: 'wiki' in x or True, stat_dist_files):
    ds_name = fn.rsplit('/', 1)[-1].replace('_stat_dists.df', '')
    print ds_name.center(120, '#')
    worker_results.append(worker_pool.apply_async(create_plots, args=(fn, special_cats[ds_name],)))
    # create_plots(fn, special_cats[ds_name])
worker_pool.close()
worker_pool.join()
assert all((i.successful() for i in worker_results)) and len(worker_results) > 0
print 'ALL DONE'
