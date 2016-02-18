from __future__ import division, print_function

from sys import platform as _platform

import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from matplotlib import gridspec
from post_processing.plotting import *
import os
from tools.basics import create_folder_structure, find_files, try_catch_traceback_wrapper
import traceback
import numpy as np
from graph_tool.all import *
import datetime
import network_matrix_tools
import random
import tools.mpl_tools as plt_tools
from scipy.sparse import diags
import sys
import itertools
import link_insertion_strategies
import multiprocessing as mp

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
matplotlib.rcParams.update({'font.size': 25})
default_x_ticks_pad = matplotlib.rcParams['xtick.major.pad']
default_y_ticks_pad = matplotlib.rcParams['xtick.major.pad']
matplotlib.rcParams['xtick.major.pad'] *= 2
matplotlib.rcParams['ytick.major.pad'] *= 2


def plot_df_fac(df, filename):
    df['stat_dist_fac'] = df['stat_dist'] / df['unbiased_stat_dist']
    x = df['com-size']
    c = df['bias_strength']
    y = df['stat_dist_fac']
    plt.scatter(x, y, c=c, lw=0, alpha=0.2, cmap='gist_rainbow')
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.xlabel('com-size')
    plt.ylabel(r'$\tau$')
    plt.title('all datasets')
    cbar = plt.colorbar()
    cbar.set_label('bias strength')
    plt.tight_layout()
    plt.plt.savefig(filename.rsplit('.', 1)[0] + '_scatter.png', dpi=150)
    plt.close('all')
    ax = None

    for i in sorted(set(df['bias_strength'])):
        filt_df = df[df['bias_strength'] == i]
        grp = filt_df.groupby('com-size')
        tmp_df = grp.agg(np.mean)
        # print tmp_df
        # tmp_df.to_pickle(filename.rsplit('/', 1)[0] + '/tmp.df')
        x = tmp_df.index
        y = tmp_df['stat_dist_fac']
        err = np.array(grp.agg(np.std)['stat_dist_fac'])
        tmp_df = pd.DataFrame(data=zip(np.array(x), np.array(y), err), columns=['com-size', 'stat_dist', 'err'])
        ax = tmp_df.plot(x='com-size', y='stat_dist', yerr='err', label='bs: ' + str(i), ax=ax, lw=2)
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.xlabel('category size')
    plt.ylabel(r'$\tau$')
    plt.title('all datasets')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close('all')


class GetOutOfLoops(Exception):
    pass


def add_links_and_calc((sample_size, com_nodes), net=None, bias_strength=2, method='rnd', num_links=1,
                       top_measure=None):
    if sample_size > 0.21:
        # print 'skip sample-size:', sample_size
        return [np.nan, np.nan]
    if not np.isclose(sample_size, add_links_and_calc.sample_size):
        add_links_and_calc.sample_size = sample_size
        add_links_and_calc.calc_counter = 0
        add_links_and_calc.ss_string = "%.3f" % sample_size
        print('')
    add_links_and_calc.calc_counter += 1
    if isinstance(num_links, str):
        if num_links == 'fair':
            bias_m = np.zeros(net.num_vertices()).astype('int')
            bias_m[com_nodes] = 1
            bias_m = diags(bias_m, 0)
            num_links = int(bias_m.dot(adjacency(net)).sum()) * (bias_strength - 1)
            # print 'fair links:', num_links
    print('\r', add_links_and_calc.ss_string, add_links_and_calc.calc_counter, '#links:', int(num_links / 1000), 'k',
          end='')
    sys.stdout.flush()

    new_edges = list()
    orig_num_edges = net.num_edges()
    orig_num_com_nodes = len(com_nodes)
    if orig_num_com_nodes >= net.num_vertices():
        return None
    com_nodes_set = set(com_nodes)
    other_nodes = set(range(0, net.num_vertices())) - com_nodes_set

    com_internal_links = True
    if com_internal_links:
        other_nodes |= com_nodes_set

    if method == 'rnd':
        new_edges = link_insertion_strategies.get_random_links(com_nodes_set, other_nodes, num_links)
    elif method == 'top_block':
        new_edges = link_insertion_strategies.get_top_block_links(com_nodes_set, other_nodes, num_links, top_measure)

    assert len(new_edges) == num_links
    tmp_net = net.copy()
    tmp_net.add_edge_list(new_edges)
    adj = adjacency(tmp_net)
    orig_adj = adjacency(net)
    parallel_edges_perc = (int((adj - adj.astype('bool').astype('int')).sum()) - int(
            (orig_adj - orig_adj.astype('bool').astype('int')).sum())) / num_links * 100
    print(' | added parallel edges:', int(parallel_edges_perc), '%', end='')
    sys.stdout.flush()
    _, relinked_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adj, method='EV',
                                                                            smooth_bias=False,
                                                                            calc_entropy_rate=False, verbose=False)
    assert orig_num_com_nodes == len(com_nodes)
    relinked_stat_dist_sum = relinked_stat_dist[com_nodes].sum()
    return [relinked_stat_dist_sum, parallel_edges_perc]


def plot_dataframe(df_fn, net, bias_strength, filename):
    df = pd.read_pickle(df_fn)
    label_dict = dict()
    label_dict['ratio_com_out_deg_in_deg'] = r'degree ratio ($d_t^r$)'
    label_dict['com_in_deg'] = r'$d_t^-$'
    label_dict['com_out_deg'] = r'$d_t^+$'
    gb = df[['sample-size', 'stat_dist_com_sum']].groupby('sample-size')
    trans_lambda = lambda x: (x-x.mean()) / x.std()
    gb = gb.transform(trans_lambda)
    # print gb
    stat_dist = np.array(gb['stat_dist_com_sum']).astype('float')
    stat_dist[np.invert(np.isfinite(stat_dist))] = np.nan
    df['stat_dist_normed'] = stat_dist
    df.dropna(axis=0, how='any', inplace=True)
    # print df
    orig_columns = set(df.columns)

    df_plot = df[df['sample-size'] < .21]

    in_deg = np.array(net.degree_property_map('in').a)
    # df_plot['in_neighbours_in_deg'] = df_plot['com_in_neighbours'].apply(lambda x: in_deg[list(x)].sum())

    out_deg = np.array(net.degree_property_map('out').a)
    # df_plot['out_neighbours_out_deg'] = df_plot['com_out_neighbours'].apply(lambda x: out_deg[list(x)].sum())

    # df_plot['ratio_out_out_deg_in_in_deg'] = df_plot['out_neighbours_out_deg'] / df_plot['in_neighbours_in_deg']

    df_plot['com_in_deg'] = df_plot['node-ids'].apply(lambda x: in_deg[list(x)].sum()) - df_plot['intra_com_links']
    df_plot['com_out_deg'] = df_plot['node-ids'].apply(lambda x: out_deg[list(x)].sum()) - df_plot['intra_com_links']
    df_plot['ratio_com_out_deg_in_deg'] = df_plot['com_out_deg'] / df_plot['com_in_deg']
    orig_columns.remove('orig_stat_dist_sum')

    df_plot['stat_dist_sum_fac'] = df_plot['stat_dist_com_sum'] / df_plot['orig_stat_dist_sum']
    df_plot['stat_dist_diff'] = df_plot['stat_dist_com_sum'] - df_plot['orig_stat_dist_sum']
    orig_columns.add('stat_dist_sum_fac')
    orig_columns.add('stat_dist_diff')

    ds_name = filename.rsplit('/', 1)[-1].rsplit('.gt', 1)[0]
    results = dict()
    for col_name in sorted(set(df_plot.columns) - orig_columns):
        current_filename = filename[:-4] + '_' + col_name.replace(' ', '_')
        current_filename = current_filename.rsplit('/', 1)
        current_filename = current_filename[0] + '/' + ds_name + '/' + current_filename[1].replace('.gt', '')
        create_folder_structure(current_filename)
        print('plot:', col_name)
        if False:
            for normed_stat_dist in [True, False]:
                y = df_plot[col_name]
                x = df_plot['sample-size']
                c = df_plot['stat_dist_normed'] if normed_stat_dist else df_plot['stat_dist_com_sum']
                fix, ax = plt.subplots()
                ac = ax.scatter(x, y, c=c, lw=0, alpha=0.7, cmap='coolwarm')
                ax.set_xticks(sorted(set(x)), minor=True)
                cbar = plt.colorbar(ac)
                plt.xlabel(r'$\phi$')
                plt.xlim([0, df_plot['sample-size'].max() + 0.01])
                y_range_one_perc = (df_plot[col_name].max() - df_plot[col_name].min()) * 0.01
                plt.ylim([df_plot[col_name].min() - y_range_one_perc, df_plot[col_name].max() + y_range_one_perc])
                plt.ylabel(col_name.replace('_', ' '))
                cbar.set_label(r'\phi standardized $\sum \pi$' if normed_stat_dist else r'$\pi_G$')
                # cbar.set_label('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')

                plt.title(ds_name + '\nBias Strength: ' + str(int(bias_strength)))
                plt.tight_layout()
                out_f = (current_filename + '_normed.png') if normed_stat_dist else (current_filename + '.png')

                plt.grid(which='minor', axis='x')
                plt.savefig(out_f, dpi=150)
                plt.close('all')

        df_plot.sort_values(by=col_name, inplace=True)

        label_dict['stat_dist_com_sum'] = r"energy ($\pi'_t$)"
        label_dict['add_top_block_links_fair'] = r"energy ($\pi'_t$)"
        label_dict['stat_dist_sum_fac'] = r'navigational potential ($\tau$)'
        label_dict['add_top_block_links_fair_fac'] = r'navigational potential ($\tau$)'
        label_dict['stat_dist_diff'] = r"$\pi'_t - \pi_t$"
        results[col_name] = plot_lines_plot(df_plot, col_name, 'stat_dist_com_sum', current_filename, '_lines', label_dict=label_dict,)
        # plot_lines_plot(df_plot, col_name, 'stat_dist_diff', current_filename, '_lines_diff', label_dict=label_dict,
        #                ds_name=ds_name)
        plot_lines_plot(df_plot, col_name, 'stat_dist_sum_fac', current_filename, '_lines_fac', label_dict=label_dict)

        plot_lines_plot(df_plot, col_name, 'add_top_block_links_fair', current_filename, '_lines_link_ins',
                        label_dict=label_dict, ylim1=[0, 80], ylim2=[0.4, 1.], ylim3=[0.25, 0.9])
        df_plot['add_top_block_links_fair_fac'] = df_plot['add_top_block_links_fair'] / df_plot['stat_dist_com_sum']
        plot_lines_plot(df_plot, col_name, 'add_top_block_links_fair_fac', current_filename, '_lines_link_ins_fac',
                        label_dict=label_dict)
    return results


def plot_lines_plot(df, x_col_name, y_col_name, out_fn_base, out_fn_ext, plt_font_size=25,
                    label_dict=None, plt_std=True, ylim1=None, ylim2=None, ylim3=None):
    if x_col_name not in {'ratio_com_out_deg_in_deg', 'com_in_deg', 'com_out_deg'}:
        return
    if 'ratio' not in x_col_name:
        pass
    default_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': plt_font_size})
    if label_dict is None:
        label_dict = dict()
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])
    ax1 = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax3)
    ax3.plot([np.nan], [np.nan], label=r'fraction of target nodes ($\phi$)', c='white', alpha=0.)

    lw_func = lambda x: 2
    legend_plot = True

    min_x_val, max_x_val = df[x_col_name].min(), df[x_col_name].max()
    min_y_val, max_y_val = df[y_col_name].min(), df[y_col_name].max()
    num_bins = 6
    x_val_range = max_x_val - min_x_val
    y_val_range = max_y_val - min_y_val

    if 'ratio' in x_col_name:
        plt_x_range = [min_x_val, max_x_val]
    else:
        x_offset = x_val_range / 100 * 2
        plt_x_range = [min_x_val - x_offset, max_x_val + x_offset]

    y_offset = y_val_range / 100 * 2
    plt_y_range = [min_y_val - y_offset, max_y_val + y_offset]

    # plt_x_center = plt_x_range[0] + ((plt_x_range[1] - plt_x_range[0]) / 2)
    df['sample-size'] = np.round(df['sample-size'], decimals=3)
    sample_sizes = {0.01, 0.1, 0.2}
    df = df[map(lambda x: x in sample_sizes, df['sample-size'])]
    colors = ['#e41a1c','#377eb8','#4daf4a']
    markers = "o^s"
    print(x_col_name, y_col_name, sorted(set(df['sample-size'])))
    avg_measure = dict()
    for style_idx, (key, grp) in enumerate(df[['sample-size', x_col_name, y_col_name]].groupby('sample-size')):
        key = np.round(key, decimals=3)
        if key not in sample_sizes:
            continue
        key_str = ('%.3f' % key).rstrip('0')

        grp_x_min = grp[x_col_name].min()
        grp_x_max = grp[x_col_name].max()
        avg_measure[key] = grp[x_col_name].mean()
        bins_step_size = (grp_x_max - grp_x_min) / num_bins
        start_point = grp_x_min + bins_step_size / 2
        bin_points = np.array([start_point + i * bins_step_size for i in range(num_bins)])

        c = colors[style_idx % len(colors)]
        m = markers[style_idx % len(markers)]
        grp['bin'] = grp[x_col_name].apply(lambda x: min(int((x - grp_x_min) / bins_step_size), num_bins - 1))
        grp_obj = grp[['bin', y_col_name]].groupby('bin')
        tmp_grp = grp_obj.mean()
        tmp_grp['std'] = grp_obj.std()
        tmp_grp['count'] = grp_obj.count()

        tmp_grp['bin_center'] = tmp_grp.index
        tmp_grp['bin_center'] = tmp_grp['bin_center'].apply(lambda x: bin_points[x])
        tmp_grp = tmp_grp.sort_values(by='bin_center')

        ax2_plt_kwargs = dict(x='bin_center', y=y_col_name, color=c, lw=lw_func(key), solid_capstyle="round", alpha=.95,
                              label='  ' + key_str, marker=m, markersize=12, markeredgewidth=2,
                              markeredgecolor=(.99, .99, .99, .9))

        if key < 0.09:
            ax2 = tmp_grp.plot(ax=ax2, **ax2_plt_kwargs)
            tmp_kwargs = ax2_plt_kwargs.copy()
            tmp_kwargs.pop('x')
            tmp_kwargs.pop('y')
            ax3.plot([np.nan], [np.nan], **tmp_kwargs)
        else:
            ax3 = tmp_grp.plot(ax=ax3, **ax2_plt_kwargs)

        if plt_std:
            tmp_grp['std'].fillna(0., inplace=True)
            tmp_ax = ax2 if key < 0.09 else ax3
            tmp_ax.fill_between(tmp_grp['bin_center'], tmp_grp[y_col_name] - tmp_grp['std'],
                                tmp_grp[y_col_name] + tmp_grp['std'], color=c, alpha=0.25, label=None,
                                interpolate=True)

        ax1 = tmp_grp.plot(x='bin_center', y='count', ax=ax1, lw=lw_func(key), color=c, solid_capstyle="round",
                           alpha=.95, marker=m, markersize=12)
    x_label = label_dict[x_col_name] if x_col_name in label_dict else x_col_name.replace('_', ' ')
    plt.xlabel(x_label)
    y_label = label_dict[y_col_name] if y_col_name in label_dict else y_col_name.replace('_', ' ')

    ax2.legend_.remove()
    ax3.legend_.remove()
    ax1.legend_.remove()
    ax1.set_xlim(plt_x_range)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel('N')
    ax1.grid(b=True, which='major', axis='y', linewidth=3, alpha=0.2, ls='--')
    ax1.set_axisbelow(True)
    ax1.set_xlim(plt_x_range)
    # ax2.set_ylim(plt_y_range)
    ax2.grid(b=True, which='major', axis='y', linewidth=3, alpha=0.2, ls='--')
    ax3.grid(b=True, which='major', axis='y', linewidth=3, alpha=0.2, ls='--')

    if ylim1: ax1.set_ylim(ylim1)
    if ylim2: ax3.set_ylim(ylim2)
    if ylim3: ax2.set_ylim(ylim3)

    if 'ratio' in x_col_name:
        plt.xticks()
    else:
        plt.xticks()
        ax2.ticklabel_format(style='sci', axis='x', useOffset=True, useMathText=True, scilimits=(0, 0))
        ax3.ticklabel_format(style='sci', axis='x', useOffset=True, useMathText=True, scilimits=(0, 0))
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)

    from matplotlib.ticker import MaxNLocator, AutoLocator

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    sample_dist_plot_color = 'gray'
    ax1.tick_params(axis='y', colors=sample_dist_plot_color)
    ax1.yaxis.label.set_color(sample_dist_plot_color)
    ax1.spines['bottom'].set_color(sample_dist_plot_color)
    ax1.spines['top'].set_color(sample_dist_plot_color)
    ax1.spines['left'].set_color(sample_dist_plot_color)
    ax1.spines['right'].set_color(sample_dist_plot_color)

    ax3.set_ylabel(y_label, labelpad=30)
    plt.tight_layout(h_pad=0.001)
    gs.update(wspace=0.025, hspace=0.2)
    plt_fn = out_fn_base + out_fn_ext
    plt_tools.save_n_crop(plt_fn + '.pdf')
    if legend_plot and set(df['sample-size']) == sample_sizes:
        plt_tools.plot_legend(ax3, out_fn_base.rsplit('/', 2)[0] + '/' + out_fn_ext.strip('_') + '_legend.pdf',
                          font_size=12, nrows=1, legend_name_idx=0, legend_name_style='it')
    plt.close('all')
    matplotlib.rcParams.update({'font.size': default_font_size})
    return avg_measure


def preprocess_df(df, net, bias_strength):
    add_links_and_calc.sample_size = -1.
    df_cols = set(df.columns)
    dirty = False
    print(' preprocessing '.center(120, '='))
    assert 'stat_dist' in df.columns and 'node-ids' in df.columns
    if 'sample-size' not in df_cols:
        num_vertices = net.num_vertices()
        df['sample-size'] = df['node-ids'].apply(lambda x: np.round(len(x) / num_vertices, decimals=3))
    if 'stat_dist_com' not in df_cols:
        print('[preprocess]: filter com stat-dist')
        not_one_idx = df.index[np.invert(np.isclose(df['stat_dist'].apply(np.sum), 1.))]
        if len(not_one_idx) > 0:
            sys.stdout.flush()
            print('\n')
            sys.stdout.flush()
            print('!!! filter out nans...', not_one_idx, '\n')
            sys.stdout.flush()
            df = df.drop(not_one_idx)
        assert np.allclose(np.array(df['stat_dist'].apply(np.sum)), 1.)
        df['stat_dist_com'] = df[['node-ids', 'stat_dist']].apply(
            lambda (node_ids, stat_dist): list(stat_dist[node_ids]), axis=1).apply(np.array)
        dirty = True
    if 'stat_dist_com_sum' not in df_cols:
        print('[preprocess]: sum com stat-dist')
        df['stat_dist_com_sum'] = df['stat_dist_com'].apply(np.sum)
        dirty = True
    if 'com_in_neighbours' not in df_cols:
        print('[preprocess]: find com in neighbours')
        print('[preprocess]: \tcreate mapping vertex->in-neighbours')
        in_neighbs = {int(v): set(map(int, v.in_neighbours())) for v in net.vertices()}
        print('[preprocess]: \tcreate union of in-neigbs')
        df['com_in_neighbours'] = df['node-ids'].apply(
            lambda x: set.union(*map(lambda v_id: in_neighbs[v_id], x)))
        print('[preprocess]: \tfilter out com nodes')
        df['com_in_neighbours'] = df[['com_in_neighbours', 'node-ids']].apply(
            lambda (com_in_neighbours, com_nodes): com_in_neighbours - set(com_nodes), axis=1)
        dirty = True
    out_neighbs = None
    if 'com_out_neighbours' not in df_cols:
        print('[preprocess]: find com out neighbours')
        print('[preprocess]: \tcreate mapping vertex->out-neighbours')
        out_neighbs = {int(v): set(map(int, v.out_neighbours())) for v in net.vertices()}
        print('[preprocess]: \tcreate union of out-neigbs')
        df['com_out_neighbours'] = df['node-ids'].apply(
            lambda x: set.union(*map(lambda v_id: out_neighbs[v_id], x)))
        print('[preprocess]: \tfilter out com nodes')
        df['com_out_neighbours'] = df[['com_out_neighbours', 'node-ids']].apply(
            lambda (com_out_neighbours, com_nodes): com_out_neighbours - set(com_nodes), axis=1)
        dirty = True
    if 'intra_com_links' not in df_cols:
        print('[preprocess]: count intra com links')
        if out_neighbs is None:
            print('[preprocess]: \tcreate mapping vertex->out-neighbours')
            out_neighbs = {int(v): set(map(int, v.out_neighbours())) for v in net.vertices()}
        df['intra_com_links'] = df['node-ids'].apply(set).apply(
            lambda x: np.array([len(out_neighbs[i] & x) for i in x]).sum())
        dirty = True
    orig_stat_dist = None
    if 'orig_stat_dist_sum' not in df_cols:
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df['orig_stat_dist_sum'] = df['node-ids'].apply(lambda x: orig_stat_dist[x].sum())
        dirty = True

    force_recalc = False
    col_label = 'add_rnd_links_fair'
    if col_label not in df_cols or force_recalc:
        print(datetime.datetime.now().replace(microsecond=0), 'calc stat dist with fair inserted random links')
        adj = adjacency(net)
        print('init # parallel links:',  int((adj - adj.astype('bool').astype('int')).sum()))
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adj, method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df[[col_label, col_label + '_parallel_perc']] = df[['sample-size', 'node-ids']].apply(add_links_and_calc, axis=1, args=(net, bias_strength, 'rnd', 'fair', None))
        dirty = True
        print('')
        print(datetime.datetime.now().replace(microsecond=0), '[OK]')

    force_recalc = False
    col_label = 'add_top_block_links_fair'
    if col_label not in df_cols or force_recalc:
        print(datetime.datetime.now().replace(microsecond=0), 'calc stat dist with fair inserted top block links')
        adj = adjacency(net)
        print('def parallel links:',  int((adj - adj.astype('bool').astype('int')).sum()))
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adj, method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df[[col_label, col_label + '_parallel_perc']] = df[['sample-size', 'node-ids']].apply(add_links_and_calc, axis=1, args=(net, bias_strength, 'top_block', 'fair', orig_stat_dist))
        dirty = True
        print('')
        print(datetime.datetime.now().replace(microsecond=0), '[OK]')

    if not dirty:
        print(' preprocessing nothing to do '.center(120, '='))
    else:
        print(' preprocessing done '.center(120, '='))
    return df, dirty


def plot_inserted_links(df, columns, filename):
    filename = filename.replace('.gt', '')
    print('plot inserted links:', filename)
    filt_df = df[df['sample-size'] < 0.21]
    used_columns = ['orig_stat_dist_sum', 'stat_dist_com_sum'] + columns
    grp_df = filt_df.groupby('sample-size')[used_columns]
    used_columns = [i.replace('add_', '').replace('_links_', ' ') for i in used_columns]
    grp_df.columns = ['unbiased', 'biased'] + used_columns[2:]

    grp_mean = grp_df.mean()
    grp_std = grp_df.std()
    # grp_mean.to_excel(filename + '_inserted_links.xls')
    fig, ax = plt.subplots()
    ax.plot([np.nan], [np.nan], label=r'manipulation', c='white', alpha=0.)
    print(grp_df.columns)
    label_dict = dict()
    label_dict['unbiased'] = 'unmodified network'
    label_dict['biased'] = 'navigational bias'
    # label_dict['rnd fair'] = 'random'
    label_dict['top_block fair'] = 'link insertion'
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
    markers = "ov^<>sp*+x"
    for idx, (i, label) in enumerate(filter(lambda x: x[1] in label_dict, zip(grp_mean.columns, grp_df.columns))):
        c = colors[idx]
        marker = markers[idx]
        label = label_dict[label]
        ax.plot(np.array(grp_mean.index), np.array(grp_mean[i]), label=label, marker=marker, ms=12, lw=3, color=c,
                alpha=0.9, solid_capstyle="round", markeredgewidth = 2, markeredgecolor = (.99, .99, .99, .9))
        ax.fill_between(np.array(grp_mean.index), np.array(grp_mean[i]) - np.array(grp_std[i]),
                        np.array(grp_mean[i]) + np.array(grp_std[i]), color=c, alpha=0.25, label=None,
                        interpolate=True)
    ax.grid(b=True, which='major', axis='y', linewidth=3, alpha=0.2, ls='--')
    plt.xlabel(r'fraction of target nodes ($\phi$)')
    plt.ylabel(r"energy ($\pi'_t$)")
    ax.set_xlim([grp_mean.index.min(), grp_mean.index.max()])
    ax.set_axisbelow(True)
    ax.set_ylim([0., 0.7])
    plt.tight_layout()
    out_fn = filename + '_inserted_links.pdf'
    plt_tools.save_n_crop(out_fn)
    legend_fname = filename.rsplit('/', 1)[0] + '/inserted_links_legend.pdf'
    plt_tools.plot_legend(ax, legend_fname, font_size=12, legend_name_idx=0, legend_name_style='it')


def worker_func(df_filename, net, bias_strength, out_dir):
    try:
        preprocessed_filename = df_filename.rsplit('.df', 1)[0] + '_preprocessed.df'
        if os.path.isfile(
                preprocessed_filename):  # and time.ctime(os.path.getmtime(preprocessed_filename)) > time.ctime(os.path.getmtime(i)):
            print('read preprocessed file:', preprocessed_filename.rsplit('/', 1)[-1])
            try:
                df = pd.read_pickle(preprocessed_filename)
            except:
                print(traceback.format_exc())
                print('fallback: read orig file:', df_filename.rsplit('/', 1)[-1])
                df = pd.read_pickle(df_filename)
        else:
            print('read:', df_filename.rsplit('/', 1)[-1])
            df = pd.read_pickle(df_filename)

        df, is_df_dirty = preprocess_df(df, net, bias_strength)
        if is_df_dirty:
            print('store preprocessed df')
            while True:
                # in case of str+c
                try:
                    cache_fname = preprocessed_filename + '.cache'
                    df.to_pickle(cache_fname)
                    shutil.move(cache_fname, preprocessed_filename)
                    break
                except:
                    print(traceback.format_exc())
                    print('Warning currently writing file: retry')
        print('plot:', df_filename.rsplit('/', 1)[-1])
        print('all cols:', df.columns)
        print('=' * 80)

        # use to skip plotting
        if True:
            out_fn = out_dir + df_filename.rsplit('/', 1)[-1][:-3]
            insert_links_labels = sorted(
                    filter(lambda x: x.startswith(('add_top_links_fair', 'add_rnd_links_fair', 'add_top_block_links_fair')),
                           df.columns))
            plot_inserted_links(df, insert_links_labels, out_fn)

            out_fn += '.png'
            return out_fn, bias_strength, plot_dataframe(preprocessed_filename, net, bias_strength, out_fn)
    except Exception as e:
        sys.stdout.flush()
        print('\n#######ERROR:\n', traceback.format_exc(), '\n')
        sys.stdout.flush()
        exit()


def main():
    base_dir = '/home/fgeigl/navigability_of_networks/output/opt_link_man/'
    out_dir = base_dir + 'plots/'
    create_folder_structure(out_dir)

    result_files = filter(lambda x: '_bs' in x, find_files(base_dir, '.df'))
    print(result_files)
    net_name = ''
    net = None
    skipped_ds = set()
    worker_pool = mp.Pool(processes=1)
    #skipped_ds.add('daserste')
    #skipped_ds.add('wiki4schools')
    #skipped_ds.add('tvthek_orf')
    apply_results = list()
    results_list = list()
    for df_filename in sorted(filter(lambda x: 'preprocessed' not in x, result_files),
                              key=lambda x: (
                                      x.rsplit('_bs', 1)[0], 1000 - int(x.rsplit('_bs', 1)[-1].rsplit('.', 1)[0])),
                              reverse=True):
        current_net_name = df_filename.rsplit('_bs', 1)[0]
        bias_strength = int(df_filename.split('_bs')[-1].split('.')[0])
        if False and not (149 < bias_strength < 200) and not np.isclose(bias_strength, 5.):
            print('skip bs:', bias_strength)
            continue
        elif any((i in current_net_name for i in skipped_ds)):
            print('skip ds:', current_net_name)
            continue
        else:
            print('add bs:', current_net_name, bias_strength)
        if current_net_name != net_name:
            print('*' * 120)
            print('load network:', current_net_name.rsplit('/', 1)[-1])
            net = load_graph(current_net_name)
            net_name = current_net_name
        assert net is not None
        apply_results.append(worker_pool.apply_async(func=worker_func, args=(df_filename, net, bias_strength, out_dir),
                                                     callback=results_list.append))
    worker_pool.close()
    worker_pool.join()
    assert all(i.successful() for i in apply_results)
    reduced_results = dict()
    for name, bs, data in results_list:
        reduced_results[name.rsplit('/', 1)[-1].split('.gt', 1)[0]] = data
    print(reduced_results)

    df = pd.DataFrame(columns=reduced_results[reduced_results.keys()[0]].keys())
    for c in df.columns:
        df[c] = df[c].astype('object')
    for name, data in reduced_results.iteritems():
        for d_name, val in data.iteritems():
            try:
                df[d_name] = df[d_name].astype('object')
                df.at[str(name), str(d_name)] = [val]
                df[d_name] = df[d_name].astype('object')
            except:
                print(traceback.format_exc())
                print('ERROR:', name, d_name, val)
                print(type(name), type(d_name), type(val))
    print(df.head(2))
    df.to_pickle(base_dir + 'avg_measures.df')


if __name__ == '__main__':
    start = datetime.datetime.now()
    print('START:', start)
    main()
    print('sort and manage results')
    import results_sorter
    print('ALL DONE. Time:', datetime.datetime.now() - start)

