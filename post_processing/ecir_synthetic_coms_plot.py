from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from plotting import *
import os, sys
from tools.basics import create_folder_structure, find_files
import multiprocessing
import traceback
from utils import check_aperiodic
import numpy as np
import multiprocessing as mp
from graph_tool.all import *
import datetime
import time
import network_matrix_tools
import operator
import random
import tools.mpl_tools as plt_tools
from scipy.sparse import diags

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
matplotlib.rcParams.update({'font.size': 20})


def plot_df_fac(df, filename):
    df['stat_dist_fac'] = df['stat_dist'] / df['unbiased_stat_dist']
    x = df['com-size']
    c = df['bias_strength']
    y = df['stat_dist_fac']
    plt.scatter(x, y, c=c, lw=0, alpha=0.2, cmap='gist_rainbow')
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.xlabel('com-size')
    plt.ylabel('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')
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
        #print tmp_df
        #tmp_df.to_pickle(filename.rsplit('/', 1)[0] + '/tmp.df')
        x = tmp_df.index
        y = tmp_df['stat_dist_fac']
        err = np.array(grp.agg(np.std)['stat_dist_fac'])
        tmp_df = pd.DataFrame(data=zip(np.array(x), np.array(y), err), columns=['com-size', 'stat_dist', 'err'])
        ax = tmp_df.plot(x='com-size', y='stat_dist', yerr='err', label='bs: ' + str(i), ax=ax, lw=2)
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.xlabel('category size')
    plt.ylabel('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')
    plt.title('all datasets')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close('all')


class GetOutOfLoops(Exception):
    pass


def add_links_and_calc((sample_size, com_nodes), net=None, method='rnd', num_links=1, top_measure=None):
    if sample_size > 0.21:
        # print 'skip sample-size:', sample_size
        return np.nan
    if isinstance(num_links, str):
        if num_links == 'fair':
            bias_m = np.zeros(net.num_vertices()).astype('int')
            bias_m[com_nodes] = 1
            bias_m = diags(bias_m, 0)
            num_links = int(bias_m.dot(adjacency(net)).sum())
            # print 'fair links:', num_links
    new_edges = set()
    orig_num_edges = net.num_edges()
    orig_num_com_nodes = len(com_nodes)
    if orig_num_com_nodes >= net.num_vertices():
        return None
    other_nodes = set(range(0, net.num_vertices())) - set(com_nodes)

    if method == 'rnd':
        remain_edges = num_links - len(new_edges)
        while remain_edges:
            srcs = random.sample(other_nodes, min(remain_edges, len(other_nodes)))
            dests = random.sample(com_nodes, min(remain_edges, len(com_nodes)))
            new_edges.update(set(filter(lambda (s, d): net.edge(s, d) is None, zip(srcs, dests))))
            remain_edges = num_links - len(new_edges)

    elif method == 'top':
        if top_measure is None:
            nodes_measure = np.array(net.degree_property_map('in').a)
        else:
            nodes_measure = top_measure

        try:
            sorted_other_nodes = sorted(other_nodes, key=lambda x: nodes_measure[x], reverse=True)
            sorted_com_nodes = sorted(com_nodes, key=lambda x: nodes_measure[x], reverse=True)
            for dest in sorted_com_nodes:
                for src in sorted_other_nodes:
                    if net.edge(src, dest) is None:
                        new_edges.add((src, dest))
                        if len(new_edges) >= num_links:
                            raise GetOutOfLoops
            print 'could not insert all links:', len(new_edges), 'of', num_links
        except GetOutOfLoops:
            pass
    elif method == 'top_block':
        max_links = int(num_links / orig_num_com_nodes) + 5
        # print 'max links:', max_links
        if top_measure is None:
            nodes_measure = np.array(net.degree_property_map('in').a)
        else:
            nodes_measure = top_measure

        sorted_other_nodes = sorted(other_nodes, key=lambda x: nodes_measure[x], reverse=True)
        sorted_com_nodes = sorted(com_nodes, key=lambda x: nodes_measure[x], reverse=True)
        max_links -= 1
        for max_links_add in range(3):
            max_links += 1
            sorted_other_nodes_block = sorted_other_nodes[:max_links]
            # sorted_com_nodes_block = sorted_com_nodes
            new_edges = filter(lambda l_e: net.edge(*l_e) is None, ((src, dest) for dest in sorted_com_nodes for src in sorted_other_nodes_block))[:num_links]
            new_edges = set(new_edges)
            if len(new_edges) >= num_links:
                break
            else:
                print 'retry with bigger block'
        if len(new_edges) < num_links:
            print 'could not insert all links:', len(new_edges), 'of', num_links


    assert len(new_edges) == num_links
    net.add_edge_list(new_edges)
    _, relinked_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                            smooth_bias=False,
                                                                            calc_entropy_rate=False, verbose=False)
    assert orig_num_com_nodes == len(com_nodes)
    relinked_stat_dist_sum = relinked_stat_dist[com_nodes].sum()
    for src, dest in new_edges:
        e = net.edge(src, dest)
        net.remove_edge(e)
    assert net.num_edges() == orig_num_edges
    print '.',
    sys.stdout.flush()
    return relinked_stat_dist_sum


def plot_dataframe(df, net, bias_strength, filename):
    label_dict = dict()
    label_dict['ratio_com_out_deg_in_deg'] = r'$k_g^r$'
    label_dict['com_in_deg'] = r'$k_g^-$'
    label_dict['com_out_deg'] = r'$k_g^+$'
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
    df_plot['in_neighbours_in_deg'] = df_plot['com_in_neighbours'].apply(lambda x: in_deg[list(x)].sum())

    out_deg = np.array(net.degree_property_map('out').a)
    df_plot['out_neighbours_out_deg'] = df_plot['com_out_neighbours'].apply(lambda x: out_deg[list(x)].sum())

    df_plot['ratio_out_out_deg_in_in_deg'] = df_plot['out_neighbours_out_deg'] / df_plot['in_neighbours_in_deg']

    df_plot['com_in_deg'] = df_plot['node-ids'].apply(lambda x: in_deg[list(x)].sum()) - df_plot['intra_com_links']
    df_plot['com_out_deg'] = df_plot['node-ids'].apply(lambda x: out_deg[list(x)].sum()) - df_plot['intra_com_links']
    df_plot['ratio_com_out_deg_in_deg'] = df_plot['com_out_deg'] / df_plot['com_in_deg']

    df_plot['stat_dist_sum_fac'] = df_plot['stat_dist_com_sum'] / df_plot['orig_stat_dist_sum']
    orig_columns.add('stat_dist_sum_fac')

    ds_name = filename.rsplit('/', 1)[-1].rsplit('.gt',1)[0]
    for col_name in sorted(set(df_plot.columns) - orig_columns):
        current_filename = filename[:-4] + '_' + col_name.replace(' ', '_')
        current_filename = current_filename.rsplit('/', 1)
        current_filename = current_filename[0] + '/' + ds_name + '/' + current_filename[1].replace('.gt', '')
        create_folder_structure(current_filename)
        print 'plot:', col_name
        for normed_stat_dist in [True, False]:
            y = df_plot[col_name]
            x = df_plot['sample-size']
            c = df_plot['stat_dist_normed'] if normed_stat_dist else df_plot['stat_dist_com_sum']
            fix, ax = plt.subplots()
            ac = ax.scatter(x, y, c=c, lw=0, alpha=0.7, cmap='coolwarm')
            ax.set_xticks(sorted(set(x)), minor=True)
            cbar = plt.colorbar(ac)
            plt.xlabel('sample size')
            plt.xlim([0, df_plot['sample-size'].max() + 0.01])
            y_range_one_perc = (df_plot[col_name].max() - df_plot[col_name].min()) * 0.01
            plt.ylim([df_plot[col_name].min() - y_range_one_perc, df_plot[col_name].max() + y_range_one_perc])
            plt.ylabel(col_name.replace('_', ' '))
            cbar.set_label('sample size standardized $\\sum \\pi$' if normed_stat_dist else r'$\pi_g$')
            # cbar.set_label('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')

            plt.title(ds_name + '\nBias Strength: ' + str(int(bias_strength)))
            plt.tight_layout()
            out_f = (current_filename + '_normed.png') if normed_stat_dist else (current_filename + '.png')

            plt.grid(which='minor', axis='x')
            plt.savefig(out_f, dpi=150)
            plt.close('all')

        df_plot.sort(col_name, inplace=True)

        label_dict['stat_dist_com_sum'] = r'$\pi_g^b$'
        label_dict['stat_dist_sum_fac'] = r'$\frac{\pi_g^b}{\pi_g^u}$'
        plot_lines_plot(df_plot, col_name, 'stat_dist_com_sum', current_filename, '_lines', label_dict=label_dict,
                        ds_name=ds_name)
        plot_lines_plot(df_plot, col_name, 'stat_dist_sum_fac', current_filename, '_lines_fac', label_dict=label_dict,
                        ds_name=ds_name)
        # exit()
    return 0


def plot_lines_plot(df, x_col_name, y_col_name, out_fn_base,out_fn_ext, one_subplot=True, plt_font_size=20,
                    fig_size=(16, 10), label_dict=None, ds_name=''):
    default_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': plt_font_size})
    if label_dict is None:
        label_dict = dict()
    if one_subplot:
        fig, ax2 = plt.subplots()
        ax1 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    if not one_subplot:
        ax1.plot(None, label='sample-size', c='white')
    ax2.plot(None, label='sample-size', c='white')
    lw_func = lambda x: 1. + ((x - .01) / (.2 - .01)) * 3

    if 'ratio' in x_col_name:
        min_x_val, max_x_val = 0.5, 2.
    else:
        min_x_val, max_x_val = df[x_col_name].min(), df[x_col_name].max()
    min_y_val, max_y_val = df[y_col_name].min(), df[y_col_name].max()
    num_bins = 6
    x_val_range = max_x_val - min_x_val
    y_val_range = max_y_val - min_y_val

    x_annot_offset = x_val_range / 15

    if 'ratio' in x_col_name:
        plt_x_range = [min_x_val, max_x_val]
    else:
        x_offset = x_val_range / 100 * 2
        plt_x_range = [min_x_val - x_offset, max_x_val + x_offset]

    y_offset = y_val_range / 100 * 2
    plt_y_range = [min_y_val - y_offset, max_y_val + y_offset]

    plt_x_center = plt_x_range[0] + ((plt_x_range[1] - plt_x_range[0]) / 2)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    for color_idx, (key, grp) in enumerate(df[['sample-size', x_col_name, y_col_name]].groupby('sample-size')):
        use_arrows = False
        rnd_label_pos = False

        key = np.round(key, decimals=3)
        key_str = ('%.3f' % key).rstrip('0')
        grp_x_min = grp[x_col_name].min()
        grp_x_max = grp[x_col_name].max()
        bins_step_size = (grp_x_max - grp_x_min) / num_bins
        start_point = grp_x_min + bins_step_size / 2
        bin_points = np.array([start_point + i * bins_step_size for i in range(num_bins)])

        if not one_subplot:
            ax1 = grp.plot(x=x_col_name, y=y_col_name, ax=ax1, label='  ' + key_str)
        c = colors[color_idx % len(colors)]
        grp['bin'] = grp[x_col_name].apply(lambda x: min(int((x - grp_x_min) / bins_step_size), num_bins - 1))
        tmp_grp = grp[['bin', y_col_name]].groupby('bin').mean()
        tmp_grp['bin_center'] = tmp_grp.index
        # tmp_grp = tmp_grp[tmp_grp['bin_center'] < tmp_grp['bin_center'].max()]
        tmp_grp['bin_center'] = tmp_grp['bin_center'].apply(lambda x: bin_points[x])
        tmp_grp = tmp_grp.sort('bin_center')
        if len(tmp_grp) > 5 and rnd_label_pos:
            annotate_idx = np.random.randint(2, len(tmp_grp) - 3)
        else:
            annotate_idx = int(len(tmp_grp) / 2)
        center_row = tmp_grp.iloc[annotate_idx]
        x_center, y_center = center_row[['bin_center', y_col_name]]
        annotate_font_size = plt_font_size / 2
        last_x, last_y = tmp_grp.iloc[max(0, annotate_idx - 1)][['bin_center', y_col_name]]
        next_x, next_y = tmp_grp.iloc[min(len(tmp_grp) - 1, annotate_idx + 1)][['bin_center', y_col_name]]

        ax2_plt_kwargs = dict(x='bin_center', y=y_col_name, color=c, lw=lw_func(key), solid_capstyle="round", alpha=.9)
        annotate_bbox = dict(boxstyle='round4,pad=0.2', fc='white', ec=c, alpha=0.7)
        annotate_kwargs = dict(ha='center', va='center', bbox=annotate_bbox, fontsize=annotate_font_size)
        annotate_arrow = dict(arrowstyle="->, head_width=1.", facecolor='black',
                              connectionstyle='angle3') #tail_width=.25, shrink_factor=0.05

        if grp_x_max - grp_x_min < x_val_range / 100 * 5 and use_arrows:
            if len(tmp_grp) % 2 == 0:
                x_center = last_x + (next_x - last_x) / 2
                y_center = last_y + (next_y - last_y) / 2
            if len(tmp_grp) == 1:
                c_bin_center, c_y_val = tmp_grp.iloc[0][['bin_center', y_col_name]]
                tmp_grp = pd.DataFrame(columns=['bin_center', y_col_name],
                                       data=[(c_bin_center - bins_step_size, c_y_val),
                                             (c_bin_center + bins_step_size, c_y_val)])

            ax2 = tmp_grp.plot(label='  ' + key_str, ax=ax2, **ax2_plt_kwargs)
            ax2.annotate(key_str, xy=(x_center, y_center), xytext=(
                (x_center + x_annot_offset) if x_center < plt_x_center else (x_center - x_annot_offset), y_center),
                         arrowprops=annotate_arrow, **annotate_kwargs)
        else:
            ax2 = tmp_grp.plot(label='  ' + key_str, ax=ax2, **ax2_plt_kwargs)
            x_diff = next_x - last_x
            y_diff = next_y - last_y
            x_diff /= (plt_x_range[1] - plt_x_range[0])
            y_diff /= (plt_y_range[1] - plt_y_range[0])
            k = y_diff / x_diff
            rotn = np.degrees(np.arctan(k)) * .8
            # print 'rotn:', rotn * .9
            ax2.annotate(key_str, xy=(x_center, y_center), xytext=(x_center, y_center), rotation=rotn, **annotate_kwargs)

    # grp_df.plot(x=x_col_name, legend=False)
    x_label = label_dict[x_col_name] if x_col_name in label_dict else x_col_name.replace('_', ' ')
    plt.xlabel(x_label)
    y_label = label_dict[y_col_name] if y_col_name in label_dict else y_col_name.replace('_', ' ')
    plt.ylabel(y_label)

    if not one_subplot:
        ax1.legend(loc='best', prop={'size': 12})
        ax2.legend(loc='best', prop={'size': 12})
        ax1.set_xlim(plt_x_range)
        ax1.set_ylim(plt_y_range)
        ax1.grid(which='major', axis='y')
        plt.title(ds_name)
    else:
        ax2.legend_.remove()
    ax2.set_xlim(plt_x_range)
    ax2.set_ylim(plt_y_range)
    ax2.grid(which='major', axis='y')
    if 'ratio' in x_col_name:
        plt.xticks()
    else:
        plt.xticks()
        ax2.ticklabel_format(style='sci', axis='x', useOffset=True, useMathText=True, scilimits=(0, 0))

    plt.tight_layout()
    plt_fn = out_fn_base + out_fn_ext
    if not one_subplot:
        plt.savefig(plt_fn + '.png', dpi=150)
    else:
        plt_tools.save_n_crop(plt_fn + '.pdf')
        plt_tools.plot_legend(ax2, out_fn_base.rsplit('/', 2)[0] + '/' + out_fn_ext.strip('_') + '_legend.pdf',
                              font_size=12)
    plt.close('all')
    matplotlib.rcParams.update({'font.size': default_font_size})


def preprocess_df(df, net):
    df_cols = set(df.columns)
    dirty = False
    print ' preprocessing '.center(120, '=')
    if 'sample-size' not in df_cols:
        num_vertices = net.num_vertices()
        df['sample-size'] = df['node-ids'].apply(lambda x: len(x) / num_vertices)
    if 'stat_dist_com' not in df_cols:
        print '[preprocess]: filter com stat-dist'
        df['stat_dist_com'] = df[['node-ids', 'stat_dist']].apply(
            lambda (node_ids, stat_dist): list(stat_dist[node_ids]), axis=1).apply(np.array)
        dirty = True
    if 'stat_dist_com_sum' not in df_cols:
        print '[preprocess]: sum com stat-dist'
        df['stat_dist_com_sum'] = df['stat_dist_com'].apply(np.sum)
        dirty = True
    if 'com_in_neighbours' not in df_cols:
        print '[preprocess]: find com in neighbours'
        print '[preprocess]: \tcreate mapping vertex->in-neighbours'
        in_neighbs = {int(v): set(map(int, v.in_neighbours())) for v in net.vertices()}
        print '[preprocess]: \tcreate union of in-neigbs'
        df['com_in_neighbours'] = df['node-ids'].apply(
            lambda x: set.union(*map(lambda v_id: in_neighbs[v_id], x)))
        print '[preprocess]: \tfilter out com nodes'
        df['com_in_neighbours'] = df[['com_in_neighbours', 'node-ids']].apply(
            lambda (com_in_neighbours, com_nodes): com_in_neighbours - set(com_nodes), axis=1)
        dirty = True
    out_neighbs = None
    if 'com_out_neighbours' not in df_cols:
        print '[preprocess]: find com out neighbours'
        print '[preprocess]: \tcreate mapping vertex->out-neighbours'
        out_neighbs = {int(v): set(map(int, v.out_neighbours())) for v in net.vertices()}
        print '[preprocess]: \tcreate union of out-neigbs'
        df['com_out_neighbours'] = df['node-ids'].apply(
            lambda x: set.union(*map(lambda v_id: out_neighbs[v_id], x)))
        print '[preprocess]: \tfilter out com nodes'
        df['com_out_neighbours'] = df[['com_out_neighbours', 'node-ids']].apply(
            lambda (com_out_neighbours, com_nodes): com_out_neighbours - set(com_nodes), axis=1)
        dirty = True
    if 'intra_com_links' not in df_cols:
        print '[preprocess]: count intra com links'
        if out_neighbs is None:
            print '[preprocess]: \tcreate mapping vertex->out-neighbours'
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
    links_range = [1, 5, 10, 20, 100]
    force_recalc = False

    col_label = 'add_rnd_links_fair'
    if col_label not in df_cols or force_recalc:
        print datetime.datetime.now().replace(microsecond=0), 'calc stat dist with fair inserted random links'
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df[col_label] = df[['sample-size', 'node-ids']].apply(add_links_and_calc, axis=1, args=(net, 'rnd', 'fair',))
        dirty = True
        print ''
        print datetime.datetime.now().replace(microsecond=0), '[OK]'

    force_recalc = False
    col_label = 'add_top_links_fair'
    if col_label not in df_cols or force_recalc:
        print datetime.datetime.now().replace(microsecond=0), 'calc stat dist with fair inserted top links'
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df[col_label] = df[['sample-size', 'node-ids']].apply(add_links_and_calc, axis=1, args=(net, 'top', 'fair', orig_stat_dist))
        dirty = True
        print ''
        print datetime.datetime.now().replace(microsecond=0), '[OK]'

    force_recalc = False
    col_label = 'add_top_block_links_fair'
    if col_label not in df_cols or force_recalc:
        print datetime.datetime.now().replace(microsecond=0), 'calc stat dist with fair inserted top block links'
        if orig_stat_dist is None:
            _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                smooth_bias=False,
                                                                                calc_entropy_rate=False, verbose=False)
        df[col_label] = df[['sample-size', 'node-ids']].apply(add_links_and_calc, axis=1, args=(net, 'top_block', 'fair', orig_stat_dist))
        dirty = True
        print ''
        print datetime.datetime.now().replace(microsecond=0), '[OK]'

    if not dirty:
        print ' preprocessing nothing to do '.center(120, '=')
    else:
        print ' preprocessing done '.center(120, '=')
    return df, dirty


def plot_inserted_links(df, columns, filename):
    filename = filename.replace('.gt', '')
    print 'plot inserted links:', filename
    filt_df = df[df['sample-size'] < 0.21]
    used_columns = ['orig_stat_dist_sum', 'stat_dist_com_sum'] + columns
    grp_df = filt_df.groupby('sample-size')[used_columns]
    used_columns = [i.replace('add_', '').replace('_links_', ' ') for i in used_columns]
    grp_df.columns = ['unbiased', 'biased'] + used_columns[2:]

    grp_mean = grp_df.mean()
    grp_std = grp_df.std()
    grp_mean.to_excel(filename + '_inserted_links.xls')
    fig, ax = plt.subplots()

    for i, label in zip(grp_mean.columns, grp_df.columns):
        # ax.errorbar(x=grp_mean.index, y=grp_mean[i], yerr=grp_std[i], label=i.replace('add_','').replace('_links_',''), lw=2)
        links = label.split()
        if len(links) > 1:
            try:
                links = int(links[-1])
            except ValueError:
                links = 5
        else:
            # lw of other lines
            links = 5

        if links == 1:
            linew = 1
        elif links == 5:
            linew = 1.5
        elif links == 10:
            linew = 2
        elif links == 20:
            linew = 2.5
        elif links == 100:
            linew = 3
        else:
            linew = links
        if 'top' in label:
            c = 'red'
            marker = '*'
        elif 'rnd' in label:
            c = 'green'
            marker = '^'
        elif label == 'biased':
            c = 'blue'
            marker = 'D'
        elif label == 'unbiased':
            c = 'lightblue'
            marker = 'd'
        ax.plot(np.array(grp_mean.index), np.array(grp_mean[i]), label=label, marker=marker, lw=linew, color=c,
                alpha=0.8)
    plt.xlabel('sample-size')
    plt.ylabel(r'$\frac{\sum_i \pi\delta}{\sum_i\delta}$')
    plt.xlim([0, 0.2])
    plt.tight_layout()
    out_fn = filename + '_inserted_links.pdf'
    plt_tools.save_n_crop(out_fn)
    legend_fname = filename.rsplit('/', 1)[0] + '/inserted_links_legend.pdf'
    plt_tools.plot_legend(ax, legend_fname, font_size=12, ncols=4)


def main():
    base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
    out_dir = base_dir + 'plots/'
    create_folder_structure(out_dir)

    result_files = find_files(base_dir, '.df')
    print result_files
    cors = list()
    all_dfs = list()
    net_name = ''
    net = None
    skipped_ds = set()
    # skipped_ds.add('daserste')
    for i in sorted(filter(lambda x: 'preprocessed' not in x, result_files), reverse=True):
        current_net_name = i.rsplit('_bs', 1)[0]
        bias_strength = int(i.split('_bs')[-1].split('.')[0])
        if bias_strength > 2:
            continue
        elif any((i in current_net_name for i in skipped_ds)):
            print 'skip ds:', current_net_name
            continue
        if current_net_name != net_name:
            print 'load network:', current_net_name.rsplit('/', 1)[-1]
            net = load_graph(current_net_name)
            net_name = current_net_name
        assert net is not None
        preprocessed_filename = i.rsplit('.df', 1)[0] + '_preprocessed.df'
        if os.path.isfile(preprocessed_filename): # and time.ctime(os.path.getmtime(preprocessed_filename)) > time.ctime(os.path.getmtime(i)):
            print 'read preprocessed file:', preprocessed_filename.rsplit('/', 1)[-1]
            try:
                df = pd.read_pickle(preprocessed_filename)
            except:
                print traceback.format_exc()
                print 'fallback: read orig file:', i.rsplit('/', 1)[-1]
                df = pd.read_pickle(i)
        else:
            print 'read:', i.rsplit('/', 1)[-1]
            df = pd.read_pickle(i)
        df, is_df_dirty = preprocess_df(df, net)
        if is_df_dirty:
            print 'store preprocessed df'
            while True:
                # in case of str+c
                try:
                    cache_fname = preprocessed_filename + '.cache'
                    df.to_pickle(cache_fname)
                    shutil.move(cache_fname, preprocessed_filename)
                    break
                except:
                    print traceback.format_exc()
                    print 'Warning currently writing file: retry'
        print 'plot:', i.rsplit('/', 1)[-1]
        print 'all cols:', df.columns
        print '=' * 80

        out_fn = out_dir + i.rsplit('/', 1)[-1][:-3]
        insert_links_labels = sorted(
            filter(lambda x: x.startswith(('add_top_links_fair', 'add_rnd_links_fair', 'add_top_block_links_fair')),
                   df.columns))
        plot_inserted_links(df, insert_links_labels, out_fn)

        out_fn += '.png'
        cors.append(plot_dataframe(df, net, bias_strength, out_fn))
        df['bias_strength'] = bias_strength
        # exit()
        #all_dfs.append(df.copy())
    cors = np.array(cors)
    print 'average corr:', cors.mean()
    #all_dfs = pd.concat(all_dfs)
    #plot_df_fac(all_dfs, out_dir + '/all_dfs.png')


if __name__ == '__main__':
    start = datetime.datetime.now()
    print 'START:', start
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

