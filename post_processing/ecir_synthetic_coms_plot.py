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
import time
import network_matrix_tools
import operator
import random

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


def add_links_and_calc(com_nodes, net=None, method='rnd', num_links=1, top_measure=None):
    new_edges = set()
    orig_num_edges = net.num_edges()
    orig_num_com_nodes = len(com_nodes)
    if orig_num_com_nodes >= net.num_vertices():
        return None
    other_nodes = set(range(0, net.num_vertices())) - set(com_nodes)
    if method == 'rnd':
        for e_count in xrange(num_links):
            while True:
                src = random.sample(other_nodes, 1)[0]
                dest = random.sample(com_nodes, 1)[0]
                if net.edge(src, dest) is not None and (src, dest) not in new_edges:
                    new_edges.add((src, dest))
                    break
    elif method == 'top':
        if top_measure is None:
            nodes_measure = np.array(net.degree_property_map('in').a)
        else:
            nodes_measure = top_measure

        try:
            sorted_other_nodes = sorted(other_nodes, key=lambda x: nodes_measure[x])
            sorted_com_nodes = sorted(com_nodes, key=lambda x: nodes_measure[x])
            for dest in sorted_com_nodes:
                for src in sorted_other_nodes:
                    if net.edge(src, dest) is None:
                        new_edges.add((src, dest))
                        if len(new_edges) >= num_links:
                            raise GetOutOfLoops
            print 'could not insert all links:', len(new_edges), 'of', num_links
        except GetOutOfLoops:
            pass

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
    return relinked_stat_dist_sum


def plot_df(df, net, bias_strength, filename):
    label_dict = dict()
    label_dict['ratio_com_out_deg_in_deg'] = r'$\frac{k_{com}^+}{k_{com}^-}$'
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
    # plot_df = df[df['sample-size'] < 0.31].copy()
    plot_df = df

    in_deg = np.array(net.degree_property_map('in').a)
    plot_df['in_neighbours_in_deg'] = plot_df['com_in_neighbours'].apply(lambda x: in_deg[list(x)].sum())

    out_deg = np.array(net.degree_property_map('out').a)
    plot_df['out_neighbours_out_deg'] = plot_df['com_out_neighbours'].apply(lambda x: out_deg[list(x)].sum())

    plot_df['ratio_out_out_deg_in_in_deg'] = plot_df['out_neighbours_out_deg'] / plot_df['in_neighbours_in_deg']

    plot_df['com_in_deg'] = plot_df['node-ids'].apply(lambda x: in_deg[list(x)].sum()) - plot_df['intra_com_links']
    plot_df['com_out_deg'] = plot_df['node-ids'].apply(lambda x: out_deg[list(x)].sum()) - plot_df['intra_com_links']
    plot_df['ratio_com_out_deg_in_deg'] = plot_df['com_out_deg'] / plot_df['com_in_deg']

    plot_df['stat_dist_sum_fac'] = plot_df['stat_dist_com_sum'] / plot_df['orig_stat_dist_sum']
    orig_columns.add('stat_dist_sum_fac')

    ds_name = filename.rsplit('/', 1)[-1].rsplit('.')[0]
    for col_name in set(plot_df.columns) - orig_columns:
        current_filename = filename[:-4] + '_' + col_name.replace(' ', '_')
        sub_folder = current_filename.rsplit('/', 1)[-1].split('.gt', 1)[0]
        current_filename = current_filename.rsplit('/', 1)
        current_filename = current_filename[0] + '/' + sub_folder + '/' + current_filename[1]
        create_folder_structure(current_filename)
        print 'plot:', col_name
        for normed_stat_dist in [True, False]:
            y = plot_df[col_name]
            x = plot_df['sample-size']
            c = plot_df['stat_dist_normed'] if normed_stat_dist else plot_df['stat_dist_com_sum']
            fix, ax = plt.subplots()
            ac = ax.scatter(x, y, c=c, lw=0, alpha=0.7, cmap='coolwarm')
            ax.set_xticks(sorted(set(x)), minor=True)
            cbar = plt.colorbar(ac)
            plt.xlabel('sample size')
            plt.xlim([0, plot_df['sample-size'].max() + 0.01])
            y_range_one_perc = (plot_df[col_name].max() - plot_df[col_name].min()) * 0.01
            plt.ylim([plot_df[col_name].min() - y_range_one_perc, plot_df[col_name].max() + y_range_one_perc])
            plt.ylabel(col_name.replace('_', ' '))
            cbar.set_label('sample size standardized $\\sum \\pi$' if normed_stat_dist else '$\\sum \\pi$')
            # cbar.set_label('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')

            plt.title(ds_name + '\nBias Strength: ' + str(int(bias_strength)))
            plt.tight_layout()
            out_f = (current_filename + '_normed.png') if normed_stat_dist else (current_filename + '.png')

            plt.grid(which='minor', axis='x')
            plt.savefig(out_f, dpi=150)
            plt.close('all')

        plot_df.sort(col_name, inplace=True)
        print '\tsum'
        # sum
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        ax1.plot(None, label='sample-size', c='white')
        ax2.plot(None, label='sample-size', c='white')
        for key, grp in plot_df.groupby('sample-size'):
            if key < .21:
                ax1 = grp.plot(x=col_name, y='stat_dist_com_sum', ax=ax1, label='  ' + '%.2f' % key)
                grp['tmp'] = pd.rolling_mean(grp['stat_dist_com_sum'], window=int(.25 * len(grp)), center=True)
                ax2 = grp.plot(x=col_name, y='tmp', ax=ax2, label='  ' + '%.2f' % key)
        # grp_df.plot(x=col_name, legend=False)
        x_label = label_dict[col_name] if col_name in label_dict else col_name.replace('_', ' ')
        plt.xlabel(x_label)
        plt.ylabel(r'$\sum \pi$')
        out_f = current_filename + '_lines.png'
        ax1.legend(loc='best', prop={'size': 12})
        ax2.legend(loc='best', prop={'size': 12})
        ax1.grid(which='major', axis='y')
        ax2.grid(which='major', axis='y')
        ax2.set_xlim([0, 2])
        plt.title(ds_name)
        plt.tight_layout()
        plt.savefig(out_f, dpi=150)
        plt.close('all')

        # fac
        print '\tfrac'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        ax1.plot(None, label='sample-size', c='white')
        ax2.plot(None, label='sample-size', c='white')
        for key, grp in plot_df.groupby('sample-size'):
            if key < .21:
                ax1 = grp.plot(x=col_name, y='stat_dist_sum_fac', ax=ax1, label='  ' + '%.2f' % key)
                grp['tmp'] = pd.rolling_mean(grp['stat_dist_sum_fac'], window=int(.25 * len(grp)), center=True)
                ax2 = grp.plot(x=col_name, y='tmp', ax=ax2, label='  ' + '%.2f' % key)

        # grp_df.plot(x=col_name, legend=False)
        x_label = label_dict[col_name] if col_name in label_dict else col_name.replace('_', ' ')
        plt.xlabel(x_label)
        plt.ylabel(r'$\frac{\sum \pi_b}{\sum \pi_o}$')
        out_f = current_filename + '_lines_fac.png'
        ax1.legend(loc='best', prop={'size': 12})
        ax2.legend(loc='best', prop={'size': 12})
        ax1.grid(which='major', axis='y')
        ax2.grid(which='major', axis='y')
        ax2.set_xlim([0, 2])
        plt.title(ds_name)
        plt.tight_layout()
        plt.savefig(out_f, dpi=150)
        plt.close('all')
    return 0


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
                                                                                calc_entropy_rate=False)
        df['orig_stat_dist_sum'] = df['node-ids'].apply(lambda x: orig_stat_dist[x].sum())
        dirty = True
    links_range = [1, 5, 10, 20, 100]
    for i in links_range:
        col_label = 'add_rnd_links_' + str(i).zfill(3)
        if col_label not in df_cols:
            print 'calc stat dist with', i, ' inserted random links'
            if orig_stat_dist is None:
                _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                    smooth_bias=False,
                                                                                    calc_entropy_rate=False)
            df[col_label] = df['node-ids'].apply(add_links_and_calc, args=(net, 'rnd', i,))
            dirty = True

    for i in links_range:
        col_label = 'add_top_links_' + str(i).zfill(3)
        if col_label not in df_cols:
            print 'calc stat dist with', i, ' inserted top links'
            if orig_stat_dist is None:
                _, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(net), method='EV',
                                                                                    smooth_bias=False,
                                                                                    calc_entropy_rate=False)
            df[col_label] = df['node-ids'].apply(add_links_and_calc, args=(net, 'top', i, orig_stat_dist))
            dirty = True
    if not dirty:
        print ' preprocessing nothing to do '.center(120, '=')
    else:
        print ' preprocessing done '.center(120, '=')
    return df, dirty


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
    for i in sorted(filter(lambda x: 'preprocessed' not in x, result_files), reverse=True):
        current_net_name = i.rsplit('_bs', 1)[0]
        bias_strength = int(i.split('_bs')[-1].split('.')[0])
        if bias_strength > 2:
            continue
        if current_net_name != net_name:
            print 'load network:', current_net_name.rsplit('/', 1)[-1]
            net = load_graph(current_net_name)
            net_name = current_net_name
        assert net is not None
        preprocessed_filename = i.rsplit('.df', 1)[0] + '_preprocessed.df'
        if os.path.isfile(preprocessed_filename) and time.ctime(os.path.getmtime(preprocessed_filename)) < time.ctime(
                os.path.getmtime(i)):
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
                    df.to_pickle(preprocessed_filename)
                    break
                except:
                    print traceback.format_exc()
                    print 'Warning currently writing file: retry'
        print 'plot:', i.rsplit('/', 1)[-1]
        print 'all cols:', df.columns
        print '=' * 80
        insert_links_labels = sorted(filter(lambda x: x.startswith(('add_top_links_', 'add_rnd_links_')), df.columns))
        print 'inserted links cols'
        df.groupby('sample-size').mean()[['orig_stat_dist_sum', 'stat_dist_com_sum'] + insert_links_labels].to_excel(
            'overview.xls')

        out_fn = out_dir + i.rsplit('/', 1)[-1][:-3] + '.png'
        cors.append(plot_df(df, net, bias_strength, out_fn))
        df['bias_strength'] = bias_strength
        exit()
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

