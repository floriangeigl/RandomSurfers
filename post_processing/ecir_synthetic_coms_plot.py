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


def plot_df(df, net, bias_strength, filename):
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

    in_deg = np.array(net.degree_property_map('in').a)
    df['in_neighbours_in_deg'] = df['com_in_neighbours'].apply(lambda x: in_deg[list(x)].sum())

    out_deg = np.array(net.degree_property_map('out').a)
    df['out_neighbours_out_deg'] = df['com_out_neighbours'].apply(lambda x: out_deg[list(x)].sum())

    df['ratio_out_out_deg_in_in_deg'] = df['out_neighbours_out_deg'] / df['in_neighbours_in_deg']

    df['com_in_deg'] = df['node-ids'].apply(lambda x: in_deg[list(x)].sum())
    df['com_out_deg'] = df['node-ids'].apply(lambda x: out_deg[list(x)].sum())
    df['ratio_com_out_deg_in_deg'] = df['com_out_deg'] / df['com_in_deg']

    for col_name in set(df.columns) - orig_columns:
        current_filename = filename[:-4] + '_' + col_name.replace(' ', '_')
        print 'plot:', col_name
        for normed_stat_dist in [True, False]:
            plot_df = df[df['sample-size'] < 0.3]
            y = plot_df[col_name]
            x = plot_df['sample-size']
            c = plot_df['stat_dist_normed'] if normed_stat_dist else plot_df['stat_dist_com_sum']
            plt.scatter(x, y, c=c, lw=0, alpha=0.7, cmap='coolwarm')
            cbar = plt.colorbar()
            plt.xlabel('sample size')
            plt.xlim([0, plot_df['sample-size'].max() + 0.05])
            plt.ylim([plot_df[col_name].min(), plot_df[col_name].max()])
            plt.ylabel(col_name.replace('_', ' '))
            cbar.set_label('$\\sum \\pi$')
            # cbar.set_label('$\\frac{\\sum \\pi_b}{\\sum \\pi_{ub}}$')

            plt.title(filename.rsplit('/', 1)[-1].rsplit('.')[0] + '\nBias Strength: ' + str(int(bias_strength)))
            plt.tight_layout()
            out_f = (current_filename + '_normed.png') if normed_stat_dist else (current_filename + '.png')
            plt.savefig(out_f, dpi=150)
            plt.close('all')
    return 0


def preprocess_df(df, net):
    df_cols = set(df.columns)
    dirty = False
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
    for i in sorted(result_files):
        current_net_name = i.rsplit('_bs', 1)[0]
        if current_net_name != net_name:
            print 'load network:', current_net_name.rsplit('/', 1)[-1]
            net = load_graph(current_net_name)
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
        print df.columns
        bias_strength = int(i.split('_bs')[-1].split('.')[0])

        out_fn = out_dir + i.rsplit('/', 1)[-1][:-3] + '.png'
        cors.append(plot_df(df, net, bias_strength, out_fn))
        df['bias_strength'] = bias_strength
        #all_dfs.append(df.copy())
    cors = np.array(cors)
    print 'average corr:', cors.mean()
    #all_dfs = pd.concat(all_dfs)
    #plot_df_fac(all_dfs, out_dir + '/all_dfs.png')


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

