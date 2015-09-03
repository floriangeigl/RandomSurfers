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

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
matplotlib.rcParams.update({'font.size': 20})


def get_stat_dist_bias_sum(df, col_name, cat_name, category_col_name):
    filt_df = df[[col_name, category_col_name]]
    assert np.isclose(df[col_name].sum(), 1)
    cat_filter = filt_df[category_col_name] == cat_name
    bias_sum = filt_df[col_name][cat_filter].sum()
    return bias_sum, cat_filter.sum()


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)
bias_range = None
# bias_range = [1, 10]

stat_dist_files = find_files(base_dir, 'stat_dists.df')
print stat_dist_files
network_files = {i.rsplit('/', 1)[-1][:-3]: i for i in find_files(base_dir, '.gt')}
print network_files

for stat_dist_fn in stat_dist_files:
    ds_name = stat_dist_fn.rsplit('/', 1)[-1].split('_stat_dist')[0]
    print ds_name
    net_file = network_files[ds_name]
    net = load_graph(net_file)
    print net
    res_df = pd.DataFrame(index=[1.])
    feas_df = pd.DataFrame(index=[1.])
    cat_size = pd.DataFrame()
    df = pd.read_pickle(stat_dist_fn)
    bias_base_names = set(map(lambda x: x.split('_cs')[0], filter(lambda x: '_bs' and '_cs' in x, df.columns)))
    print bias_base_names
    for bias_name in bias_base_names:
        bias_columns = filter(lambda x: x.startswith(bias_name), df.columns)
        bias_label = bias_name + ' (' + str(
            "%.2f" % (float(bias_columns[0].split('_cs')[-1].split('_bs')[0]) / len(df) * 100)) + '%)'
        # print bias_columns
        # unbiased
        res_df.at[1., bias_label], cat_size.at[1, bias_label] = get_stat_dist_bias_sum(df, 'adjacency', bias_name,
                                                                                    'category')
        orig_weight = net.new_edge_property('float', 1.)
        biased_nodes = net.new_vertex_property('bool')
        biased_nodes.a = np.array(df['category'] == bias_name)
        feas_df.at[1., bias_label] = 1.

        fast_calc = False
        feas_fact = None

        '''
        def worker_func(net, biased_nodes, bs):
            current_feas_dist = list()
            for v in net.vertices():
                node_feas = np.array([bs if biased_nodes[e.target()] else 1. for e in v.out_edges()])
                node_feas = node_feas.max() / node_feas.min()
                current_feas_dist.append(node_feas)
            current_feas_dist = np.array(current_feas_dist)
            current_feas_fac = current_feas_dist.max()
            # print bs, 'done'
            return bs, current_feas_fac
        '''

        results = list()
        worker_pool = mp.Pool(processes=10)
        for idx_bc, bc in enumerate(bias_columns):
            bs = float(bc.split('_bs')[-1])
            if bs > 1. and (bias_range is None or bias_range[0] <= bs <= bias_range[1]):
                # worker_pool.apply_async(worker_func, args=(net, biased_nodes, bs), callback=results.append)
                res_df.at[bs, bias_label], _ = get_stat_dist_bias_sum(df, bc, bias_name, 'category')
        worker_pool.close()
        worker_pool.join()
        if results:
            for bs, feas_fac in results:
                feas_df.at[bs, bias_label] = feas_fac

    res_df.sort(inplace=True)
    feas_df.sort(inplace=True)
    # print res_df
    res_df_i = res_df.reindex(range(int(res_df.index.min()), int(res_df.index.max()) + 1))
    res_df_i = res_df_i.interpolate(method='linear')
    # print res_df
    limits = [.25, .5, .75]
    limits_df = pd.DataFrame(index=limits, columns=res_df_i.columns)
    for l in limits:
        tmp = list()
        for col in res_df_i.columns:
            try:
                tmp.append(res_df_i[col][res_df_i[col] < l].last_valid_index() + 1)
            except TypeError:
                tmp.append(np.nan)
        limits_df.loc[l] = tmp
    print limits_df
    limits_df.columns = map(lambda x: x.rsplit(' (', 1)[0], limits_df.columns)
    limits_df.to_pickle(base_dir + 'data/' + ds_name + '_limits.df')

    # res_df /= res_df.max()
    out_fn = out_dir + ds_name + '_bias_influence.pdf'
    if len(res_df.columns) > 5:
        ax = res_df.plot(legend=False, logx=True, logy=True)
        plt.xlabel('bias strength')
        plt.ylabel('sum of stationary values')
        plt.tight_layout()
        plt.savefig(out_fn)
    else:
        ax = res_df.plot(lw=2, style=['-*', '-o', '-D'])
        plt.xlabel('bias strength')
        plt.ylabel('sum of stationary values')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.75])
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.55))
        # plt.tight_layout()
        plt.savefig(out_fn, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')

    tmp_df = res_df / res_df.min()
    ax = tmp_df.plot(lw=3, style='-*')
    plt.xlabel('bias strength')
    plt.ylabel(r'$\frac{biased\ stat.\ values\ sum}{unbiased\ stat.\ values\ sum}$')
    plt.ylim([1., tmp_df.max().max()])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.75])
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.55))
    # plt.tight_layout()
    plt.savefig(out_dir + ds_name + '_bias_influence_norm.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')

    print cat_size.sum()
    print res_df.sum()
    tmp_df = res_df / (res_df.min() / cat_size.sum())
    ax = tmp_df.plot(lw=3, style='-*')
    plt.xlabel('bias strength')
    plt.ylabel(r'$\frac{biased\ stat.\ values\ sum}{unbiased\ stat.\ values\ sum}$')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.75])
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.55))
    plt.savefig(out_dir + ds_name + '_bias_influence_nodes_norm.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')

    ax = feas_df.plot(lw=2, style=['-*', '-o', '-D'])
    plt.xlabel('bias strength')
    plt.ylabel('feasibility')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.025, box.width, box.height * 0.75])
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.55))
    # plt.tight_layout()
    plt.savefig(out_dir + ds_name + '_bias_feasability.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')
