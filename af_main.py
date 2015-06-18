from __future__ import division
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
import multiprocessing
import datetime
import traceback
from self_sim_entropy import self_sim_entropy
import os
from data_io import *
import utils
from graph_tool.all import *
import pandas as pd
from network_matrix_tools import stationary_dist, calc_entropy_and_stat_dist
from scipy.sparse import csr_matrix, dia_matrix

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 300)

def generate_weighted_matrix(net, eweights):
    if net.get_vertex_filter()[0] is not None:
        v_map = {v: i for i, v in enumerate(net.vertices())}
        col_idx = [v_map[e.source()] for e in net.edges()]
        row_idx = [v_map[e.target()] for e in net.edges()]
    else:
        col_idx = [int(e.source()) for e in net.edges()]
        row_idx = [int(e.target()) for e in net.edges()]
    data = [eweights[e] for e in net.edges()]
    shape = net.num_vertices()
    shape = (shape, shape)
    return csr_matrix((data, (row_idx, col_idx)), shape=shape)

def filter_and_calc(net, eweights=None, vfilt=None, efilt=None, merge_type='+'):
    orig_nodes = net.num_vertices()
    if vfilt is not None:
        net.set_vertex_filter(vfilt)
    if efilt is not None:
        net.set_edge_filter(efilt)
    print 'filtered network vertices:', net.num_vertices(), '(', net.num_vertices() / orig_nodes * 100, '%)'
    mapping = defaultdict(int, {i: j for i, j in enumerate(net.vertices())})
    a = adjacency(net)
    if eweights is not None:
        w_mat = generate_weighted_matrix(net, eweights)
        if merge_type == '+':
            a += w_mat
        elif merge_type == '*':
            a = a.multiply(w_mat)
    entropy_rate, stat_dist = calc_entropy_and_stat_dist(a)
    stat_dist = defaultdict(int, {mapping[i]: j for i, j in enumerate(stat_dist)})
    net.clear_filters()
    return entropy_rate, np.array([stat_dist[v] for v in net.vertices()])

def main():
    base_outdir = 'output/iknow/'
    basics.create_folder_structure(base_outdir)
    stat_dist = pd.DataFrame()
    entropy_rate = pd.DataFrame()
    post_fix = ''
    print 'load network'
    net = load_graph('/home/fgeigl/navigability_of_networks/preprocessing/data/af.gt')
    remove_self_loops(net)
    if False:
        net.set_vertex_filter(net.vp['strong_lcc'])
        net.purge_vertices()
        net.clear_filters()
    print net

    assert net.get_vertex_filter()[0] is None
    a = adjacency(net)
    print 'calc stat dist adj matrix'
    entropy_rate['adj'], stat_dist['adj'] = calc_entropy_and_stat_dist(a)
    print 'calc stat dist weighted click subgraph'
    click_pmap = net.new_edge_property('float')
    clicked_nodes = net.new_vertex_property('bool')
    tele_map = net.ep['click_teleportations']
    loops_map = net.ep['click_loops']
    trans_map = net.ep['click_transitions']
    for e in net.edges():
        e_trans = trans_map[e]
        if e_trans and not loops_map[e] and not tele_map[e]:
            s, t = e.source(), e.target()
            clicked_nodes[s] = True
            clicked_nodes[t] = True
            click_pmap[e] = e_trans
    entropy_rate['click_sub'], stat_dist['click_sub'] = filter_and_calc(net, eweights=click_pmap, vfilt=clicked_nodes)
    page_c_pmap = net.vp['view_counts']
    entropy_rate['page_counts'], stat_dist['page_counts'] = np.nan, page_c_pmap.a / page_c_pmap.a.sum()

    urls_pmap = net.vp['url']
    stat_dist['url'] = [urls_pmap[v] for v in net.vertices()]
    print stat_dist
    print stat_dist[['adj', 'click_sub', 'page_counts']].sum()
    print 'adj top10'
    print stat_dist.sort('adj', ascending=False).head()
    print 'click top10'
    print stat_dist.sort('click_sub', ascending=False).head()
    print 'page views top10'
    print stat_dist.sort('page_counts', ascending=False).head()
    print 'adj last10'
    print stat_dist.sort('adj', ascending=True).head()
    print 'click last10'
    print stat_dist.sort('click_sub', ascending=True).head()
    print 'page views last10'
    print stat_dist.sort('page_counts', ascending=True).head()

    clicked_stat_dist = stat_dist[stat_dist['click_sub'] > 0]
    print 'clicked pages'.center(80, '-')
    clicked_stat_dist[['adj', 'click_sub', 'page_counts']].sum()
    stat_dist.to_pickle(base_outdir + 'stationary_dist.df')
    print 'entropy rates'.center(80, '-')
    entropy_rate.to_pickle(base_outdir + 'entropy_rate.df')

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start