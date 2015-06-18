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
from network_matrix_tools import stationary_dist
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
    stat_dist = stationary_dist(a)
    stat_dist = defaultdict(int, {mapping[i]: j for i, j in enumerate(stat_dist)})
    net.clear_filters()
    return np.array([stat_dist[v] for v in net.vertices()])

def main():
    base_outdir = 'output/iknow/'
    basics.create_folder_structure(base_outdir)
    stat_dist = pd.DataFrame()
    post_fix = ''
    print 'load network'
    net = load_graph('/home/fgeigl/navigability_of_networks/preprocessing/data/af.gt')
    net.set_vertex_filter(net.vp['strong_lcc'])
    net.purge_vertices()
    net.clear_filters()
    print net

    assert net.get_vertex_filter()[0] is None
    a = adjacency(net)
    print 'calc stat dist adj matrix'
    stat_dist['adj'] = stationary_dist(a)
    print 'calc stat dist weighted click subgraph'
    click_pmap = net.new_edge_property('float')
    clicked_nodes = net.new_vertex_property('bool')
    tele_map = net.ep['click_teleportations']
    loops_map = net.ep['click_loops']
    trans_map = net.ep['click_transitions']
    for e in net.edges():
        e_trans = trans_map[e]
        if e_trans and not loops_map[e] and not tele_map[e]:
            s, t = e.source(),e.target()
            clicked_nodes[s] = True
            clicked_nodes[t] = True
            click_pmap[e] = e_trans
    stat_dist['click_sub'] = filter_and_calc(net, eweights=click_pmap, vfilt=clicked_nodes)
    page_c_pmap = net.vp['view_counts']
    stat_dist['page_counts'] = page_c_pmap.a / page_c_pmap.a.sum()

    urls_pmap = net.vp['url']
    stat_dist['url'] = [urls_pmap[v] for v in net.vertices()]
    print stat_dist
    print stat_dist[['adj', 'click_sub', 'page_counts']].sum()
    print stat_dist.sort('adj', ascending=False).head()
    print stat_dist.sort('click_sub', ascending=False).head()
    print stat_dist.sort('page_counts', ascending=False).head()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start