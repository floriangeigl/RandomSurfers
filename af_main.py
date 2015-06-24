from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import tools.basics as basics
import datetime
from data_io import *
import utils
from graph_tool.all import *
import pandas as pd
from network_matrix_tools import stationary_dist, calc_entropy_and_stat_dist, entropy_rate
from scipy.sparse import csr_matrix, dia_matrix
pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 300)


def is_hierarchical_link(source, target):
    return source.rsplit('/', 1)[0] != target.rsplit('/', 1)[0]

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


def filter_and_calc(net, eweights=None, vfilt=None, efilt=None, merge_type='+', stat_dist=None):
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
    if stat_dist is None:
        entropy_r, stat_dist = calc_entropy_and_stat_dist(a)
    else:
        assert isinstance(stat_dist, PropertyMap)
        if not stat_dist.a.sum() == 1:
            stat_dist = np.array([stat_dist[v] for v in net.vertices()])
            stat_dist /= stat_dist.sum()
        entropy_r = entropy_rate(a, stat_dist=stat_dist)
    stat_dist = defaultdict(int, {mapping[i]: j for i, j in enumerate(stat_dist)})
    net.clear_filters()
    return entropy_r, np.array([stat_dist[v] for v in net.vertices()])

def main():
    base_outdir = 'output/iknow/'
    basics.create_folder_structure(base_outdir)
    stat_dist = pd.DataFrame()
    entropy_rate_df = pd.DataFrame()
    post_fix = ''
    print 'load network'
    net = load_graph('/home/fgeigl/navigability_of_networks/preprocessing/data/af.gt')
    remove_self_loops(net)
    if False:
        net.set_vertex_filter(net.vp['strong_lcc'])
        net.purge_vertices()
        net.purge_edges()
        net.clear_filters()
    print net

    assert net.get_vertex_filter()[0] is None
    a = adjacency(net)
    print 'calc stat dist adj matrix'
    entropy_rate_df.at[1, 'adj'], stat_dist['adj'] = calc_entropy_and_stat_dist(a)
    print 'calc stat dist weighted click subgraph'
    # remove_parallel_edges(net)
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

    # sublinear tf scaling
    tmp_clicks = np.array(click_pmap.a)
    mask = tmp_clicks > 0
    tmp_clicks[mask] = 1 + np.log(tmp_clicks[mask])

    click_pmap.a = tmp_clicks
    click_map_ser = pd.Series(click_pmap.a)
    click_map_ser.plot(kind='hist', bins=10,logy=True)
    plt.ylabel('number of edges')
    plt.xlabel('clicks')
    plt.savefig('link_clicks_histo.png')
    plt.close('all')
    click_map_ser[click_map_ser > 0].plot(kind='hist', bins=10,logy=True)
    plt.ylabel('number of edges')
    plt.xlabel('clicks')
    plt.savefig('link_clicks_histo_filtzero.png')
    plt.close('all')
    print 'max click:', click_pmap.a.max()
    print 'min click:', click_pmap.a.min()
    entropy_rate_df.at[1, 'click_sub'], stat_dist['click_sub'] = filter_and_calc(net, eweights=click_pmap,
                                                                              vfilt=clicked_nodes)
    page_c_pmap = net.vp['view_counts']
    page_c_stat_dist = page_c_pmap.a / page_c_pmap.a.sum()
    stat_dist['page_counts'] = page_c_stat_dist
    lateral_nodes = net.new_vertex_property('bool')
    lateral_nodes.a = np.array(page_c_pmap.a) > 0

    entropy_rate_df.at[1, 'page_counts'] = filter_and_calc(net, eweights=click_pmap, vfilt=lateral_nodes,
                                                           stat_dist=page_c_pmap)

    urls_pmap = net.vp['url']
    stat_dist['url'] = [urls_pmap[v] for v in net.vertices()]
    print stat_dist.head()
    print stat_dist.tail()
    print stat_dist[['adj', 'click_sub', 'page_counts']].sum()
    print 'adj top'
    print stat_dist.sort('adj', ascending=False).head()
    print 'click top'
    print stat_dist.sort('click_sub', ascending=False).head()
    print 'page views top'
    print stat_dist.sort('page_counts', ascending=False).head()
    print 'adj last'
    print stat_dist.sort('adj', ascending=True).head()
    print 'click last'
    print stat_dist.sort('click_sub', ascending=True).head()
    print 'page views last'
    print stat_dist.sort('page_counts', ascending=True).head()

    clicked_stat_dist = stat_dist[stat_dist['click_sub'] > 0]
    print 'clicked pages'.center(80, '-')
    print clicked_stat_dist[['adj', 'click_sub', 'page_counts']].sum()
    stat_dist.to_pickle(base_outdir + 'stationary_dist.df')
    print 'entropy rates'.center(80, '-')
    print entropy_rate_df
    entropy_rate_df.to_pickle(base_outdir + 'entropy_rate.df')
    gini_df = pd.DataFrame()
    for i in clicked_stat_dist.columns:
        if i is not 'url':
            gini_df.at[1, i] = utils.gini_coeff(clicked_stat_dist[clicked_stat_dist[i] > 0][i])
    print 'gini'.center(80,'-')
    print gini_df
    gini_df.to_pickle(base_outdir + 'gini.df')
    if False:
        print 'analyze link categories'
        assert net.get_vertex_filter()[0] is None
        url_pmap = net.vp['url']
        edge_cat = net.new_edge_property('bool')
        clicked_edge_cat = net.new_edge_property('int')
        e_idx = net.edge_index
        valid_e_idx = [e_idx[e] for e in net.edges()]
        edge_cat.a[valid_e_idx] = [is_hierarchical_link(url_pmap[e.source()], url_pmap[e.target()]) for e in net.edges()]
        net_hier_links = edge_cat.a.sum() / net.num_edges()
        print 'network hierarchical links:', net_hier_links * 100, '%'
        clicked_edge_cat.a = np.array(edge_cat.a) * np.array(click_pmap.a)
        click_hier_links = clicked_edge_cat.a.sum() / (np.array(click_pmap.a) > 0).sum()
        print 'clicked hierarchical links:', click_hier_links * 100, '%'

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start