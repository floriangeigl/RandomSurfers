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

def main():
    base_outdir = 'output/iknow/'
    basics.create_folder_structure(base_outdir)
    stat_dist = pd.DataFrame()
    post_fix = ''
    print 'load network'
    af_net = load_graph('/home/fgeigl/navigability_of_networks/preprocessing/data/af.gt' + post_fix)
    print af_net
    a = adjacency(af_net)
    print 'calc stat dist adj matrix'
    stat_dist['adj'] = stationary_dist(a)
    print 'load click matrix'
    click_mat = try_load('/home/fgeigl/navigability_of_networks/preprocessing/data/af_click_matrix' + post_fix)
    assert af_net.num_vertices() == click_mat.shape[0] == click_mat.shape[1]
    # views = try_load('/home/fgeigl/navigability_of_networks/preprocessing/data/view_counts')
    # assert len(views) == af_net.num_vertices()
    # stat_dist['views'] = views / views.sum()
    print 'find clicked nodes'
    clicked_nodes = map(set, click_mat.nonzero())
    clicked_nodes = clicked_nodes[0] | clicked_nodes[1]
    print 'percentage clicked nodes:', len(clicked_nodes) / click_mat.shape[0]
    diag_data = np.array([1 if i in clicked_nodes else 0 for i in xrange(click_mat.shape[0])])
    stat_dist['clicked_nodes'] = diag_data
    print 'create sub click mat'
    diag = dia_matrix((diag_data, 0), shape=click_mat.shape)
    af_click_sub = diag * a * diag
    print 'calc stat dist of clicked sub'
    stat_dist['clicked_sub'] = stationary_dist(click_mat + af_click_sub)
    clicked_nodes_sd = stat_dist[stat_dist['clicked_nodes'] == 1]
    print clicked_nodes_sd
    print clicked_nodes_sd.sum()



if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start