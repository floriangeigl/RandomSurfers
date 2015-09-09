# -*- coding: utf-8 -*-
from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from tools.gt_tools import SBMGenerator, load_edge_list
from graph_tool.all import *
import tools.basics as basics
import multiprocessing
import datetime
from self_sim_entropy import self_sim_entropy
import network_matrix_tools
from data_io import *
import utils
import Queue
import os
import operator
from preprocessing.categorize_network_nodes import get_cat_dist
import copy, time
import pandas as pd
from post_processing.ecir_synthetic_coms_plot import plot_df


def get_stat_dist_sum(net, ds_name, bias_strength, com_sizes, num_samples, out_dir, method='EV'):
    all_vertices = set(net.vertices())
    net_indegree = net.degree_property_map('in')
    adjacency_matrix = adjacency(net)
    results = list()
    print_prefix = ds_name
    #_, orig_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, method=method,
    #                                                                 print_prefix=print_prefix + ' [unbiased baseline] ',
    #                                                                 smooth_bias=False,
    #                                                                 calc_entropy_rate=False)
    for com_s in com_sizes:
        nodes_per_com = min(int(np.round(com_s * net.num_vertices())), net.num_vertices())
        coms = [sorted(random.sample(all_vertices, nodes_per_com)) for i in range(num_samples)]
        print 'com-size:', com_s, ' #coms:', len(coms)

        for idx, c in enumerate(coms):
            bias_vec = np.ones(net.num_vertices())
            c_idx = map(int, c)
            bias_vec[c_idx] = bias_strength
            _, stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, bias_vec, method=method,
                                                                           print_prefix=print_prefix + ' [%.2f' % com_s + '| %3.0f' % (
                                                                               (idx + 1) / len(coms) * 100) + '%] ',
                                                                           smooth_bias=False,
                                                                           calc_entropy_rate=False)
            #sum_unbiased = orig_stat_dist[c_idx].sum()
            sum_stat_dist = stat_dist[c_idx].sum()
            com_nodes = set(c)
            in_neighbours = map(int, list({v_in for v in com_nodes for v_in in v.in_neighbours()} - com_nodes))
            in_neighbours_in_deg = net_indegree.a[in_neighbours].sum()
            results.append((com_s, sum_stat_dist, in_neighbours_in_deg))
    df = pd.DataFrame(columns=['com-size', 'stat_dist', 'in_neighbours_in_deg'], data=results)
    out_fn = out_dir + ds_name + '/' + ds_name + '_bs' + str(int(bias_strength)) + '.png'
    plot_df(df, bias_strength, out_fn)
    df.to_pickle(out_dir + ds_name + '/' + ds_name + '_bs' + str(int(bias_strength)) + '.df')
    return df

def main():
    multip = True  # multiprocessing flag (warning: suppresses exceptions)
    base_outdir = 'output/ecir_synthetic_coms/'
    empiric_data_dir = '/opt/datasets/'
    method = 'EV' # EV: Eigenvector, PR: PageRank
    bias_strength = [2, 5, 10, 100]
    com_sizes = [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    num_samples = 100

    print 'bias strength:', bias_strength
    datasets = list()
    datasets.append({'name': empiric_data_dir + 'wikiforschools/wiki4schools.gt', 'directed': True})
    datasets.append({'name': empiric_data_dir + 'orf_tvthek/tvthek_orf.gt', 'directed': True})
    datasets.append({'name': empiric_data_dir + 'daserste/daserste.gt', 'directed': True})

    basics.create_folder_structure(base_outdir)
    if multip:
        worker_pool = multiprocessing.Pool(processes=10)
    else:
        worker_pool = None
    results = list()

    # multi processing error q
    manager = multiprocessing.Manager()
    error_q = manager.Queue()

    async_callback = results.append
    network_prop_file = base_outdir + 'network_properties.txt'
    if os.path.isfile(network_prop_file):
        os.remove(network_prop_file)
    num_tasks = 0
    for ds in datasets:
        network_name = ds['name']
        ds.pop("name", None)
        file_name = network_name.rsplit('/', 1)[-1]
        print file_name.center(80, '=')
        net = get_network(network_name, **ds)
        network_name, file_name = file_name, network_name
        out_dir = base_outdir + network_name + '/'
        basics.create_folder_structure(out_dir)
        for bias_st in bias_strength:
            if multip:
                worker_pool.apply_async(get_stat_dist_sum, args=(net),),
                                        kwds={'bias_strength': bias_st, 'com_sizes': com_sizes,
                                              'num_samples': num_samples, 'method': method, 'ds_name': network_name, 'out_dir': base_outdir}, callback=async_callback)
                num_tasks += 1
            else:
                results.append(
                    get_stat_dist_sum(net, bias_strength=bias_st, com_sizes=com_sizes, num_samples=num_samples,
                                      method=method, ds_name=network_name, out_dir=base_outdir))
        print 'all jobs added'

    if multip:
        worker_pool.close()
        while True:
            time.sleep(60)
            remaining_processes = len(worker_pool._cache)
            print 'overall process status:', (num_tasks - remaining_processes) / num_tasks * 100, '%'
            if remaining_processes == 0:
                break
        worker_pool.join()
    while True:
        try:
            print 'checking for errors...',
            q_elem = error_q.get(timeout=1)
            print '\nError'.center(80, '-')
            print utils.color_string('[' + str(q_elem[0]) + ']')
            print q_elem[-1]
        except Queue.Empty:
            print '[OK]'
            break
    print results[0]


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start