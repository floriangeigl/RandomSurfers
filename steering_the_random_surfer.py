from __future__ import division
from tools.gt_tools import SBMGenerator, load_edge_list
from graph_tool.all import *
import tools.basics as basics
import multiprocessing
import datetime
from self_sim_entropy import self_sim_entropy
from data_io import *
import utils
import Queue
import os


def main():
    multip = True  # multiprocessing flag (warning: suppresses exceptions)
    fast_test = False
    rewires = 0
    base_outdir = 'output/steering_rnd_surfer/'
    empiric_data_dir = '/opt/datasets/'
    method = 'EV'  # EV: Eigenvector, PR: PageRank
    biases = ['adjacency', 'eigenvector', 'deg', 'inv_sqrt_deg', 'sigma', 'sigma_sqrt_deg_corrected']
    # biases = ['adjacency', 'topic_1', 'topic_2', 'topic_3']
    datasets = list()
    # datasets.append({'name': 'toy_example', 'directed': False})
    # datasets.append({'name': 'karate'})
    # datasets.append({'name': empiric_data_dir + 'karate/karate.edgelist', 'directed': False})
    if not fast_test:
        # datasets.append({'name': empiric_data_dir + 'milan_spiele/milan_spiele', 'directed': True})
        datasets.append({'name': empiric_data_dir + 'getdigital/getdigital_de_resolved_cleaned.gt', 'directed': True})
        # datasets.append({'name': empiric_data_dir + 'thinkgeek/thinkgeek', 'directed': True})
        # datasets.append({'name': empiric_data_dir + 'new_w4s/wiki4schools', 'directed': True})
        # datasets.append({'name': empiric_data_dir + 'bar_wiki/bar_wiki', 'directed': True})
        # datasets.append({'name': empiric_data_dir + 'orf_tvthek/tvthek_orf', 'directed': True})
        # datasets.append({'name': empiric_data_dir + 'daserste/daserste', 'directed': True})
        # pass
        # datasets.append({'name': '/opt/datasets/facebook/facebook', 'directed': False})
    basics.create_folder_structure(base_outdir)
    if multip:
        worker_pool = multiprocessing.Pool(processes=2)
    else:
        worker_pool = None
    results = list()

    if multip:
        manager = multiprocessing.Manager()
        error_q = manager.Queue()
    else:
        error_q = None
    async_callback = results.append
    network_prop_file = base_outdir + 'network_properties.txt'
    if os.path.isfile(network_prop_file):
        os.remove(network_prop_file)

    for ds in datasets:
        network_name = ds['name']
        ds.pop("name", None)
        file_name = network_name.rsplit('/', 1)[-1]
        print file_name.center(80, '=')
        net = get_network(network_name, **ds)
        network_name, file_name = file_name, network_name
        out_dir = base_outdir + network_name + '/'
        basics.create_folder_structure(out_dir)
        print net
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': network_name, 'out_dir': out_dir, 'biases': biases,
                                          'error_q': error_q, 'method': method}, callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=network_name, out_dir=out_dir, biases=biases, error_q=error_q,
                                            method=method))
        write_network_properties(net, network_name, network_prop_file)
        for r in xrange(rewires):
            store_fn = file_name + '_rewired_' + str(r).rjust(3, '0') + '.gt'
            if os.path.isfile(store_fn):
                net = load_graph(store_fn)
            else:
                net = net.copy()
                random_rewire(net, model='correlated')
                net.gp['type'] = 'empiric'
                net.gp['filename'] = store_fn
                net.save(store_fn)
            network_name = store_fn.rsplit('/', 1)[-1].replace('.gt', '')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,),
                                        kwds={'name': network_name, 'out_dir': out_dir, 'biases': biases,
                                              'error_q': error_q, 'method': method}, callback=async_callback)
            else:
                results.append(
                    self_sim_entropy(net, name=network_name, out_dir=out_dir, biases=biases, error_q=error_q,
                                     method=method))
            write_network_properties(net, network_name, network_prop_file)

    if multip:
        worker_pool.close()
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
    results = filter(lambda x: isinstance(x, dict), results)
    print 'result keys:', sorted(results[0].keys())
    gini_dfs = [i['gini'] for i in results]
    gini_dfs = gini_dfs[0].join(gini_dfs[1:])
    print 'gini coefs\n', gini_dfs
    gini_dfs.to_csv(base_outdir + 'gini_coefs.csv')
    # col_names.append(list(filter(lambda x: 'karate' in x, df.columns))[0])
    col_names = list()

    # col_names.append(list(filter(lambda x: 'price_net_n' in x, df.columns))[0])
    # col_names.append(list(filter(lambda x: 'sbm_weak_n' in x, df.columns))[0])
    # col_names.append(list(filter(lambda x: 'sbm_strong_n' in x, df.columns))[0])
    # gini_to_table(gini_dfs, base_outdir + 'gini_table.txt', digits=2, columns=col_names)
    # import filter_output


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start
