from __future__ import division
from graph_tool.all import *
from tools.gt_tools import SBMGenerator, load_edge_list
import tools.basics as basics
import multiprocessing
import datetime
from self_sim_entropy import self_sim_entropy
from data_io import *
import utils
import Queue


def get_network(name, directed=True):
    # synthetic params
    num_links = 1200
    num_nodes = 300
    num_blocks = 5
    price_net_m = 4
    price_net_g = 1
    strong_others_con = 0.1
    weak_others_con = 0.7
    print 'get network:', name.rsplit('/', 1)[-1]
    if name == 'toy_example':
        net = price_network(5, m=2, directed=directed)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')

    elif name == 'karate':
        directed = False
        net = load_edge_list('/opt/datasets/karate/karate.edgelist', directed=directed)
        net.gp['type'] = net.new_graph_property('string', 'empiric')

    elif name == 'sbm_strong' or name == 'sbm_weak':
        if name == 'sbm_weak':
            other_con = weak_others_con
        else:
            other_con = strong_others_con
        generator = SBMGenerator()
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links,
                                             other_con=other_con, directed=directed)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')

    elif name == 'price_network':
        net = price_network(num_nodes, m=price_net_m, gamma=price_net_g, directed=directed)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')

    else:
        net = load_edge_list(name, directed=directed, vertex_id_dtype=None)
        net.gp['filename'] = net.new_graph_property('string', name)
        net.gp['type'] = net.new_graph_property('string', 'empiric')

    l = label_largest_component(net, directed=directed)
    net.set_vertex_filter(l)
    net.purge_vertices()
    net.clear_filters()
    return net


def main():
    multip = True  # multiprocessing flag (warning: suppresses exceptions)
    fast_test = True
    rewires = 2
    base_outdir = 'output/'
    empiric_data_dir = '/opt/datasets/'
    method = 'EV' # EV: Eigenvector, PR: PageRank
    biases = ['adjacency', 'eigenvector', 'deg', 'inv_sqrt_deg', 'sigma', 'sigma_sqrt_deg_corrected']
    datasets = list()
    datasets.append({'name': 'toy_example', 'directed': False})
    datasets.append({'name': 'karate'})
    if not fast_test:
        datasets.append({'name': empiric_data_dir + 'milan_spiele', 'directed': True})
        datasets.append({'name': empiric_data_dir + 'getdigital', 'directed': True})
        datasets.append({'name': empiric_data_dir + 'thinkgeek', 'directed': True})
        datasets.append({'name': '/opt/datasets/wikiforschools/graph', 'directed': True})
        # datasets.append({'name': '/opt/datasets/facebook/facebook', 'directed': False})

    basics.create_folder_structure(base_outdir)
    if multip:
        worker_pool = multiprocessing.Pool(processes=14)
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
        print network_name.center(80, '=')
        out_dir = base_outdir + network_name + '/'
        basics.create_folder_structure(out_dir)
        ds.pop("name", None)
        net = get_network(network_name, **ds)
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
            random_rewire(net, model='correlated')
            tmp_network_name = network_name + '_rewired_' + str(r).rjust(len(str(rewires - 1)), '0')
            net.gp['type'] = net.new_graph_property('string', 'synthetic')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,),
                                        kwds={'name': tmp_network_name, 'out_dir': out_dir, 'biases': biases,
                                              'error_q': error_q, 'method': method}, callback=async_callback)
            else:
                results.append(
                    self_sim_entropy(net, name=tmp_network_name, out_dir=out_dir, biases=biases, error_q=error_q,
                                     method=method))
            write_network_properties(net, tmp_network_name, network_prop_file)

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

    #col_names.append(list(filter(lambda x: 'price_net_n' in x, df.columns))[0])
    #col_names.append(list(filter(lambda x: 'sbm_weak_n' in x, df.columns))[0])
    #col_names.append(list(filter(lambda x: 'sbm_strong_n' in x, df.columns))[0])
    #gini_to_table(gini_dfs, base_outdir + 'gini_table.txt', digits=2, columns=col_names)
    # import filter_output

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start