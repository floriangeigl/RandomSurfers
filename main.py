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


def main():
    generator = SBMGenerator()
    base_outdir = 'output/'
    basics.create_folder_structure(base_outdir)

    first_two_only = True  # quick test flag (disables multiprocessing to get possibles exceptions)
    toy_example = False
    multip = True  # multiprocessing flag (warning: suppresses exceptions)
    synthetic = False
    empiric_crawled = True
    empiric_downloaded = False
    karate_club = True
    if first_two_only:
        multip = False
        synthetic = True
        karate_club = True
    if multip:
        worker_pool = multiprocessing.Pool(processes=14)
    else:
        worker_pool = None
    results = list()
    biases = ['adjacency', 'eigenvector', 'deg', 'inv_sqrt_deg', 'sigma', 'sigma_sqrt_deg_corrected']
    if multip:
        manager = multiprocessing.Manager()
        error_q = manager.Queue()
    else:
        error_q = None
    async_callback = results.append
    network_prop_file = base_outdir + 'network_properties.txt'
    if os.path.isfile(network_prop_file):
        os.remove(network_prop_file)

    num_links = 1200
    num_nodes = 300
    num_blocks = 5

    if toy_example:
        # strong sbm ============================================
        print 'toy_example'.center(80, '=')
        name = 'toy_example'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = price_network(5, m=2, directed=False)
        net.gp['type'] = net.new_graph_property('string')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)

    if karate_club:
        # karate ninja bam bam ============================================
        print 'karate'.center(80, '=')
        name = 'karate'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/karate/karate.edgelist', directed=False)
        net.gp['type'] = net.new_graph_property('string', 'empiric')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)

    if synthetic:

        # strong sbm ============================================
        print 'sbm strong'.center(80, '=')
        name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)

        if first_two_only:
            exit()

        # weak sbm ============================================
        print 'sbm weak'.center(80, '=')
        name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.7)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')

        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)

        # price network ============================================
        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = price_network(num_nodes, m=4, gamma=1, directed=False)
        net.gp['type'] = net.new_graph_property('string', 'synthetic')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,),
                                    kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)

    if empiric_crawled:
        empiric_data_dir = '/opt/datasets/'
        empiric_data_sets = list()
        empiric_data_sets.append('milan_spiele')
        empiric_data_sets.append('getdigital')
        empiric_data_sets.append('thinkgeek')

        for name in empiric_data_sets:
            print name.center(80, '=')
            fname = empiric_data_dir + name + '/' + name + '.gt'
            net = load_graph(fname)
            l = label_largest_component(net, directed=True)
            net.set_vertex_filter(l)
            net.purge_vertices()
            net.gp['filename'] = net.new_graph_property('string', fname)
            net.gp['type'] = net.new_graph_property('string', 'empiric')
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,),
                                        kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                        callback=async_callback)
            else:
                results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
            write_network_properties(net, name, network_prop_file)
            # generator.analyse_graph(net, outdir + name, draw_net=False)
    if empiric_downloaded:
        # wiki4schools ============================================
        print 'wiki4schools'.center(80, '=')
        name = 'wiki4schools'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/wikiforschools/graph', directed=True)
        # net.vp['com'] = load_property(net, '/opt/datasets/wikiforschools/artid_catid', type='int')
        net.gp['type'] = net.new_graph_property('string', 'empiric')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)

        # facebook ============================================
        print 'facebook'.center(80, '=')
        name = 'facebook'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/facebook/facebook', directed=False)
        # net.vp['com'] = load_property(net, '/opt/datasets/facebook/facebook_com', type='int', line_groups=True)
        net.gp['type'] = net.new_graph_property('string', 'empiric')
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)
        '''
        # enron ============================================
        print 'enron'.center(80, '=')
        name = 'enron'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/enron/enron')
        net.gp['type'] = net.new_graph_property('string', 'empiric')
        # net.vp['com'] = load_property(net, '/opt/datasets/youtube/youtube_com', type='int', line_groups=True)
        print 'vertices:', net.num_vertices()
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir, 'biases': biases, 'error_q': error_q},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=error_q))
        write_network_properties(net, name, network_prop_file)
        # generator.analyse_graph(net, outdir + name, draw_net=False)
        '''

    if multip:
        worker_pool.close()
        worker_pool.join()
    while True:
        try:
            q_elem = error_q.get(timeout=3)
            print 'Error'.center(80, '-')
            print utils.color_string('[' + str(q_elem[0]) + ']')
            print q_elem[-1]
        except:
            break
    results = filter(lambda x: isinstance(x, dict), results)
    gini_dfs = [i['gini'] for i in results]
    gini_dfs = gini_dfs[0].join(gini_dfs[1:])
    print 'gini coefs\n', gini_dfs
    gini_dfs.to_csv(base_outdir + 'gini_coefs.csv')
    gini_to_table(gini_dfs, base_outdir + 'gini_table.txt', digits=2)
    # import filter_output



if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start