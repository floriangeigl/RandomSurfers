from __future__ import division
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
import multiprocessing
import datetime
import traceback
import pandas as pd
from graph_tool.all import *
from self_sim_entropy import self_sim_entropy
import powerlaw as fit_powerlaw
import numpy as np
from graph_tool.all import *
import random
import os


def write_network_properties(network, net_name, out_filename):
    with open(out_filename, 'a') as f:
        f.write('=' * 80)
        f.write('\n')
        f.write(net_name + '\n')
        f.write('\tnum nodes: ' + str(network.num_vertices()) + '\n')
        f.write('\tnum edges: ' + str(network.num_edges()) + '\n')
        deg_map = network.degree_property_map('total')
        res = fit_powerlaw.Fit(np.array(deg_map.a))
        f.write('\tpowerlaw alpha: ' + str(res.power_law.alpha) + '\n')
        f.write('\tpowerlaw xmin: ' + str(res.power_law.xmin) + '\n')
        if 'com' in network.vp.keys():
            f.write('\tcom assortativity: ' + str(assortativity(network, network.vp['com'])) + '\n')
            f.write('\tcom modularity:' + str(modularity(network, network.vp['com'])) + '\n')
        f.write('\tdeg scalar-assortativity: ' + str(scalar_assortativity(network, deg_map)) + '\n')
        f.write('\tpseudo diameter: ' + str(
            pseudo_diameter(network, random.sample(list(network.vertices()), 1)[0])[0]) + '\n')
        f.write('\tlargest eigenvalue: ' + str(eigenvector(network)[0]) + '\n')
        f.write('=' * 80)
        f.write('\n')


def main():
    generator = SBMGenerator()
    base_outdir = 'output/'
    basics.create_folder_structure(base_outdir)

    first_two_only = False  # quick test flag
    test = False  # basic test flag
    multip = True  # multiprocessing flag (warning: suppresses exceptions)

    if first_two_only:
        multip = False
    worker_pool = multiprocessing.Pool(processes=14)
    results = list()
    async_callback = results.append
    network_prop_file = base_outdir + 'network_properties.txt'
    if os.path.isfile(network_prop_file):
        os.remove(network_prop_file)

    if not test:
        num_links = 1200
        num_nodes = 300
        num_blocks = 5

        # karate ninja bam bam ============================================
        print 'karate'.center(80, '=')
        name = 'karate'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/karate/karate.edgelist')
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'empiric'
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        # strong sbm ============================================
        print 'sbm'.center(80, '=')
        name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'synthetic'
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        if first_two_only:
            exit()

        # weak sbm ============================================
        print 'sbm'.center(80, '=')
        name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.7)
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'synthetic'
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        # price network ============================================
        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = price_network(num_nodes, m=2, gamma=1, directed=False)
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'synthetic'
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        if False:
            # wiki4schools ============================================
            print 'wiki4schools'.center(80, '=')
            name = 'wiki4schools'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/wikiforschools/graph')
            # net.vp['com'] = load_property(net, '/opt/datasets/wikiforschools/artid_catid', type='int')
            net.gp['type'] = net.new_graph_property('string')
            net.gp['type'] = 'empiric'
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=async_callback)
            else:
                results.append(self_sim_entropy(net, name=name, out_dir=outdir))
            write_network_properties(net, name, network_prop_file)
            generator.analyse_graph(net, outdir + name, draw_net=False)

            # facebook ============================================
            print 'facebook'.center(80, '=')
            name = 'facebook'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/facebook/facebook')
            # net.vp['com'] = load_property(net, '/opt/datasets/facebook/facebook_com', type='int', line_groups=True)
            net.gp['type'] = net.new_graph_property('string')
            net.gp['type'] = 'empiric'
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=async_callback)
            else:
                results.append(self_sim_entropy(net, name=name, out_dir=outdir))
            write_network_properties(net, name, network_prop_file)
            generator.analyse_graph(net, outdir + name, draw_net=False)

            '''# enron ============================================
            print 'enron'.center(80, '=')
            name = 'enron'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/enron/enron')
            net.gp['type'] = net.new_graph_property('string')
            net.gp['type'] = 'empiric'
            # net.vp['com'] = load_property(net, '/opt/datasets/youtube/youtube_com', type='int', line_groups=True)
            print 'vertices:', net.num_vertices()
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=async_callback)
            else:
                results.append(self_sim_entropy(net, name=name, out_dir=outdir))
            write_network_properties(net, name, network_prop_file)
            generator.analyse_graph(net, outdir + name, draw_net=False)
            '''
    else:
        outdir = base_outdir + 'tests/'
        basics.create_folder_structure(outdir)

        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n50'
        net = complete_graph(10)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        print 'sbm'.center(80, '=')
        name = 'sbm_n10_m30'
        net = generator.gen_stock_blockmodel(num_nodes=10, blocks=2, num_links=40, self_con=1, other_con=0.1)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)

        print 'price network'.center(80, '=')
        name = 'price_net_n50_m1_g2_1'
        net = price_network(30, m=2, gamma=1, directed=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
        write_network_properties(net, name, network_prop_file)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        print 'quick tests done'.center(80, '=')

    if multip:
        worker_pool.close()
        worker_pool.join()
    gini_dfs = [i['gini'] for i in results]
    gini_dfs = gini_dfs[0].join(gini_dfs[1:])
    print 'gini coefs\n', gini_dfs
    gini_dfs.to_csv(outdir + 'gini_coefs.csv')


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start