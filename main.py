from __future__ import division
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
import multiprocessing
import datetime
import traceback
import pandas as pd
from graph_tool.all import *
from self_sim_entropy import self_sim_entropy


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
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))

        # strong sbm ============================================
        print 'sbm'.center(80, '=')
        name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'synthetic'
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
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
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))

        # price network ============================================
        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = price_network(num_nodes, m=2, gamma=1, directed=False)
        net.gp['type'] = net.new_graph_property('string')
        net.gp['type'] = 'synthetic'
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))
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
            '''
    else:
        outdir = base_outdir + 'tests/'
        basics.create_folder_structure(outdir)

        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n50'
        net = complete_graph(10)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))

        print 'sbm'.center(80, '=')
        name = 'sbm_n10_m30'
        net = generator.gen_stock_blockmodel(num_nodes=10, blocks=2, num_links=40, self_con=1, other_con=0.1)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))

        print 'price network'.center(80, '=')
        name = 'price_net_n50_m1_g2_1'
        net = price_network(30, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=async_callback)
        else:
            results.append(self_sim_entropy(net, name=name, out_dir=outdir))

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