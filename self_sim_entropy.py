from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt
import plotting
import os
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
import numpy as np
import pandas as pd
import operator
import multiprocessing
import datetime
import traceback
import utils
import network_matrix_tools
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

font_size = 12
matplotlib.rcParams.update({'font.size': font_size})
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=225)


def self_sim_entropy(network, name, out_dir):
    base_line_type = 'adjacency'
    mem_cons = list()
    mem_cons.append(('start', utils.get_memory_consumption_in_mb()))
    adjacency_matrix = adjacency(network)

    deg_map = network.degree_property_map('total')
    weights = dict()
    if network.gp['type'] == 'empiric':
        fn = network.gp['filename']
        weights['adjacency'] = lambda: None
        if not os.path.isfile(fn + '_eigenvec'):
            A_eigvalue, A_eigvector = eigenvector(network)
            A_eigvector = np.array(A_eigvector.a)
            A_eigvector.dump(fn + '_eigenvec')
            (1 / A_eigvector).dump(fn + '_eigenvector_inverse')

            katz_sim = katz_sim_network(network, largest_eigenvalue=A_eigvalue)
            katz_sim.dump(fn + '_sigma')
            (katz_sim / np.array(deg_map.a)).dump(fn + '_sigma_deg_corrected')
            calc_cosine(adjacency_matrix, weight_direct_link=True).dump(fn + '_cosine')
            np.array(betweenness(network)[0].a).dump(fn + '_betweenness')
        A_eigvector = np.load(fn + '_eigenvec')
        weights['eigenvector'] = lambda: A_eigvector
        weights['eigenvector_inverse'] = lambda: np.load(fn + '_eigenvector_inverse')
        # test = utils.softmax(np.load(fn + '_eigenvector_inverse'), t=0.01)
        # weights['test'] = lambda: test
        weights['sigma'] = lambda: np.load(fn + '_sigma')
        weights['sigma_deg_corrected'] = lambda: np.load(fn + '_sigma_deg_corrected')
        weights['cosine'] = lambda: np.load(fn + '_cosine')
        weights['betweenness'] = lambda: np.load(fn + '_betweenness')
    else:
        A_eigvalue, A_eigvector = eigenvector(network)
        A_eigvector = A_eigvector.a
        weights['adjacency'] = lambda: None
        weights['eigenvector'] = lambda: A_eigvector
        weights['eigenvector_inverse'] = lambda: 1 / A_eigvector
        # test = utils.softmax(weights['eigenvector_inverse'](), t=0.01)
        # weights['test'] = lambda: test
        katz_sim = network_matrix_tools.katz_sim_network(adjacency_matrix, A_eigvalue)
        weights['sigma'] = lambda: katz_sim
        weights['sigma_deg_corrected'] = lambda: katz_sim / np.array(deg_map.a)
        weights['cosine'] = lambda: network_matrix_tools.calc_cosine(adjacency_matrix, weight_direct_link=True)
        weights['betweenness'] = lambda: np.array(betweenness(network)[0].a)

    mem_cons.append(('stored weight functions', utils.get_memory_consumption_in_mb()))
    weights = {key.replace(' ', '_'): val for key, val in weights.iteritems()}

    entropy_df = pd.DataFrame()
    sort_df = []
    print_prefix = utils.color_string('[' + name + ']')
    print print_prefix, 'calc graph-layout'
    try:
        pos = sfdp_layout(network, groups=network.vp['com'], mu=3.0)
    except KeyError:
        pos = sfdp_layout(network)

    corr_df = pd.DataFrame(columns=['deg'], data=deg_map.a)
    stat_distributions = {}
    for key, weight in sorted(weights.iteritems(), key=operator.itemgetter(0)):
        print print_prefix, '[' + key + '] calc stat dist and entropy rate'

        # calc metric
        weight = weight()

        # replace infs and nans with zero
        if weight is not None:
            num_nans = np.isnan(weight).sum()
            num_infs = np.isinf(weight).sum()
            if num_nans > 0 or num_infs > 0:
                print print_prefix, '[' + key + ']:', utils.color_string(
                    'shape:' + str(weight.shape) + '|replace nans(' + str(num_nans) + ') and infs (' + str(
                        num_infs) + ') of metric with zero', type=utils.bcolor.RED)
                weight[np.isnan(weight) | np.isinf(weight)] = 0
        if weight is None or len(weight.shape) == 1:
            weights[key] = weight

        #print 'weight', weight
        ent, stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, weight)
        stat_distributions[key] = stat_dist
        print print_prefix, '[' + key + '] entropy rate:', ent
        entropy_df.at[0, key] = ent
        sort_df.append((key, ent))
        corr_df[key] = stat_dist
        mem_cons.append(('after ' + key, utils.get_memory_consumption_in_mb()))
    if base_line_type == 'adjacency':
        base_line_abs_vals = stat_distributions['adjacency']
    elif base_line_type == 'uniform':
        base_line_abs_vals = np.array([[1. / network.num_vertices()]])
    else:
        print print_prefix, '[' + key + ']', utils.color_string(('unkown baseline type: ' + base_line_type).upper(),
            utils.bcolors.RED)
        exit()

    #base_line = base_line_abs_vals / 100  # /100 for percent
    base_line = base_line_abs_vals
    vertex_size = network.new_vertex_property('float')
    vertex_size.a = base_line
    min_stat_dist = min([min(i) for i in stat_distributions.values()])
    max_stat_dist = max([max(i) for i in stat_distributions.values()])
    min_val = min([min(i/base_line) for i in stat_distributions.values()])
    max_val = max([max(i/base_line) for i in stat_distributions.values()])
    trapped_df = pd.DataFrame(index=range(network.num_vertices()))

    # calc max vals for graph-coloring
    all_vals = [j for i in stat_distributions.values() for j in i / base_line]
    max_val = np.mean(all_vals) + (2 * np.std(all_vals))

    # plot all biased graphs and add biases to trapped plot
    for key, stat_dist in sorted(stat_distributions.iteritems(), key=operator.itemgetter(0)):
        stat_dist_diff = stat_dist / base_line
        stat_dist_diff[np.isclose(stat_dist_diff, 1.)] = 1.
        plotting.draw_graph(network, color=stat_dist_diff, min_color=min_val, max_color=max_val, sizep=deg_map,
                            groups='com', output=out_dir + name + '_graph_' + key, pos=pos)
        plt.close('all')

        stat_dist_ser = pd.Series(data=stat_dist)
        x = ('stationary value of adjacency', base_line_abs_vals)
        y = (key.replace('_', ' ') + ' difference', stat_dist_diff)
        plotting.create_scatter(x=x, y=y, fname=out_dir + name + '_scatter_' + key)

        x = ('popularity', np.array(deg_map.a))
        plotting.create_scatter(x=x, y=y, fname=out_dir + name + '_scatter_popularity_' + key)

        #x = ('betweenness', np.array(weights['betweenness']))
        #plotting.create_scatter(x=x, y=y, fname=out_dir + name + '_scatter_between_' + key)

        # plot stationary distribution
        if False:
            stat_dist_fname = out_dir + name + '_stat_dist_' + key + '.png'
            plotting.plot_stat_dist(stat_dist_ser, stat_dist_fname, bins=25, range=(min_stat_dist, max_stat_dist), lw=0)

        # calc gini coef and trapped values
        stat_dist_ser.sort(ascending=True)
        stat_dist_ser.index = range(len(stat_dist_ser))
        key += ' gc:' + ('%.4f' % utils.gini_coeff(stat_dist_ser))
        trapped_df[key] = stat_dist_ser.cumsum()
        trapped_df[key] /= trapped_df[key].max()
        mem_cons.append(('after ' + key + ' scatter', utils.get_memory_consumption_in_mb()))

    # add uniform to trapped plot
    key = 'uniform'
    uniform = np.array([1]*len(trapped_df))
    key += ' gc:' + ('%.4f' % utils.gini_coeff(uniform))
    trapped_df[key] = uniform.cumsum()
    trapped_df[key] /= trapped_df[key].max()

    trapped_df.index += 1
    trapped_df['idx'] = np.round(np.array(trapped_df.index).astype('float') / len(trapped_df) * 100)

    if len(trapped_df) > 50:
        trapped_df['idx'] = trapped_df['idx'].astype('int')
        trapped_df['idx'] = trapped_df['idx'].apply(lambda x: int(x / 5) * 5)
        trapped_df.drop_duplicates(subset=['idx'], inplace=True)
    trapped_df.plot(x='idx', lw=2, alpha=0.5, style=['-o', '-v', '-^', '-s', '-*', '-D'])
    plt.yticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    plt.xlabel('percent of nodes')
    plt.ylim([0, 1])
    plt.ylabel('cumulative sum of stationary distribution values')
    plt.tight_layout()
    plt.savefig(out_dir + name + '_trapped.png')
    plt.close('all')

    try:
        num_cols = len(corr_df.columns) * 3
        pd.scatter_matrix(corr_df, figsize=(num_cols, num_cols), diagonal='kde', range_padding=0.2, grid=True)
        plt.tight_layout()
        plt.savefig(out_dir + name + '_scatter_matrix.png')
        plt.close('all')
    except:
        print print_prefix, '[', key, ']', utils.color_string('plot scatter-matrix failed'.upper(), utils.bcolors.RED)

    sorted_keys, sorted_values = zip(*sorted(sort_df, key=lambda x: x[1], reverse=True))
    if len(set(sorted_values)) == 1:
        sorted_keys = sorted(sorted_keys)
    entropy_df = entropy_df[list(sorted_keys)]
    print print_prefix, ' entropy rates:\n', entropy_df
    ax = entropy_df.plot(kind='bar', label=[i.replace('_', ' ') for i in entropy_df.columns])
    min_e, max_e = entropy_df.loc[0].min(), entropy_df.loc[0].max()
    ax.set_ylim([min_e * 0.95, max_e * 1.01])
    #ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(True)
    #ax.spines['right'].set_visible(True)
    plt.ylabel('entropy rate')
    plt.legend(loc='upper left')
    plt.xlim([-1, 0.4])
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.savefig(out_dir + name + '_entropy_rates.png')
    plt.close('all')
    mem_df = pd.DataFrame(columns=['state', 'memory in MB'], data=mem_cons)
    mem_df.plot(x='state', y='memory in MB', rot=45, label='MB')
    plt.title('memory consumption')
    plt.tight_layout()
    plt.savefig(out_dir + name + '_mem_status.png')
    plt.close('all')
    print print_prefix, utils.color_string('>>all done<<', type=utils.bcolors.GREEN)


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
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

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
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
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
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

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
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
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
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
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
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)

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
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
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
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

        print 'sbm'.center(80, '=')
        name = 'sbm_n10_m30'
        net = generator.gen_stock_blockmodel(num_nodes=10, blocks=2, num_links=40, self_con=1, other_con=0.1)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

        print 'price network'.center(80, '=')
        name = 'price_net_n50_m1_g2_1'
        net = price_network(30, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

        print 'quick tests done'.center(80, '=')

    if multip:
        worker_pool.close()
        worker_pool.join()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start