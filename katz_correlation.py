from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats

import math

from graph_tool.all import *
import graph_tool as gt

from community import SBMGenerator
from linalg import matrix_spectrum
from linalg import katz_sim_matrix
from linalg import adj_matrix
import multiprocessing
from multiprocessing import Pool
import navigator as nav
import operator
import os
import traceback
import utils as ut
import time
from collections import defaultdict
import pandas as pd
import datetime
np.set_printoptions(linewidth=225)

def blocks2():
    blocks = np.array([40, 30, 30])
    blockp = np.matrix("0.8, 0.01, 0.05; 0.01, 0.7, 0.05; 0.05, 0.05, 0.6")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A


def blocks3():
    blocks = np.array([40, 30, 30])
    blockp = np.matrix("0.02, 0.02, 0.02; 0.02, 0.02, 0.02; 0.02, 0.02, 0.02")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A


def blocks4():
    blocks = np.array([40, 30, 30])
    blockp = np.matrix("0.9, 0.01, 0.01; 0.01, 0.9, 0.01; 0.01, 0.01, 0.9")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A


def generate_blocks(blocks, blockp):
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A


def generate_blocks_degree_corrected(blocks, blockp, degree_sequence):
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate_degree_corrected(degree_sequence)
    return A


'''
def wiki():
    gif = "/home/dhelic/work/courses/netsci_slides/examples/measures/ipython/graph"
    G, nodelist = read_graph(gif)
    A = adj_matrix(G, nodelist)
    print A
    return A, G.number_of_nodes()
'''


def corr(x, y):
    print x, y
    if x == y:
        return 0.099
    else:
        return 0.00025


def graph_gen(self_con, other_con, nodes=500, groups=5):
    for i in range(10):
        print stats.powerlaw.rvs(2.1, size=1)
    g, bm = random_graph(nodes, lambda: (1 - stats.powerlaw.rvs(2.7, size=1)) * 50, directed=False,
                         model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)),
                         vertex_corr=corr)
    # lambda: (1 - stats.powerlaw.rvs(2.7, size=1)) * 50
    #lambda: stats.poisson.rvs(10),
    return g, bm


def gen_stock_blockmodel(num_nodes=100, blocks=10, self_con=1, other_con=0.2, directed=False):
    g = Graph(directed=directed)



def gen_com_pmap(net,blocks):
    com = net.new_vertex_property('int')
    current_node_id = 0
    for com_id, i in enumerate(blocks):
        for j in range(i):
            com[net.vertex(current_node_id)] = com_id
            current_node_id += 1
    return com


def calc_entropy(graph, A, sigma):
    total_entropy = 0
    AT = A.T
    selection_range = set(map(int, graph.vertices()))
    for v in selection_range:
        # exclude loop
        current_selection = list(selection_range - {v})
        # stack the katz row of the target vertex N-1 times
        stacked_row_sigma = sigma[[v] * (sigma.shape[0] - 1), :]
        # multiply katz with transposed A -> only katz values on real links
        res = np.multiply(stacked_row_sigma, AT[current_selection, :])
        # calc entropy per row and add it to the overall entropy
        ent = stats.entropy(res.T)
        total_entropy += ent.sum()
        if False:
            print 'target vertex:', int(v)
            print 'Katz row'
            print katz_mat[int(v), :]
            print 'AT'
            print AT[current_selection, :]
            print 'KATZ*AT'
            # res = res[]
            print res
            print 'entropy'
            print ent
            # print entropy(res_trim.T)
            #break
    max_entropy = np.log(AT.sum(0)).sum() / A.shape[0]
    num_v = graph.num_vertices()
    avg_entropy = total_entropy / (num_v * (num_v - 1))
    return avg_entropy, max_entropy


def do_calc(i, blocks, blockp, num_pairs, com_greedy, legend, plot, plot_dir, id):
    try:
        c_list = ut.get_colors_list()
        p_id = multiprocessing.current_process()._identity[0]
        c = c_list[p_id % len(c_list)]
        p_name = ut.color_string('[Worker ' + str(p_id) + ']', type=c)
        print p_name, 'start calc of id:', id
        mi = 5
        ma = 15
        power = 1
        if i < len(blockp):
            A = generate_blocks(blocks, 4 * np.matrix(blockp[i]))
            g = gt.Graph(directed=False)
            for node_id in range(A.shape[0]):
                g.add_vertex()
            g.add_edge_list(np.transpose(A.nonzero()))
            com = gen_com_pmap(g, blocks)
            if plot:
                pos = gt.draw.sfdp_layout(g)
                deg = g.degree_property_map("total")
                gt.draw.graph_draw(g, pos=pos, vertex_size=prop_to_size(deg, mi=mi, ma=ma, power=power),
                                   vertex_fill_color=com, output=plot_dir + "blocks%d.png" % i)
        elif i == len(blockp):
            g = gt.generation.price_network(500, m=20, directed=False)
            com = g.new_vertex_property('int')
            A = np.zeros((500, 500))
            gt.spectral.adjacency(g).todense(out=A)
            if plot:
                pos = gt.draw.sfdp_layout(g)
                deg = g.degree_property_map("total")
                gt.draw.graph_draw(g, pos=pos, vertex_size=prop_to_size(deg, mi=mi, ma=ma, power=power),
                                   output=plot_dir + "power-law.png")
        else:
            degree_sequence = (1 - stats.powerlaw.rvs(2.7, size=500)) * 80 + 5
            # print degree_sequence
            A = generate_blocks_degree_corrected(blocks, np.matrix(blockp[i - 2]), degree_sequence)
            g = gt.Graph(directed=False)
            g.add_edge_list(np.transpose(A.nonzero()))
            com = gen_com_pmap(g, blocks)
            if plot:
                pos = gt.draw.sfdp_layout(g)
                deg = g.degree_property_map("total")
                gt.draw.graph_draw(g, pos=pos, vertex_fill_color=com,
                                   vertex_size=prop_to_size(deg, mi=mi, ma=ma, power=power),
                                   vcmap=matplotlib.cm.gist_heat_r,
                                   output=plot_dir + "power-law-blocks.png")

                #plot the largest component
                l = gt.topology.label_largest_component(g)
                u = gt.GraphView(g, vfilt=l)
                pos = gt.draw.sfdp_layout(u)
                deg = u.degree_property_map("total")
                gt.draw.graph_draw(u, pos=pos, vertex_fill_color=com,
                                   vertex_size=prop_to_size(deg, mi=mi, ma=ma, power=power),
                                   output=plot_dir + "power-law-blocks-lcc.png")

        l, v = matrix_spectrum(A)
        kappa_1 = l[0].real
        alpha_max = 1.0 / kappa_1
        gt.stats.remove_parallel_edges(g)
        print p_name, legend[i], 'nodes:', g.num_vertices(), '||edges:', g.num_edges()
        print p_name, 'kappa1:', kappa_1
        print p_name, 'alpha max:', alpha_max

        alpha_max *= 0.99
        alpha = alpha_max
        sigma_global = katz_sim_matrix(A, alpha)
        avg_entropy, max_entropy = calc_entropy(g, A, sigma_global)
        np.fill_diagonal(sigma_global, 0.0)
        gpearson = stats.pearsonr(A.flatten(), sigma_global.flatten())
        print p_name, "global vs adjacency"
        print p_name, gpearson

        alphas = [0.0, alpha_max * 1e-8, alpha_max * 1e-7, alpha_max * 1e-6, alpha_max * 1e-5, alpha_max * 1e-4,
                  alpha_max * 1e-3, alpha_max * 1e-2,
                  alpha_max * 1e-1, alpha_max * 0.15, alpha_max * 0.2, alpha_max * 0.25, alpha_max * 0.3,
                  alpha_max * 0.35, alpha_max * 0.4,
                  alpha_max * 0.45, alpha_max * 0.5, alpha_max * 0.55, alpha_max * 0.6, alpha_max * 0.65,
                  alpha_max * 0.7, alpha_max * 0.75,
                  alpha_max * 0.8, alpha_max * 0.85, alpha_max * 0.9, alpha_max * 0.95, alpha_max * 0.96,
                  alpha_max * 0.97, alpha_max * 0.98,
                  alpha_max * 0.99, alpha_max]

        data = [gpearson[0]]
        pearson80 = 0.0
        set_pearson80 = False
        for idx, alpha in enumerate(alphas):
            if alpha == 0.0:
                continue
            # print p_name, 'katz sim:', idx / len(alphas) * 100, '%'
            sigma = katz_sim_matrix(A, alpha)
            np.fill_diagonal(sigma, 0.0)
            pearson = stats.pearsonr(sigma.flatten(), sigma_global.flatten())
            if not set_pearson80 and pearson[0] > 0.8:
                pearson80 = alpha * kappa_1
                set_pearson80 = True
            data.append(pearson[0])
            # print "global vs %e"%alpha
            #print pearson[0]
        if not com_greedy:
            com = None
        sr, stretch = nav.random_walk(g, max_steps=0, avoid_revisits=True, num_pairs=num_pairs, com=com)
        stretch_avg = np.mean(stretch)
        res_dict = dict()
        res_dict['type'] = legend[i]
        res_dict['n'] = g.num_vertices()
        res_dict['m'] = g.num_edges()
        res_dict['kappa_1'] = kappa_1
        res_dict['alpha_max'] = 1/kappa_1
        res_dict['gpearson'] = gpearson[0]
        res_dict['pearson80'] = pearson80
        res_dict['alpha80'] = pearson80 / kappa_1
        res_dict['stretch'] = stretch_avg
        res_dict['sr'] = sr
        res_dict['entropy_avg'] = avg_entropy
        res_dict['entropy_max'] = max_entropy
    except:
        print str(traceback.format_exc())
        exit()
    return ([a * kappa_1 for a in alphas], data, res_dict), id


def plot_scatter(data_dict, first_prop, second_prop, output_dir):
    print 'plot scatter:', first_prop, 'vs', second_prop,
    scatter_df = None
    x = list()
    y = list()

    for idx, (key, val) in enumerate(data_dict.iteritems()):
        if scatter_df is None:
            scatter_df = pd.DataFrame(columns=[key + '_' + first_prop, key + '_' + second_prop], data=val)
        else:
            scatter_df[key + '_' + first_prop], scatter_df[key + '_' + second_prop] = zip(*val)
        x.append(key + '_' + first_prop)
        y.append(key + '_' + second_prop)
    ax = None
    colors = ['blue', 'green', 'yellow', 'black', 'pink']
    marker = ['^', 'D', 's', 'p', 'o']
    for idx, (i, j) in enumerate(zip(x, y)):
        ax = scatter_df.plot(x=i, y=j, legend=True, label=i.replace('_' + first_prop, ''), kind='scatter', ax=ax,
                             c=colors[idx % len(colors)], marker=marker[idx % len(marker)], s=50, alpha=0.75, lw=0)
    ax.set_ylabel(second_prop)
    ax.set_xlabel(first_prop)
    plt.legend(loc='upper right')
    plt.savefig(output_dir + first_prop + '_' + second_prop + '_scatter.png', dpi=150)
    plt.close('all')
    print '[OK]'


def main():
    n_exper = 15
    plot = True
    num_pairs = 1000
    com_greedy = True
    blocks = np.array([100, 100, 100, 100, 100])
    blockp = [
        "0.02, 0.02, 0.02, 0.02, 0.02; 0.02, 0.02, 0.02, 0.02, 0.02; 0.02, 0.02, 0.02, 0.02, 0.02; 0.02, 0.02, 0.02, 0.02, 0.02; 0.02, 0.02, 0.02, 0.02, 0.02",
        "0.08, 0.005, 0.005, 0.005, 0.005; 0.005, 0.08, 0.005, 0.005, 0.005; 0.005, 0.005, 0.08, 0.005, 0.002; 0.005, 0.005, 0.005, 0.08, 0.005; 0.005, 0.005, 0.005, 0.005, 0.08",
        "0.099, 0.00025, 0.00025, 0.00025, 0.00025; 0.00025, 0.099, 0.00025, 0.00025, 0.00025; 0.00025, 0.00025, 0.099, 0.00025, 0.00025; 0.00025, 0.00025, 0.00025, 0.099, 0.00025; 0.00025, 0.00025, 0.00025, 0.00025, 0.099"]
    legend = ["Random", "Weak Comm.", "Strong Comm.", "Power-Law", "Deg. Stoch. Blocks"]

    output_dir = "output/"
    rfile = output_dir + "/gvsl.txt"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # A = blocks2()
    #A = blocks3()
    #A = blocks4()
    #A, n = wiki()
    #adata = []
    #aalphas = []

    results = []
    res_appender = results.append

    worker_pool = Pool(processes=12)
    id = 0
    for count in range(n_exper):
        for i in range(len(blockp) + 2):
            worker_pool.apply_async(func=do_calc,
                                    args=(i, blocks, blockp, num_pairs, com_greedy, legend, plot, output_dir, id),
                                    callback=res_appender)
            if i > len(blockp):
                plot = False
            id += 1
    worker_pool.close()
    while True:
        time.sleep(60)
        remaining_processes = len(worker_pool._cache)
        print ut.color_string(
            str(' overall process status:' + str((id - remaining_processes) / id * 100) + '% ').center(100, '-'),
            ut.bcolors.LIGHT_BLUE)
        if remaining_processes == 0:
            break
    worker_pool.join()
    print 'all workers finished'
    print 'sort data'
    results.sort(key=operator.itemgetter(1))
    results, result_ids = zip(*results)
    aalphas, adata, file_cont = zip(*results)
    gpearson_stretch_data = defaultdict(list)
    entropyavg_stretch_data = defaultdict(list)
    entropymax_stretch_data = defaultdict(list)
    with open(rfile, 'w') as f:
        for i in file_cont:
            net_type = i['type']
            gpearson_stretch_data[net_type].append((i['gpearson'], i['stretch']))
            entropyavg_stretch_data[net_type].append((i['entropy_avg'], i['stretch']))
            entropymax_stretch_data[net_type].append((i['entropy_max'], i['stretch']))
            f.write(net_type + '\n')
            del i['type']
            for key, val in sorted(i.iteritems(), key=operator.itemgetter(0)):
                f.write(str(key) + ': ' + str(val) + '\n')
            f.write('-' * 80)
            f.write('\n')

    plot_scatter(gpearson_stretch_data, 'gpearson', 'stretch', output_dir)
    plot_scatter(entropyavg_stretch_data, 'entropy_avg', 'stretch', output_dir)
    plot_scatter(entropymax_stretch_data, 'entropy_max', 'stretch', output_dir)

    print 'plot lineplots'
    for i in range(len(blockp) + 2):
        data = []
        alphas = aalphas[i]
        #print alphas
        for j in range(n_exper):
            data.append(adata[j * (len(blockp) + 2) + i])
        d = np.array(data)
        print d.shape
        avg = np.mean(d, axis=0)
        print avg
        sem = stats.sem(d, axis=0)
        print sem
        # std = np.std(d, axis=0)
        # print std
        # print std / math.sqrt(10)
        plt.errorbar(alphas, avg, yerr=sem, label=legend[i])

    plt.grid()
    plt.title("Corr. between global and fractional knowledge ($n=1000, m\\approx5000$)")
    plt.legend(loc=4)
    plt.xlabel("Fraction of global knowledge")
    plt.ylabel("$\\rho$")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.savefig(output_dir + 'corr.png', dpi=150)
    plt.close('all')

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start