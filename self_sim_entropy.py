from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt

import sys
from tools.gt_tools import SBMGenerator, load_edge_list
import numpy as np
from linalg import *
import scipy.stats as stats
import scipy
import pandas as pd
import operator
import random
from collections import defaultdict
import multiprocessing
import traceback
np.set_printoptions(linewidth=225)
import seaborn
from timeit import Timer
np.set_printoptions(precision=2)
import copy

generator = SBMGenerator()
granularity = 10
num_samples = 10
outdir = 'output/'

test = True


def self_sim_entropy(network, name, out_dir):
    A = adjacency(network)
    A_eigv = eigenvector(network)[-1].a
    weights = dict()
    weights['eigenvector'] = A_eigv


if test:
    outdir += 'tests/'
    print 'sbm'.center(80, '=')
    name = 'sbm_n10_m30'
    net = generator.gen_stock_blockmodel(num_nodes=100, blocks=3, num_links=200)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net, name=name, out_dir=outdir)
    print 'price network'.center(80, '=')
    name = 'price_net_n50_m1_g2_1'
    net = price_network(30, m=1, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net, name=name, out_dir=outdir)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n50'
    net = complete_graph(30)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net, name=name, out_dir=outdir)
    print 'quick tests done'.center(80, '=')
else:
    num_links = 2000
    num_nodes = 1000
    num_blocks = 10
    print 'sbm'.center(80, '=')
    name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net, name=name, out_dir=outdir)
    print 'sbm'.center(80, '=')
    name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.7)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net,name=name, out_dir=outdir)
    print 'powerlaw'.center(80, '=')
    name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net,  name=name, out_dir=outdir)
    print 'price network'.center(80, '=')
    name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
    net = price_network(num_nodes, m=2, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net,  name=name, out_dir=outdir)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n' + str(num_nodes)
    net = complete_graph(num_nodes)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net, name=name, out_dir=outdir)
    print 'karate'.center(80, '=')
    name = 'karate'
    net = load_edge_list('/opt/datasets/karate/karate.edgelist')
    generator.analyse_graph(net, outdir + name, draw_net=False)
    self_sim_entropy(net,  name=name, out_dir=outdir)
    exit()
    print 'wiki4schools'.center(80, '=')
    net = load_edge_list('/opt/datasets/wikiforschools/graph')
    self_sim_entropy(net,  name='wiki4schools')
    print 'facebook'.center(80, '=')
    net = load_edge_list('/opt/datasets/facebook/facebook')
    self_sim_entropy(net, name='facebook')
    print 'youtube'.center(80, '=')
    net = load_edge_list('/opt/datasets/youtube/youtube')
    self_sim_entropy(net,  name='youtube')
    print 'dblp'.center(80, '=')
    net = load_edge_list('/opt/datasets/dblp/dblp')
    self_sim_entropy(net,  name='dblp')