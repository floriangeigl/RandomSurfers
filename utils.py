from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from graph_tool.all import *
from scipy.stats import poisson, powerlaw
import operator
import pandas as pd
import numpy as np
import psutil
import os


def graph_gen(self_con, other_con, nodes=100, groups=10):
    corr = lambda x, y: self_con if x == y else other_con
    g, bm = random_graph(nodes, lambda: (powerlaw.rvs(2.4, size=1) * -1 + 1) * 20, directed=False, model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)), vertex_corr=corr)
    # poisson(10).rvs(1)
    return g, bm


def plot_measurements_of_ranking(ranking, measurements_dict, filename=None, logx=True, logy=True):
    df = pd.DataFrame(columns=['ranking'], data=ranking)
    for key, pmap in measurements_dict.iteritems():
        net = pmap.get_graph()
        df[key] = df['ranking'].apply(lambda x: pmap[net.vertex(x)])
    df.drop('ranking', axis=1, inplace=True)
    df.plot(lw=2, alpha=0.8, logx=logx, logy=logy)
    plt.ylabel('ranking value')
    plt.xlabel('ranked vertices')
    outfname = 'output/measurements_of_ranking.png' if filename is None else filename
    plt.savefig(outfname, dpi=150)
    plt.close('all')


def get_ranking(vp_map):
    network = vp_map.get_graph()
    result = map(int, sorted(network.vertices(), key=lambda v: vp_map[v], reverse=True))
    return result


def get_ranking_df(ranking, weights):
    data = [(val, vertex) for val, vertex in zip(weights, ranking)]
    df = pd.DataFrame(columns=['values', 'ranked_vertex'], data=data)
    return df


def shift_data_pos(data, shift_min=True):
    changed_data = False
    data_lower_z = data < 0
    if any(data_lower_z):
        data += data[data_lower_z].min()
        changed_data = True
    data_near_z = np.isclose(data, 0.)
    if any(data_near_z):
        changed_data = True
        if shift_min:
            data += data[data > 0].min()
        else:
            data += np.finfo(float).eps
    return data, changed_data


def gini_coeff(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    n = len(y)
    if n <= 1:
        return 0
    y.sort()
    y_sum = y.sum()
    if np.isclose(y_sum, 0.0):
        return 0.
    gini = 1 - 2 / (n - 1) * (n - sum((i + 1) * yi for i, yi in enumerate(y)) / y_sum)
    #print (y, gini)
    return gini


def get_memory_consumption_in_mb():
    return psutil.Process(os.getpid()).get_memory_info()[0] / float(2 ** 20)


class bcolors:
    prefix = '\33'
    ENDC = prefix + '[0m'
    gen_c = lambda x, ENDC=ENDC, prefix=prefix: ENDC + prefix + '[' + str(x) + 'm'
    HEADER = gen_c(95)
    WARNING = gen_c(93)
    FAIL = gen_c(91)

    BLACK = gen_c('0;30')
    WHITE = gen_c('1;37')

    BLUE = gen_c('0;34')
    GREEN = gen_c('0;32')
    PURPLE = gen_c('0;35')
    RED = gen_c('0;31')
    YELLOW = gen_c('1;33')
    CYAN = gen_c('0;36')

    DARK_GRAY = gen_c('1;30')

    LIGHT_BLUE = gen_c('1;34')
    LIGHT_GREEN = gen_c('1;32')
    LIGHT_CYAN = gen_c('1;36')
    LIGHT_RED = gen_c('1;31')
    LIGHT_PURPLE = gen_c('1;35')
    LIGHT_GRAY = gen_c('0;37')


def color_string(string, type=bcolors.BLUE):
    return type + str(string) + bcolors.ENDC


def get_colors_list():
    return [bcolors.BLUE, bcolors.CYAN, bcolors.GREEN, bcolors.LIGHT_RED, bcolors.PURPLE]


def softmax(w, t=1.0):
    dist = np.array(w, dtype='float')
    dist /= dist.sum()
    dist = np.exp(dist / t)
    dist /= dist.sum()
    return dist


