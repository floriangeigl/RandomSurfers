from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from plotting import *
import os
from tools.basics import create_folder_structure, find_files
import multiprocessing
import traceback
from utils import check_aperiodic
import multiprocessing as mp
from graph_tool.all import *

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
# matplotlib.rcParams.update({'font.size': 20})


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir/'
base_line = 'adjacency'
out_dir = base_dir + 'plots/'
create_folder_structure(out_dir)

network_files = {i.rsplit('/', 1)[-1][:-3]: i for i in find_files(base_dir, '.gt')}
print network_files
limits_files = find_files(base_dir, '_limits.df')
print limits_files
for lf in limits_files:
    if 'wiki' not in lf:
        continue
    print 'process:', lf
    limits = pd.read_pickle(lf)
    analysed_categories = set(limits.columns)
    print analysed_categories
    ds_name = lf.rsplit('/', 1)[-1].rsplit('_', 1)[0]
    net = load_graph(network_files[ds_name])
    print net
    cat_pmap = net.vp['category']
    cat_to_vertices = defaultdict(set)
    for v in net.vertices():
        c = cat_pmap[v]
        if c in analysed_categories:
            cat_to_vertices[c].add(v)
    analysed_properties = dict()
    for i, j in cat_to_vertices.iteritems():
        print i, len(j)
    print 'calc network metrics'
    analysed_properties['in-degree'] = net.degree_property_map('in')
    analysed_properties['out-degree'] = net.degree_property_map('out')
    print '\tpagerank'
    analysed_properties['pagerank'] = pagerank(net)
    print '\tbetweeness'
    analysed_properties['betweenness'] = betweenness(net)[0]
    print '\teigenvec'
    analysed_properties['eigenvector'] = eigenvector(net)[1]
    print '\tcloseness'
    analysed_properties['closeness'] = closeness(net)
    print '\tclustering'
    analysed_properties['local_clustering'] = local_clustering(net)

    results = dict()
    for l in limits.index:
        print '\tlimit:', l
        property_names = sorted(analysed_properties.keys())
        col_names = list()
        for pn in property_names:
            col_names.append(pn + '_sum')
        #    col_names.append(pn + '_avg')
        #    col_names.append(pn + '_med')
        data = []
        for cat in limits.columns:
            cat_data = [limits.at[l, cat]]
            print '\t\tcat:', cat
            cat_nodes = cat_to_vertices[cat]
            for prop_name in property_names:
                p_map = analysed_properties[prop_name]
                vals = np.array([p_map[v] for v in cat_nodes])
                cat_data.append(vals.sum())
                #cat_data.append(vals.mean())
                #cat_data.append(np.median(vals))
            cat_data = tuple(cat_data)
            data.append(cat_data)
        results[l] = pd.DataFrame(data=data, columns=['strength'] + col_names)
    for l, df in results.iteritems():
        print 'limit:', l
        try:
            pd.scatter_matrix(df, figsize=(16, 16))
            out_fn = out_dir + 'limits/' + ds_name + '_l_' + str(l) + '.png'
            create_folder_structure(out_fn)
            plt.savefig(out_fn, dpi=150)
        except:
            print traceback.format_exc()
        print df
        print '-' * 80

