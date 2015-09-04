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


def calc_cor(limits_filename, network_files, out_dir):
    try:
        #if 'wiki' not in limits_filename:
        #    return
        print 'process:', limits_filename
        limits = pd.read_pickle(limits_filename)
        analysed_categories = set(limits.columns)
        print analysed_categories
        ds_name = limits_filename.rsplit('/', 1)[-1].rsplit('_', 1)[0]
        net_filename = network_files[ds_name]
        print 'load:', net_filename
        net = load_graph(net_filename)
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
        net_dirty = False
        analysed_properties['in-degree'] = net.degree_property_map('in')
        analysed_properties['out-degree'] = net.degree_property_map('out')
        print '\tpagerank'
        try:
            analysed_properties['pagerank'] = net.vp['pagerank']
        except KeyError:
            net_dirty = True
            net.vp['pagerank'] = pagerank(net)
            analysed_properties['pagerank'] = net.vp['pagerank']
        print '\tbetweeness'
        try:
            analysed_properties['betweenness'] = net.vp['betweenness']
        except KeyError:
            net_dirty = True
            net.vp['betweenness'] = betweenness(net)[0]
            analysed_properties['betweenness'] = net.vp['betweenness']
        print '\teigenvec'
        try:
            analysed_properties['eigenvector'] = net.vp['eigenvector']
        except KeyError:
            net_dirty = True
            net.vp['eigenvector'] = eigenvector(net)[1]
            analysed_properties['eigenvector'] = net.vp['eigenvector']
        print '\tcloseness'
        try:
            analysed_properties['closeness'] = net.vp['closeness']
        except KeyError:
            net_dirty = True
            net.vp['closeness'] = closeness(net)
            analysed_properties['closeness'] = net.vp['closeness']
        print '\tclustering'
        try:
            analysed_properties['local_clustering'] = net.vp['local_clustering']
        except KeyError:
            net_dirty = True
            net.vp['local_clustering'] = local_clustering(net)
            analysed_properties['local_clustering'] = net.vp['local_clustering']
        if net_dirty:
            print 'save modified network file:', net_filename
            net.save(net_filename)

        results = dict()
        for l in limits.index:
            print '\tlimit:', l
            property_names = sorted(analysed_properties.keys())
            col_names = list()
            for pn in property_names:
                col_names.append(pn + '_sum')
            #    col_names.append(pn + '_avg')
            #    col_names.append(pn + '_med')
            col_names.append('internal_e')
            col_names.append('in_e')
            col_names.append('out_e')
            col_names.append('#nodes')
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
                internal_links = 0
                external_in = 0
                external_out = 0
                for v in map(lambda x: net.vertex(x), cat_nodes):
                    for out_v in v.out_neighbours():
                        if out_v in cat_nodes:
                            internal_links += 1
                        else:
                            external_out += 1
                    external_in += len(list(filter(lambda in_v: in_v not in cat_nodes, v.in_neighbours())))
                cat_data.append(internal_links)
                cat_data.append(external_in)
                cat_data.append(external_out)
                cat_data.append(len(cat_nodes))

                cat_data = tuple(cat_data)
                data.append(cat_data)
            results[l] = pd.DataFrame(data=data, columns=['strength'] + col_names)
        out_fn = None
        for l, df in results.iteritems():
            print 'limit:', l
            try:
                ax_array = pd.scatter_matrix(df, alpha=0.5, lw=0, figsize=(24, 24))
                # fig = plt.figure(figsize=(24, 24 / len(ax_array)))
                for i in ax_array[1:]:
                    for j in i:
                        plt.delaxes(j)
                cor_array = np.array(df.corr(method='pearson'))
                cor_array = cor_array[0]
                ax_array = ax_array[0]

                center_ax_idx = int(len(ax_array) / 2)
                for idx, (cor, ax, x_label) in enumerate(zip(cor_array, ax_array, df.columns)):
                    ax_title = ''
                    if idx == center_ax_idx:
                        ax_title += ds_name + ' Limit:' + str(l) + '\n'
                    ax_title += '$\\rho=$' + '%.2f' % cor
                    ax.set_title(ax_title)
                    ax.xaxis.set_visible(True)
                    ax.set_xlabel(x_label)
                    #for ax in ax_row:
                        #ax.set_xscale('log')
                        # ax.set_yscale('log')
                out_fn = out_dir + 'limits/' + ds_name + '_l_' + str(l) + '.png'
                create_folder_structure(out_fn)
                plt.savefig(out_fn, dpi=150)
                os.system('convert ' + out_fn + ' -trim ' + out_fn)
            except:
                print traceback.format_exc()
            print df
            print '-' * 80
        out_fn_base = out_fn.rsplit('/', 1)
        out_fn_base = out_fn_base[0] + '/' + out_fn_base[1].split('_', 1)[0] + '*'
        os.system('convert ' + out_fn_base + ' -append ' + out_fn_base[:-1] + '.png')
    except:
        print 'ERROR:', traceback.format_exc()
        raise


network_files = {i.rsplit('/', 1)[-1][:-3]: i for i in find_files(base_dir, '.gt')}
print network_files
limits_files = find_files(base_dir, '_limits.df')
print limits_files
worker_pool = mp.Pool(processes=3)
for lf in limits_files:
    worker_pool.apply_async(calc_cor, args=(lf, network_files, out_dir))
worker_pool.close()
worker_pool.join()
