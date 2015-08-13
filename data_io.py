from __future__ import division
from graph_tool.all import *
import cPickle
import traceback
import numpy as np
import random
import powerlaw as fit_powerlaw
import os
from collections import defaultdict
import h5py
import scipy
import sys
from scipy.sparse import csr_matrix
from tools.gt_tools import load_edge_list

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
        net = load_edge_list('/opt/datasets/karate/karate.edgelist', directed=directed, sep=None, vertex_id_dtype='int')
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
        net = load_edge_list(name, directed=directed, vertex_id_dtype='string')
        net.gp['filename'] = net.new_graph_property('string', name)
        net.gp['type'] = net.new_graph_property('string', 'empiric')

    l = label_largest_component(net, directed=directed)
    net.set_vertex_filter(l)
    net.purge_vertices()
    remove_self_loops(net)
    net.clear_filters()
    return net

def read_edge_list(filename, encoder=None):
    store_fname = filename + '.gt'
    if not os.path.isfile(store_fname):
        print 'read edgelist:', filename
        g = Graph(directed=True)
        get_v = defaultdict(g.add_vertex)
        edge_list = []
        edge_list_a = edge_list.append
        with open(filename, 'r') as f:
            for line in map(lambda x: x.strip().split('\t'), filter(lambda x: not x.startswith('#'), f)):
                try:
                    if encoder is not None:
                        s_link, t_link = map(encoder, line)
                    else:
                        s_link, t_link = line
                    edge_list_a((int(get_v[s_link]), int(get_v[t_link])))
                except ValueError:
                    print '-' * 80
                    print line
                    print '-' * 80
        print 'insert', len(edge_list), 'edges'
        g.add_edge_list(edge_list)
        url_pmap = g.new_vertex_property('string')
        for link, v in get_v.iteritems():
            url_pmap[v] = link
        g.vp['url'] = url_pmap
        g.gp['vgen'] = g.new_graph_property('object', {i: int(j) for i, j in get_v.iteritems()})
        g.gp['mtime'] = g.new_graph_property('object', os.path.getmtime(filename))
        print 'created af-network:', g.num_vertices(), 'vertices,', g.num_edges(), 'edges'
        g.save(store_fname)
    else:
        print 'load graph:', store_fname
        try:
            g = load_graph(store_fname)
        except:
            print 'failed loading. recreate graph'
            os.remove(store_fname)
            return read_edge_list(filename)
        if 'mtime' in g.gp.keys():
            if g.gp['mtime'] != os.path.getmtime(filename):
                print 'modified edge-list. recreate graph'
                os.remove(filename + '.gt')
                return read_edge_list(filename)
        get_v = g.gp['vgen']
        print 'loaded network:', g.num_vertices(), 'vertices', g.num_edges(), 'edges'
    return g, get_v

def gini_to_table(df, out_fname, digits=2,columns=None):
    if columns is None:
        col_names = list()
    else:
        col_names = columns

    print col_names
    digits = str(digits)
    with open(out_fname,'w') as f:
        f.write('\t'.join(col_names) + '\n')
        for name, data in df[col_names].iterrows():
            name = name.replace('_', ' ')
            name = name.replace('inverse', 'inv.').replace('corrected', 'cor.')
            # name = name.split()
            # name = ' '.join([i[:3] + '.' for i in name])
            line_string = name + ' & $' + '$ & $'.join(
                map(lambda x: str("{:2." + str(digits) + "f}").format(x), list(data))) + '$ \\\\'
            print line_string
            line_string += '\n'
            f.write(line_string)

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


def try_dump(data, filename, mask=None):
    if mask is not None and (isinstance(data, np.matrix) or (
                        isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[0] > 1 and data.shape[
                1] > 1) or scipy.sparse.issparse(data)):
        print 'dump data. mask data...'
        mask = mask.astype('bool')
        data = csr_matrix(data)
        data.eliminate_zeros()
        data = data.multiply(mask)
        data.eliminate_zeros()
    try:
        data.dump(filename)
        return True
    except (SystemError, AttributeError):
        try:
            with open(filename, 'wb') as f:
                cPickle.dump(data, f)
        except:
            print traceback.format_exc()
            print 'try hdf5'
            try:
                h5f = h5py.File(filename, 'w')
                h5f.create_dataset('bias', data=data)
                h5f.close()
            except:
                print 'dump', filename, 'failed'
                print traceback.format_exc()
        return False


def try_load(filename):
    if not os.path.isfile(filename):
        print 'need to calc bias.'
        sys.stdout.flush()
        raise IOError
    try:
        data = np.load(filename)
    except IOError:
        try:
            print 'nbload failed, cpickle'
            sys.stdout.flush()
            with open(filename, 'rb') as f:
                data = cPickle.load(f)
        except IOError:
            try:
                print 'cpickle failed. h5py'
                sys.stdout.flush()
                h5f = h5py.File(filename, 'r')
                data = h5f['bias'][:]
                h5f.close()
            except IOError:
                print 'load', filename, 'FAILED'
                sys.stdout.flush()
                raise IOError
        except:
            print traceback.format_exc()
            print 'load', filename, 'FAILED'
            raise IOError
    return data
