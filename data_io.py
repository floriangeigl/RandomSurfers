from __future__ import division
from graph_tool.all import *
import cPickle
import traceback
import numpy as np
import random
import powerlaw as fit_powerlaw
import os
from collections import defaultdict

def read_edge_list(filename, encoder=None):
    store_fname = filename + '.gt'
    if not os.path.isfile(store_fname):
        print 'read edgelist:', filename
        g = Graph(directed=True)
        get_v = defaultdict(lambda: g.add_vertex())
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('#'):
                    try:
                        s_link, t_link = line.split('\t')
                        if encoder is not None:
                            s_link, t_link = encoder(s_link), encoder(t_link)
                        s, t = get_v[s_link], get_v[t_link]
                        g.add_edge(s, t)
                    except ValueError:
                        print '-' * 80
                        print line
                        print '-' * 80
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


def try_dump(data, filename):
    try:
        data.dump(filename)
        return True
    except (SystemError, AttributeError):
        try:
            with open(filename, 'wb') as f:
                cPickle.dump(data, f)
        except:
            pass
        return False


def try_load(filename):
    try:
        data = np.load(filename)
    except IOError:
        try:
            with open(filename, 'rb') as f:
                data = cPickle.load(f)
        except IOError:
            raise IOError
        except:
            print traceback.format_exc()
            raise IOError
    return data