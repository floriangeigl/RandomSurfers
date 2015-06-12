from __future__ import division
from graph_tool.all import *
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import tables as tb
import numpy as np
from collections import defaultdict
import os
import difflib
import urllib
import multiprocessing as mp
import datetime
import traceback

def convert_url(url):
    try:
        return urllib.unquote(url.strip().decode('utf8').encode('latin1').decode('utf8')).encode('utf8')
        # return urllib.unquote(urllib.quote(url.strip(), ':/%')).encode('utf8').decode('utf8').lower()
    except:
        print traceback.format_exc()
        print 'FAILED:', url, type(url)
        exit()


def read_and_map_hdf5(filename, mapping, shape=None):
    print 'read and map hdf5'
    with tb.open_file(filename, 'r') as h5:
        data = list()
        row_idx = list()
        col_idx = list()
        unmapped = 0
        unmapped_clicks = 0
        orig_clicks = np.array(h5.root.data).sum()
        print 'clicks in hdf5 file:', orig_clicks
        for d, r, c in zip(h5.root.data, h5.root.row_indices, h5.root.column_indices):
            try:
                r = mapping[r]
                c = mapping[c]
                data.append(d)
                row_idx.append(r)
                col_idx.append(c)
            except KeyError:
                unmapped += 1
                unmapped_clicks += d
        shape = (h5.root.shape_dimensions[0, 0], h5.root.shape_dimensions[0, 1]) if shape is None else shape
        if unmapped:
            print 'unmapped cells:', unmapped, unmapped / len(h5.root.data) * 100, '%'

            print 'unmapped clicks:', unmapped_clicks, unmapped_clicks / np.array(h5.root.data).sum() * 100, '%'
    return csr_matrix((data, (row_idx, col_idx)), shape=shape)


def create_mapping(user_map, net_map, find_best_match=True):
    net_map = {i.replace('http://austria-forum.org', ''): int(j) for i, j in net_map.iteritems()}
    transf_map = dict()
    print 'create user to net mapping'
    unmapped = 0
    unmapped_urls = set()
    for url, url_id in user_map.iteritems():
        try:
            net_id = net_map[url]
            transf_map[url_id] = net_id
        except KeyError:
            print 'can not map:', url
            if find_best_match:
                print ' best match:',
                pool = mp.Pool(processes=15)
                res = []
                call_back = lambda x: res.append(x)
                for i in net_map.keys():
                    pool.apply_async(string_sim, args=(i, url,), callback=call_back)
                pool.close()
                pool.join()
                best, best_url = max(res, key=lambda x: x[0])
                print best_url, 'val:', best
            unmapped += 1
            unmapped_urls.add(url)
    if unmapped:
        print 'unmapped urls:', unmapped
        print unmapped / len(user_map) * 100, '%'
    print 'done'
    with open('unmapped_urls.txt', 'w') as f:
        for i in unmapped_urls:
            f.write(i + '\n')
    return transf_map


def string_sim(s1, s2):
    return difflib.SequenceMatcher(None, a=s1, b=s2).ratio(), s1


def read_edge_list(filename):
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
                        s, t = get_v[convert_url(s_link)], get_v[convert_url(t_link)]
                        g.add_edge(s, t)
                    except ValueError:
                        print line
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
            print 'failed loading. re-create graph'
            os.remove(filename + '.gt')
            return read_edge_list(filename)
        if 'mtime' in g.gp.keys():
            if g.gp['mtime'] != os.path.getmtime(filename):
                print 'modified edge-list. re-create graph'
                os.remove(filename + '.gt')
                return read_edge_list(filename)
        get_v = g.gp['vgen']
        print 'loaded af-network:', g.num_vertices(), 'vertices', g.num_edges(), 'edges'
    return g, get_v


def read_tmat_map(filename):
    map = dict()
    visits = dict()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                line = line.split('\t')
                map[convert_url(line[1])] = int(line[0])
                visits[line[1]] = int(line[2])
    return map, visits


def main():
    af_f = 'data/austria_forum_org_cleaned.txt'
    user_tmat = 'data/transition_matrix.h5'
    user_tmat_map = 'data/mapping.csv'
    net, net_map = read_edge_list(af_f)
    user_map, visits = read_tmat_map(user_tmat_map)
    user_to_net = create_mapping(user_map, net_map)
    user_mat = read_and_map_hdf5(user_tmat, user_to_net, shape=(net.num_vertices(), net.num_vertices()))
    adj_mat = adjacency(net)
    print user_mat.shape
    print adj_mat.shape
    print 'user clicks:', user_mat.sum()
    ones_adj_mat = adj_mat.copy()
    ones_adj_mat.data = np.array([1] * len(ones_adj_mat.data))
    trans_mat = user_mat.multiply(ones_adj_mat)
    print 'adj links:', ones_adj_mat.sum()
    print 'possible clicks:', trans_mat.sum()


if __name__ == '__main__':
    print 'start:', datetime.datetime.now()
    main()
    print 'ALL DONE', datetime.datetime.now()
