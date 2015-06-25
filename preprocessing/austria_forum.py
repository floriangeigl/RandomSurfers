from __future__ import division
from tools.gt_tools import net_from_adj
from graph_tool.all import *
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import tables as tb
from collections import defaultdict
import os
import difflib
import urllib
import multiprocessing as mp
import datetime
from data_io import *
import pandas as pd
import chardet

def convert_url(url):
    url = url.strip().decode('utf8')
    try:
        try:
            url = url.encode('latin1')
        except UnicodeDecodeError:
            pass
        url = urllib.unquote(url)
        while len(url) > 1 and url.endswith(('/', '=')):
            url = url[:-1]
        return url.replace(' ', '_').replace('//', '/')
    except:
        print traceback.format_exc()
        print 'FAILED:', url, type(url)

def read_and_map_hdf5(filename, mapping, shape=None):
    print 'read and map hdf5'
    with tb.open_file(filename, 'r') as h5:
        data = list()
        row_idx = list()
        col_idx = list()
        unmapped = 0
        unmapped_clicks = 0
        unmapped_clicks_dict = defaultdict(int)
        orig_clicks = np.array(h5.root.data).sum()
        print 'clicks in hdf5 file:', orig_clicks
        unmapped_cells = set()
        for d, r, c in zip(h5.root.data, map(int,h5.root.row_indices), map(int,h5.root.column_indices)):
            try:
                mr = mapping[r]
                mc = mapping[c]
                data.append(d)
                row_idx.append(mr)
                col_idx.append(mc)
            except KeyError:
                unmapped += 1
                unmapped_clicks += d
                if r not in mapping:
                    unmapped_cells.add(r)
                    unmapped_clicks_dict[r] += d
                if c not in mapping:
                    unmapped_cells.add(c)
                    unmapped_clicks_dict[c] += d
        shape = (h5.root.shape_dimensions[0, 0], h5.root.shape_dimensions[0, 1]) if shape is None else shape
        if unmapped:
            print 'unmapped cells:', len(unmapped_cells) / shape[0] * 100, '%'
            print 'unmapped links:', unmapped / len(h5.root.data) * 100, '%'
            print 'unmapped clicks:', unmapped_clicks / np.array(h5.root.data).sum() * 100, '%'
            print 'unmapped cells with most influence:', sorted(unmapped_clicks_dict.iteritems(), key=lambda x: x[-1],
                                                                reverse=True)[:10]
    mat = csr_matrix((data, (row_idx, col_idx)), shape=shape)
    mat.eliminate_zeros()
    return mat

def create_mapping(user_map, net_map, find_best_match=False):
    transf_map = dict()
    print 'create user to net mapping'
    unmapped = 0
    unmapped_urls = set()
    tmp = list(user_map.iteritems())
    random.shuffle(tmp)
    for url, url_id in tmp:
        try:
            net_id = net_map[url]
            for i in map(int, url_id):
                transf_map[i] = int(net_id)
        except KeyError:
            if find_best_match:
                print 'can not map:', url, type(url)
                print ' best match:',
                pool = mp.Pool(processes=15)
                res = []
                call_back = lambda x: res.append(x)
                for i in net_map.keys():
                    pool.apply_async(string_sim, args=(i, url,), callback=call_back)
                pool.close()
                pool.join()
                best, best_url = max(res, key=lambda x: x[0])
                print best_url, 'val:', best, type(best_url)
                print '==', url == best_url
                for i, j in zip(url, best_url):
                    if i != j:
                        print i, '==', j, ':', i == j
                        break
                print chardet.detect(url)
                print chardet.detect(best_url)
            unmapped += 1
            unmapped_urls.add(url)
    if unmapped:
        print 'unmapped urls:\n\t', '\n\t'.join(random.sample(unmapped_urls, min(15, len(unmapped_urls))))
        print '\t', unmapped / len(user_map) * 100, '%'
        with open('unmapped_urls.txt', 'w') as f:
            for i in sorted(unmapped_urls):
                f.write(i + '\n')
    print 'done'
    return transf_map

def string_sim(s1, s2):
    return difflib.SequenceMatcher(None, a=s1, b=s2).ratio(), s1

def read_tmat_map(filename):
    print 'read user mat mapping'
    df = pd.read_pickle(filename)
    print df.columns
    df['ID'] = df['ID'].astype('int')
    # df['Page'] = df['Page'].apply(convert_url)
    print df.head()
    result = defaultdict(set)
    for i, j in zip(df['Page'], df['ID']):
        result[i].add(j)
    return result, df[['Page', 'Visits']]

def filter_sparse_matrix(mat, mapping):
    row_idx, col_idx = mat.nonzero()
    n_row_idx, n_col_idx, n_data = list(), list(), list()
    for r, c, d in zip(row_idx, col_idx, mat.data):
        try:
            r = mapping[r]
            c = mapping[c]
            n_row_idx.append(r)
            n_col_idx.append(c)
            n_data.append(d)
        except KeyError:
            pass
    shape = len(mapping)
    mat = csr_matrix((n_data, (n_row_idx, n_col_idx)), shape=(shape, shape))
    mat.eliminate_zeros()
    return mat

def store(adj, trans, net, post_fix='', draw=True):
    print 'network vertices:', net.num_vertices()
    print 'store click matrix'
    trans.eliminate_zeros()
    try_dump(trans, 'data/af_click_matrix' + post_fix)
    print 'store adj matrix'
    adj.eliminate_zeros()
    try_dump(adj, 'data/af_adj_matrix' + post_fix)
    print 'store network'
    net.save('data/af' + post_fix + '.gt')
    if draw:
        print 'calc layout'
        pos = sfdp_layout(net, max_iter=10)
        print 'draw network'
        graph_draw(net, pos=pos, output='data/af' + post_fix + '.png', output_size=(1200, 1200),
                   bg_color=[255, 255, 255, 0])

def main():
    af_f = 'data/austria_forum_org_cleaned.txt'
    user_tmat = 'data/transition_matrix.h5'
    user_tmat_map = 'data/id_name_mapping_pickled'
    # view_counts_f = 'data/page_count_df_pickled'
    insert_tele = False

    net, net_map = read_edge_list(af_f, encoder=lambda x: convert_url(x.replace('http://austria-forum.org', '')))
    net_map = {i: int(j) for i, j in net_map.iteritems()}
    remove_self_loops(net)
    user_map, view_counts_df = read_tmat_map(user_tmat_map)
    user_to_net = create_mapping(user_map, net_map)
    user_mat = read_and_map_hdf5(user_tmat, user_to_net, shape=(net.num_vertices(), net.num_vertices()))
    net.vp['strong_lcc'] = label_largest_component(net, directed=True)
    click_teleportations = net.new_edge_property('float')
    click_loops = net.new_edge_property('float')
    click_transitions = net.new_edge_property('float')
    clicked_nodes = net.new_vertex_property('bool')
    src_idx, target_idx = user_mat.nonzero()
    for s, t, d in zip(src_idx, target_idx, map(int, user_mat.data)):
        s, t = net.vertex(s), net.vertex(t)
        clicked_nodes[s] = True
        clicked_nodes[t] = True
        e = net.edge(net.vertex(s), net.vertex(t))
        if e is None and insert_tele:
            e = net.add_edge(s, t)
            if s != t:
                click_teleportations[e] = d
            else:
                click_loops[e] = d
        elif e is not None:
            click_transitions[e] = d
    net.ep['click_teleportations'] = click_teleportations
    net.ep['click_loops'] = click_loops
    net.ep['click_transitions'] = click_transitions
    net.vp['clicked_nodes'] = clicked_nodes

    view_counts = net.new_vertex_property('float')
    # view_counts_df = pd.read_pickle(view_counts_f)
    # view_counts_df.drop('ID', inplace=True, axis=1)
    # view_counts_df['Page'] = view_counts_df['Page'].apply(convert_url)
    view_counts_df['Page'] = view_counts_df['Page'].apply(lambda x: net.vertex(net_map[x]) if x in net_map else '')
    for v, views in zip(view_counts_df['Page'], view_counts_df['Visits']):
        if isinstance(v, Vertex):
            view_counts[v] = views
    net.vp['view_counts'] = view_counts
    print 'unfiltered transitions:', click_transitions.a.sum()
    print 'teleportations in click-data:', click_teleportations.a.sum() / click_transitions.a.sum() * 100, '%'
    print 'self-loops in click-data:', click_loops.a.sum() / click_transitions.a.sum() * 100, '%'
    print 'largest strongly connected component:', net.vp['strong_lcc'].a.sum() / net.num_vertices() * 100, '%'
    print 'view counts available for vertices:', (np.array(view_counts.a) > 0).sum() / net.num_vertices() * 100, '%'
    print 'clicked nodes:', clicked_nodes.a.sum() / net.num_vertices() * 100, '%'
    print 'vprops:', net.vp.keys()
    print 'eprops:', net.ep.keys()
    net.save('data/af.gt')


if __name__ == '__main__':
    print 'start:', datetime.datetime.now().replace(microsecond=0)
    main()
    print 'ALL DONE', datetime.datetime.now().replace(microsecond=0)
