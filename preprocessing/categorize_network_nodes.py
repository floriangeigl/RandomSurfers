from __future__ import division
from graph_tool.all import *
import random
from collections import defaultdict
import operator
from tools.gt_tools import load_edge_list

def get_cat_dist(g, cat_pmap, url_map):
    dist = defaultdict(list)
    for v in g.vertices():
        dist[cat_pmap[v]].append(url_map[v])
    return dist


def print_norm_dict(mydict):
    dict_sum = sum(map(len, mydict.values()))
    for key, val in sorted(mydict.iteritems(), key=lambda x: len(x[1]), reverse=True)[:100]:
        num_pages = len(val)
        print key[:20].rjust(20), '->', '%.3f' % (num_pages / dict_sum * 100), '%', '(', num_pages, ')', random.sample(
            val,
                                                                                                                  1)


def basic_cat(g, url_pmap, special_case=None):
    assert isinstance(g, Graph)
    cat_pmap = g.new_vertex_property('object')
    for v in g.vertices():
        v_cat = url_pmap[v].split('/', 3)[-1].split('/', 1)[0]
        if v_cat.endswith('.html'):
            v_cat = ''
        elif '?' in v_cat:
            v_cat = '?'
        cat_pmap[v] = v_cat

    sample_nodes = random.sample(list(g.vertices()), 20)
    for v in sample_nodes:
        print url_pmap[v], '->', cat_pmap[v]
    if special_case is None or special_case == '':
        pass

    elif special_case == 'getdigital':
        for v in g.vertices():
            cat_pmap[v] = cat_pmap[v].rsplit('_', 1)[0]
    return cat_pmap



net_name = '/opt/datasets/daserste/daserste'
g = load_edge_list(net_name, vertex_id_dtype='string', directed=True)
if 'NodeId' in g.vp.keys():
    g.vp['url'] = g.vp['NodeId'].copy()
print g.vp.keys()
url_map = g.vp['url']
cat_pmap = basic_cat(g, url_map, '')
cat_dist = get_cat_dist(g, cat_pmap, url_map)
print_norm_dict(cat_dist)
print len(cat_dist), 'categories'
