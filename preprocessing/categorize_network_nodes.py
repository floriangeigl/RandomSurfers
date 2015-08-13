from __future__ import division
import random
from collections import defaultdict
import operator
from tools.gt_tools import load_edge_list
from graph_tool.all import *
import urllib

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
            val, min(2, len(val)))


def basic_cat(g, url_pmap, special_case=None):
    assert isinstance(g, Graph)
    cat_pmap = g.new_vertex_property('object')
    for v in g.vertices():
        url_pmap[v] = urllib.unquote(url_pmap[v].decode('utf8').encode('latin1'))
    if special_case == 'orf':
        def mapper_function(v):
            v_url = url_pmap[v].split('/', 3)[-1]
            if v_url.startswith(('topic/', 'program/')):
                v_url = v_url.split('/', 2)[1]
            elif v_url.startswith('index.php/'):
                v_url = v_url.split('/', 1)[1]
                if v_url.startswith(('topic/', 'program/')):
                    v_url = v_url.split('/', 2)[1]
                else:
                    v_url = v_url.split('/', 1)[0]
            else:
                v_url = v_url.split('/', 2)[0]
            return v_url
    else:
        mapper_function = lambda v: url_pmap[v].split('/', 3)[-1].split('/', 1)[0].split('?', 1)[0]

    for v in g.vertices():
        v_cat = mapper_function(v)
        if v_cat.endswith('.html'):
            v_cat = ''
        elif '?' in v_cat:
            v_cat = '?'
        cat_pmap[v] = v_cat

    # sample_nodes = random.sample(list(g.vertices()), 20)
    # for v in sample_nodes:
    #     print url_pmap[v], '->', cat_pmap[v]
    if special_case is None or special_case == '':
        pass

    elif special_case == 'getdigital':
        for v in g.vertices():
            cat_pmap[v] = cat_pmap[v].rsplit('_', 1)[0]

    return cat_pmap


def main():
    net_name = '/opt/datasets/daserste/daserste'
    special_case = 'daserste'

    #net_name = '/opt/datasets/orf_tvthek/tvthek_orf'
    #special_case = 'orf'


    g = load_edge_list(net_name, vertex_id_dtype='string', directed=True)
    print g
    if 'NodeId' in g.vp.keys():
        g.vp['url'] = g.vp['NodeId'].copy()
    print g.vp.keys()
    url_map = g.vp['url']
    cat_pmap = basic_cat(g, url_map, special_case)
    cat_dist = get_cat_dist(g, cat_pmap, url_map)
    print_norm_dict(cat_dist)
    print len(cat_dist), 'categories'
    if not net_name.endswith('.gt'):
        net_name += '.gt'
    g.vp['category'] = cat_pmap
    g.save(net_name)

if __name__ == '__main__':
    main()

