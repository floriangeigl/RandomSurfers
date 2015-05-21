import numpy as np
from graph_tool.all import *
import bs4
import os
from collections import defaultdict
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import traceback
import cPickle

def try_dump(data, filename):
    try:
        data.dump(filename)
        return True
    except (SystemError, AttributeError):
        try:
            with open(filename, 'wb') as f:
                cPickle.dump(data, f)
        except:
            print traceback.format_exc()
        return False


g = Graph(directed=True)
vertex_mapper = defaultdict(lambda: g.add_vertex())
pmap_article_name = g.new_vertex_property('object')
print 'read article mapping'
with open('/opt/datasets/wikiforschools/artname_artid', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            article_name, article_id = line.strip().split()
            article_id = int(article_id)
            v = vertex_mapper[article_id]
            pmap_article_name[v] = article_name

print 'read edge-list'
with open('/opt/datasets/wikiforschools/graph', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            line = map(int, line.strip().split())
            src, dest = map(lambda x: vertex_mapper[x], line)
            g.add_edge(src, dest)
g.vp['article-name'] = pmap_article_name

print 'map html files'
article_name_to_html = dict()
for dirpath, dnames, fnames in os.walk('/opt/datasets/wikiforschools/html_files/'):
    for f in fnames:
        article_name_to_html[f.replace('.htm', '').replace('.html', '')] = os.path.join(dirpath, f)

pmap_html_text = g.new_vertex_property('object')
filter = g.new_vertex_property('bool')
print 'store html text'
for v in g.vertices():
    article_name = pmap_article_name[v]
    try:
        html_file = article_name_to_html[article_name]
        with open(html_file, 'r') as f:
            pmap_html_text[v] = f.read()
        filter[v] = True
    except:
        if not (article_name.endswith('_A') or article_name.endswith('_A') or article_name.endswith('Directdebit')):
            print 'ERROR:', article_name
        pmap_html_text[v] = ''
        filter[v] = False
g.vp['html'] = pmap_html_text
print 'number vertices orig:', g.num_vertices()
g.set_vertex_filter(filter)
g.purge_vertices()
l = label_largest_component(g)
g.set_vertex_filter(l)
g.purge_vertices()
print 'number vertices cleanup and lc:', g.num_vertices()

print 'create plain text'
pmap_plain_text = g.new_vertex_property('object')
for v in g.vertices():
    pmap_plain_text[v] = bs4.BeautifulSoup(pmap_html_text[v]).get_text()
g.vp['plain-text'] = pmap_plain_text

print 'calc similarity'
swords = stopwords.words('english')
plain_texts = [pmap_plain_text[v] for v in g.vertices()]
tfidf = TfidfVectorizer(stop_words=swords).fit_transform(plain_texts)
similarity = tfidf * tfidf.T

print 'create sparse similarity'
A = adjacency(g)
A.data = np.array([1.] * len(A.data), dtype=np.float64)
sparse_sim = A.multiply(similarity)
print sparse_sim
try_dump(sparse_sim, '/opt/datasets/wikiforschools/graph_with_props_text_sim.bias')

print 'save network'
g.save('/opt/datasets/wikiforschools/graph_with_props.gt')
print g



