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
from graph_tool.all import *

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)

base_dir = '/home/fgeigl/navigability_of_networks/output/wsdm/'
network_files = filter(lambda x: 'rewired' not in x, find_files(base_dir, '.gt'))
print 'name || assort-coef || variance'
results = list()
for fn in network_files:
    g = load_graph(fn)
    name = fn.rsplit('/', 1)[-1].replace('.gt', '')
    res = list(scalar_assortativity(g, 'total'))
    print name, 'scalar assort:', res
    res = [name] + res
    results.append(res)
print 'sorted---------'
for name, assort, var, in sorted(results, key=lambda x: x[1]):
    print name[:20].center(20), str(assort)[:6], str(var)[:5]
