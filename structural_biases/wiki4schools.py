from __future__ import division

from graph_tool.all import *

import tools.basics as basics
from structural_biases.process_network import process_network

base_outdir = 'output_text_sim/'

print 'wiki4schools'.center(80, '=')
name = 'wiki4schools'
biases = ['adjacency', '/opt/datasets/wikiforschools/graph_with_props_text_sim.bias', 'eigenvector', 'deg',
          'inv_sqrt_deg', 'sigma', 'sigma_sqrt_deg_corrected']
outdir = base_outdir + name + '/'
basics.create_folder_structure(outdir)
net = load_graph('/opt/datasets/wikiforschools/graph_with_props.gt')
print net
net.gp['type'] = net.new_graph_property('string','empiric')
net.gp['filename'] = net.new_graph_property('string', 'wiki4schools')
process_network(net, name=name, out_dir=outdir, biases=biases, error_q=None)
