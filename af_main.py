from __future__ import division
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
import multiprocessing
import datetime
import traceback
from self_sim_entropy import self_sim_entropy
import os
from data_io import *
import utils
from graph_tool.all import *

def main():
    base_outdir = 'output/iknow/'
    basics.create_folder_structure(base_outdir)

    results = list()
    post_fix = '_clicklc'
    # post_fix = '_lc'
    biases = ['adjacency', 'preprocessing/data/af_click_matrix' + post_fix]#, 'deg', 'eigenvector', 'inv_sqrt_deg']
    network_prop_file = base_outdir + 'network_properties.txt'
    if os.path.isfile(network_prop_file):
        os.remove(network_prop_file)

    print 'austria-forum'.center(80, '=')
    name = 'austria_forum' + post_fix
    outdir = base_outdir + name + '/'
    basics.create_folder_structure(outdir)
    fname = 'preprocessing/data/af' + post_fix + '.gt'
    net = load_graph(fname)
    net.gp['type'] = net.new_graph_property('string', 'empiric')
    net.gp['filename'] = net.new_graph_property('string', fname)
    results.append(self_sim_entropy(net, name=name, out_dir=outdir, biases=biases, error_q=None))
    write_network_properties(net, name, network_prop_file)
    # generator.analyse_graph(net, outdir + name, draw_net=False)

    while True:
        try:
            q_elem = error_q.get(timeout=3)
            print 'Error'.center(80, '-')
            print utils.color_string('[' + str(q_elem[0]) + ']')
            print q_elem[-1]
        except:
            break
    results = filter(lambda x: isinstance(x, dict), results)
    gini_dfs = [i['gini'] for i in results]
    gini_dfs = gini_dfs[0].join(gini_dfs[1:])
    print 'gini coefs\n', gini_dfs
    gini_dfs.to_csv(base_outdir + 'gini_coefs.csv')
    gini_to_table(gini_dfs, base_outdir + 'gini_table.txt', digits=2)


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start