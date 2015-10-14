from tools.basics import find_files, create_folder_structure
from tools.pd_tools import print_tex_table
import pandas as pd
import numpy as np
from graph_tool.all import *
import powerlaw
import os


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
table_dir = base_dir + 'tables/'
create_folder_structure(table_dir)
net_files = filter(lambda x: 'bow_tie' not in x, find_files(base_dir, '.gt'))
cached_results_file = ''
cached_results_file = base_dir + 'datasets_table.df'
sample_sizes = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]


if cached_results_file and os.path.isfile(cached_results_file) and False:
    df = pd.read_pickle(cached_results_file)
else:

    df = pd.DataFrame()
    for r_idx, g_f in enumerate(net_files):
        g = load_graph(g_f)
        lcc = label_largest_component(g).a
        assert np.all(lcc == 1)
        print 'analyze:', g_f.rsplit('/', 1)[-1]
        print g
        df.at[r_idx, 'dataset'] = g_f.rsplit('/', 1)[-1].replace('.gt', '').replace('_', ' ')
        df.at[r_idx, 'n'] = g.num_vertices()
        df.at[r_idx, 'm'] = g.num_edges()
        print '\tpowerlaw'
        deg_dist = g.degree_property_map('total').a
        p_law_res = powerlaw.Fit(deg_dist, discrete=True)
        df.at[r_idx, r'$\alpha$'] = p_law_res.alpha
        df.at[r_idx, 'x-min'] = p_law_res.xmin
        print '\tpseudo diameter'
        df.at[r_idx, 'pd'] = pseudo_diameter(g)[0]
        if True:
            print '\tmake graph undirected'
            g.set_directed(False)
            print g
        print '\tglobal clustering'
        df.at[r_idx, 'c'] = global_clustering(g)[0]

        for sample_s in sample_sizes:
            df.at[r_idx, 'ss ' + ('%.3f' % sample_s).rstrip('0')] = g.num_vertices() * sample_s

    df.sort_values(by='n', inplace=True)
    df.to_pickle(cached_results_file)
print df
sample_sizes_cols = filter(lambda x: x.startswith('ss '), df.columns)
sample_sizes_cols = [sample_sizes_cols[0]] + [sample_sizes_cols[-1]]
tex_table_str = print_tex_table(df, cols=['dataset', 'n', 'm', 'c', r'$\alpha$', 'x-min', 'pd'] + sample_sizes_cols,
                                mark_min=False,
                                mark_max=False, digits=[0, 0, 0, 5, 3, 0, 3] + ([0] * len(sample_sizes_cols)),
                                trim_zero_digits=True,
                                colors=[(20, 'blue'), (30, 'blue'), (40, 'blue')])
print tex_table_str
with open(table_dir + 'datasets.tex', 'w') as f:
    f.write(tex_table_str)
