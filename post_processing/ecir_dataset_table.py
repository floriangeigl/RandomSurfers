from tools.basics import find_files, create_folder_structure
from tools.pd_tools import print_tex_table
import pandas as pd
from graph_tool.all import *
import powerlaw
import os


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
table_dir = base_dir + 'tables/'
create_folder_structure(table_dir)
net_files = filter(lambda x: 'bow_tie' not in x, find_files(base_dir, '.gt'))
cached_results_file = ''
cached_results_file = base_dir + 'datasets_table.df'

if cached_results_file and os.path.isfile(cached_results_file):
    df = pd.read_pickle(cached_results_file)
else:

    df = pd.DataFrame()
    for r_idx, g_f in enumerate(net_files):
        g = load_graph(g_f)
        print 'analyze:', g_f.rsplit('/', 1)[-1], 'n:', g.num_vertices(), 'm:', g.num_edges()
        df.at[r_idx, 'dataset'] = g_f.rsplit('/', 1)[-1].replace('.gt', '').replace('_', ' ')
        df.at[r_idx, 'n'] = g.num_vertices()
        df.at[r_idx, 'm'] = g.num_edges()
        print '\tglobal clustering'
        df.at[r_idx, 'c'] = global_clustering(g)[0]
        print '\tpowerlaw'
        deg_dist = g.degree_property_map('total').a
        p_law_res = powerlaw.Fit(deg_dist, discrete=True)
        df.at[r_idx, r'$\alpha$'] = p_law_res.alpha
        df.at[r_idx, 'x-min'] = p_law_res.xmin
        print '\tpseudo diameter'
        df.at[r_idx, 'pd'] = pseudo_diameter(g)[0]

    df.sort_values(by='n', inplace=True)
    df.to_pickle(cached_results_file)
print df
tex_table_str = print_tex_table(df, cols=['dataset', 'n', 'm', 'c', r'$\alpha$', 'x-min', 'pd'], mark_min=False,
                                mark_max=False, digits=0)
print tex_table_str
with open(table_dir + 'datasets.tex', 'w') as f:
    f.write(tex_table_str)
