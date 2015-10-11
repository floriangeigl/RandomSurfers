from tools.basics import find_files, create_folder_structure
from tools.pd_tools import print_tex_table
import shutil
import pandas as pd
from graph_tool.all import *


def cp_all_files(files, dest):
    for f in files:
        try:
            shutil.copy(f, dest)
        except shutil.Error:
            pass


base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
exp1_dir = base_dir + 'exp1/'
exp2_dir = base_dir + 'exp2/'
exp3_dir = base_dir + 'exp3/'
create_folder_structure(exp1_dir)
create_folder_structure(exp2_dir)
create_folder_structure(exp3_dir)

exp1_files = find_files(base_dir,
                        ('_com_in_deg_lines.pdf', '_com_out_deg_lines.pdf', '_ratio_com_out_deg_in_deg_lines.pdf',
                         'lines_legend.pdf'))
cp_all_files(exp1_files, exp1_dir)

exp2_files = find_files(base_dir,
                        ('_com_in_deg_lines_fac.pdf', '_com_out_deg_lines_fac.pdf',
                         '_ratio_com_out_deg_in_deg_lines_fac.pdf', 'lines_fac_legend.pdf'))
cp_all_files(exp2_files, exp2_dir)

exp3_files = find_files(base_dir, ('_inserted_links.pdf', 'inserted_links_legend.pdf'))
cp_all_files(exp3_files, exp3_dir)

if False:
    table_dir = base_dir + 'tables/'
    create_folder_structure(table_dir)
    net_files = find_files(base_dir, '.gt')
    df = pd.DataFrame()
    for r_idx, g_f in enumerate(net_files):
        print 'analyze:', g_f.rsplit('/', 1)[-1]
        g = load_graph(g_f)
        df.at[r_idx, 'dataset'] = g_f.rsplit('/', 1)[-1].replace('.gt', '').replace('_', ' ')
        df.at[r_idx, 'n'] = g.num_vertices()
        df.at[r_idx, 'm'] = g.num_edges()
    df.sort_values(by='n', inplace=True)
    print df
    print print_tex_table(df, cols=['dataset', 'n', 'm'], mark_min=False, mark_max=False, digits=0)
