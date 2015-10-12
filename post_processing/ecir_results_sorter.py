from tools.basics import find_files, create_folder_structure
import shutil


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

