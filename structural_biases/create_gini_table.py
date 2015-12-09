from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from post_processing.plotting import *
import os
from tools.basics import create_folder_structure, find_files
from tools.pd_tools import print_tex_table
from graph_tool.all import *

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)

name_mapping = dict()
name_mapping['adjacency'] = 'Unbiased'
name_mapping['deg'] = 'Degree Bias Factor'
name_mapping['eigenvector'] = 'Eigenvector Centrality Bias Factor'
name_mapping['inv_sqrt_deg'] = 'Inverse Degree Bias Factor'
name_mapping['sigma'] = 'Sigma Bias Factor'
name_mapping['sigma_sqrt_deg_corrected'] = 'Degree Corrected Sigma Bias Factor'
name_mapping['getdigital'] = 'GD'
name_mapping['karate.edgelist'] = 'Toy Example'
name_mapping['milan_spiele'] = 'MS'
name_mapping['thinkgeek'] = 'TG'
name_mapping['wiki4schools'] = 'WFS'
name_mapping['bar_wiki'] = 'BW'
name_mapping['tvthek_orf'] = 'ORF'
name_mapping['daserste'] = 'DEM'

name_mapping = {key: val.split('Bias')[0].strip() for key, val in name_mapping.iteritems()}
base_dir = '/home/fgeigl/navigability_of_networks/output/wsdm/'
files = filter(lambda x: 'rewired' not in x and 'karate' not in x, find_files(base_dir, 'gini.df'))
print files
all_ginis = pd.DataFrame()
for i in files:
    df = pd.read_pickle(i)
    for c in df.columns:
        all_ginis[name_mapping[c] if c in name_mapping else c] = df[c]
print all_ginis
eigvec = all_ginis.loc['eigenvector'].copy()
all_ginis.drop('eigenvector', inplace=True)
all_ginis.loc['eigenvector'] = eigvec
print all_ginis
all_ginis.index = [name_mapping[i] if i in name_mapping else i for i in all_ginis.index]
all_ginis = all_ginis[['TG', 'GD', 'MS', 'WFS', 'BW', 'ORF', 'DEM']]
print print_tex_table(all_ginis)
