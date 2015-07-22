from __future__ import division

from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from graph_tool.all import *
import tools.basics as basics
import datetime
from data_io import *
import utils
import pandas as pd
from network_matrix_tools import stationary_dist, calc_entropy_and_stat_dist, entropy_rate
from scipy.sparse import csr_matrix, dia_matrix

elements = 20
output_folder = 'output/ideas/'
ext = '.pdf'
matplotlib.rcParams.update({'font.size': 25})
legend_loc = 'best'
legend_font_size = 15
basics.create_folder_structure(output_folder)
# 80 / 20 rule
# we want to highlight 20 % of the products
twenty_perc = np.round(elements * 0.2) - 0.5
df = pd.DataFrame()
df['orig'] = np.exp(np.random.random(elements))
df['orig'] /= df['orig'].sum()
df['orig'] = sorted(df['orig'], reverse=True)
df.plot(kind='bar')
plt.axvline(twenty_perc, lw=2, c='black', linestyle='--')
plt.annotate('20%', xy=(twenty_perc + 0.1, max(df.max())))
plt.xlabel('node-id')
plt.ylabel('visit probability')
plt.ylim([0, max(df.max()) * 1.1])
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'eighty_twenty_orig' + ext)
plt.close('all')

df['mod'] = df['orig'].copy()
fac = 0.8 / df['mod'].loc[:int(twenty_perc)].sum()
df['mod'].loc[:int(twenty_perc)] = df['mod'].loc[:int(twenty_perc)] * fac
fac = 0.2 / df['mod'].loc[int(twenty_perc) + 1:].sum()
df['mod'].loc[int(twenty_perc) + 1:] = df['mod'].loc[int(twenty_perc) + 1:] * fac
df['mod'] /= df['mod'].sum()
# print df
df.plot(kind='bar')
plt.axvline(twenty_perc, lw=2, c='black', linestyle='--')
plt.annotate('20%', xy=(twenty_perc + 0.1, max(df.max())))
plt.xlabel('node-id')
plt.ylabel('visit probability')
plt.ylim([0, max(df.max()) * 1.1])
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'eighty_twenty_mod' + ext)
plt.close('all')

# category bias
df = pd.DataFrame()
data = sorted(np.exp(np.random.random(elements)), reverse=True)
df['special category'] = data
df['others'] = data
mask = np.random.random(elements) < 0.8
df['special category'][mask] = 0.
mask = np.invert(mask)
df['others'][mask] = 0.
sum_both = df['others'].sum() + df['special category'].sum()
df['others'] /= sum_both
df['special category'] /= sum_both
sum_cat = df['special category'].sum()
ax = df.plot(y=['special category','others'], color='rb', kind='bar')
plt.title('special category $\sum$=%.3f' % sum_cat, y=1.08)
plt.ylabel('visit probability')
plt.xlabel('node-id')
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'category_bias_orig' + ext)
plt.close('all')

df['special category'] *= 2
remain = 1. - df['special category'].sum()
fac = remain/df['others'].sum()
df['others'] *= fac
sum_cat = df['special category'].sum()
ax = df.plot(y=['special category','others'], color='rb', kind='bar')
plt.title('special category $\sum$=%.3f' % sum_cat, y=1.08)
plt.ylabel('visit probability')
plt.xlabel('node-id')
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'category_bias_mod' + ext)
plt.close('all')
print df

# empty stocks
target = 0.8
df = pd.DataFrame()
data = sorted(np.exp(np.random.random(elements)), reverse=True)
df['good products'] = data
df['bad products'] = data
mask = np.random.random(elements) > 0.7
mask.sort()
df['good products'][mask] = 0.
mask = np.invert(mask)
df['bad products'][mask] = 0.
sum_both = df['bad products'].sum() + df['good products'].sum()
df['bad products'] /= sum_both
df['good products'] /= sum_both
sum_bad = df['bad products'].sum()
ax = df.plot(y=['good products', 'bad products'], color='rb', kind='bar')
plt.ylabel('visit probability')
plt.xlabel('node-id')
plt.legend(loc='upper left')
#df.plot(y='bad products', color='blue', ax=ax, kind='bar', label='bad products')
plt.title('bad products $\sum$=%.3f' % sum_bad, y=1.08)
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'empty_stocks_orig' + ext)
plt.close('all')

fac = target / df['bad products'].sum()
df['bad products'] *= fac

remain = 1. - df['bad products'].sum()
fac = remain / df['good products'].sum()
df['good products'] *= fac
sum_bad = df['bad products'].sum()
ax = df.plot(y=['good products', 'bad products'], color='rb', kind='bar')
plt.ylabel('visit probability')
plt.xlabel('node-id')
plt.legend(loc='upper left')
plt.title('bad products $\sum$=%.3f' % sum_bad, y=1.08)
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'empty_stocks_mod' + ext)
plt.close('all')

net = collection.data['karate']
weights = net.new_edge_property('float')
weights.a = np.array([1.] * net.num_edges())
weights_data = np.array(weights.a)
bias_ratios = [list(map(int, v.out_neighbours())) for v in net.vertices()]
bias_ratios = map(lambda x: weights_data[x], bias_ratios)
bias_ratios = map(lambda x: x.max() / x.min(), bias_ratios)
v_text = net.new_vertex_property('string')
for v, t in zip(net.vertices(), bias_ratios):
    v_text[v] = "%2.f" % t
bias_ratios = pd.Series(bias_ratios)
bias_ratios.plot(kind='hist', bins=net.num_vertices())
plt.ylabel('# nodes')
plt.xlabel('outgoing bias ratios')
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'bias_ratios_unbiased' + ext)
plt.close('all')
pos = sfdp_layout(net)
graph_draw(net, pos=pos, vertex_text=v_text, output=output_folder + 'bias_ratios_unbiased_net.png')
plt.close('all')

weights_data = np.array(net.degree_property_map('total').a)
bias_ratios = [list(map(int, v.out_neighbours())) for v in net.vertices()]
bias_ratios = map(lambda x: weights_data[x], bias_ratios)
bias_ratios = map(lambda x: x.max() / x.min(), bias_ratios)
for v, t in zip(net.vertices(), bias_ratios):
    v_text[v] = "%2.f" % t
bias_ratios = pd.Series(bias_ratios)
bias_ratios.plot(kind='hist', bins=net.num_vertices())
plt.ylabel('# nodes')
plt.xlabel('outgoing bias ratios')
plt.legend(loc=legend_loc, prop={'size': legend_font_size})
plt.tight_layout()
plt.savefig(output_folder + 'bias_ratios_degree_bias' + ext)
plt.close('all')
graph_draw(net, pos=pos, vertex_text=v_text, output=output_folder + 'bias_ratios_degree_bias_net.png')
plt.close('all')
