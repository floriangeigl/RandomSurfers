from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt
import matplotlib.cm as colormap
#import seaborn
import pandas as pd
import utils
import numpy as np
from graph_tool.all import *
import scipy.stats as stats
import utils

font_size = 15
matplotlib.rcParams.update({'font.size': font_size})


def create_scatter(x, y, fname, **kwargs):
    assert isinstance(x, tuple)
    assert isinstance(y, tuple)
    x_label, x_data = x
    y_label, y_data = y
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    #y_data = utils.shift_data_pos(y_data)
    #x_data = utils.shift_data_pos(x_data)
    #df = pd.DataFrame(columns=[x_label], data=x_data)
    #df[y_label] = y_data
    alpha = 1 / np.log10(len(y_data))
    f, ax = plt.subplots()
    pearson = stats.pearsonr(np.log10(x_data), np.log10(y_data))[0]
    if not np.isnan(pearson):
        ax.plot(None, lw=0, c='white', alpha=0., label='log10 pearson: ' + "%.2f" % pearson)
    for i in range(3):
        if i == 0:
            filt = y_data > 1
            label = 'increased'
            marker = '^'
            c = 'red'
        elif i == 1:
            filt = np.isclose(y_data, 1.)
            label = 'neutral'
            c = 'gray'
            marker = 'o'
        else:
            filt = y_data < 1
            label = 'decreased'
            c = 'blue'
            marker = 'v'
        x_filt, y_filt = x_data[filt], y_data[filt]
        ax.scatter(x=x_filt, y=y_filt, alpha=alpha, color=c, lw=0, label=label, marker=marker, **kwargs)
    plt.axhline(1., color='red', alpha=.25, lw=2, ls='--')
    y_min, y_max = y_data.min(), y_data.max()
    x_min, x_max = x_data.min(), x_data.max()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close('all')


def draw_graph(network, color, min_color=None, max_color=None, groups=None, sizep=None, colormap_name='bwr', min_vertex_size_shrinking_factor=4, output='graph.png', output_size=(15, 15), dpi=80, standardize=False, color_bar=True, **kwargs):
    output_splitted = output.rsplit('/', 1)[-1].split('_graph_')
    net_name, prop_key = output_splitted[0], output_splitted[-1]
    print_prefix = utils.color_string('[' + net_name + '] ') + '[' + prop_key + '] draw graph'
    print print_prefix
    print_prefix += ': '
    num_nodes = network.num_vertices()
    min_vertex_size_shrinking_factor = min_vertex_size_shrinking_factor
    if num_nodes < 10:
        num_nodes = 10
    max_vertex_size = np.sqrt((np.pi * (min(output_size) * dpi / 2) ** 2) / num_nodes)
    if max_vertex_size < min_vertex_size_shrinking_factor:
        max_vertex_size = min_vertex_size_shrinking_factor
    min_vertex_size = max_vertex_size / min_vertex_size_shrinking_factor
    if sizep is None:
        sizep = max_vertex_size + min_vertex_size
        sizep /= 3
    else:
        sizep = prop_to_size(sizep, mi=min_vertex_size / 3 * 2, ma=max_vertex_size / 3 * 2, power=2)
    v_shape = 'circle'
    if isinstance(groups, str):
        try:
            v_shape = network.vp[groups].copy()
            #groups = network.vp[groups]
            #unique_groups = set(np.array(groups.a))
            #num_groups = len(unique_groups)
            #groups_c_map = colormap.get_cmap('gist_rainbow')
            #groups_c_map = {i: groups_c_map(idx / (num_groups - 1)) for idx, i in enumerate(unique_groups)}
            #v_pen_color = network.new_vertex_property('vector<float>')
            #for v in network.vertices():
            #    v_pen_color = groups_c_map[groups[v]]

            v_shape.a %= 14
        except KeyError:
            print print_prefix + 'cannot find groups property:', groups
            v_shape = 'circle'

    cmap = colormap.get_cmap(colormap_name)
    color = color.copy()


    try:
        _ = color.a
    except AttributeError:
        c = network.new_vertex_property('float')
        c.a = color
        color = c
    min_color = color.a.min() if min_color is None else min_color
    max_color = color.a.max() if max_color is None else max_color
    if np.isclose(min_color, max_color):
        min_color = 0
        max_color = 2

    #orig_color = np.array(color.a)
    if standardize:
        color.a -= color.a.mean()
        color.a /= color.a.var()
        color.a += 1
        color.a /= 2
    else:
        #color.a -= min_color
        #color.a /= max_color
        tmp = np.array(color.a)
        tmp[tmp > 1] = 1 + (tmp[tmp > 1] / (max_color/1))
        color.a = tmp
        color.a /= 2
    if not output.endswith('.png'):
        output += '.png'
    color_pmap = network.new_vertex_property('vector<float>')
    tmp = np.array([np.array(cmap(i)) for i in color.a])
    color_pmap.set_2d_array(tmp.T)
    plt.switch_backend('cairo')
    f, ax = plt.subplots(figsize=(15, 15))
    output_size = (output_size[0] * 0.8, output_size[1])
    edge_alpha = 0.3 if network.num_vertices() < 1000 else 0.01
    pen_width = 0.8 if network.num_vertices() < 1000 else 0.1
    v_pen_color = [0., 0., 0., 1] if network.num_vertices() < 1000 else [0.0, 0.0, 0.0, edge_alpha]
    graph_draw(network, vertex_fill_color=color_pmap, mplfig=ax, vertex_pen_width=pen_width, vertex_shape=v_shape,
               vertex_color=v_pen_color, edge_color=[0.179, 0.203, 0.210, edge_alpha], vertex_size=sizep,
               output_size=output_size, output=output, **kwargs)
    if color_bar:
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array([0., 2.])
        cbar = f.colorbar(cmap, drawedges=False)
        ticks = [0, 1.0, max_color / 1]
        cbar.set_ticks([0., 1., 2.])
        tick_labels = None
        non_zero_dig = 1
        for digi in range(10):
            tick_labels = [str("{:2." + str(digi) + "f}").format(i) for i in ticks]
            if any([len(i.replace('.', '').replace('0', '').replace(' ', '').replace('-', '').replace('+', '')) > 0 for
                    i in tick_labels]):
                non_zero_dig -= 1
                if non_zero_dig == 0:
                    break
        cbar.ax.set_yticklabels(tick_labels)
        #var = stats.tvar(orig_color)
        #cbar.set_label('')
    plt.axis('off')
    plt.savefig(output, bbox_tight=True, dpi=dpi)
    plt.close('all')
    plt.switch_backend('Agg')
    print print_prefix + 'done'


def plot_stat_dist(ser, output_filename, **kwargs):
    assert isinstance(ser, pd.Series)
    ser.plot(kind='hist', **kwargs)
    # plt.title(key)
    plt.ylabel('#nodes')
    plt.xlabel('stationary value')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close('all')