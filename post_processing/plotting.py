# coding: utf-8


from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
import os
import numpy as np
import scipy.stats as stats
from utils import gini_coeff
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from tools.mpl_tools import plot_scatter_heatmap
from graph_tool.all import *
from collections import defaultdict, Counter
import powerlaw
from tools.basics import *
import tools.mpl_tools as mpl_tools


def shift_data_pos(data, shift_min=True):
    changed_data = False
    data_lower_z = data < 0
    if any(data_lower_z):
        data += data[data_lower_z].min()
        changed_data = True
    data_near_z = np.isclose(data, 0.)
    if any(data_near_z):
        changed_data = True
        if shift_min:
            data += data[data > 0].min()
        else:
            data += np.finfo(float).eps
    return data, changed_data


def create_bf_scatters_from_df(df, baseline, columns, output_folder='./', filter_zeros=True, legend=True,
                               file_ending='.png', common_range=True, y_range=None, categories=True, **kwargs):
    if isinstance(columns, list):
        columns = list(filter(lambda x: (x != 'category') if categories else True, columns))
    elif isinstance(columns, str):
        columns = [columns]
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    bias_factors_df = pd.DataFrame()
    min_x, min_y, max_x, max_y = None, None, None, None
    if common_range:
        for col in columns:
            x_data = np.array(df[baseline]).astype('float')
            y_data = np.array(df[col]).astype('float')
            if filter_zeros:
                filter_both = np.logical_and(y_data > 0, x_data > 0)
                y_data = y_data[filter_both]
                x_data = x_data[filter_both]
                x_data /= x_data.sum()
                y_data /= y_data.sum()
            y_data /= x_data
            min_x = x_data.min() if min_x is None else min(x_data.min(), min_x)
            min_y = y_data.min() if min_y is None else min(y_data.min(), min_y)
            max_x = x_data.max() if max_x is None else min(x_data.max(), max_x)
            max_y = y_data.max() if max_y is None else min(y_data.max(), max_y)
    if categories and 'category' in df.columns:
        categories = np.array(df['category'])
    else:
        categories = None
    for idx, col in enumerate(columns):
        x = np.array(df[baseline]).astype('float')
        fname = output_folder + 'bf_' + baseline.replace(' ', '_').replace('.', '') + '_' + col.replace(' ',
                                                                                                        '_').replace(
            '.', '') + file_ending
        y = np.array(df[col]).astype('float')
        cat = None
        if filter_zeros:
            filter_both = np.logical_and(x > 0, y > 0)
            x = x[filter_both]
            y = y[filter_both]
            if categories is not None:
                cat = categories[filter_both]
            x /= x.sum()
            y /= y.sum()
        else:
            cat = categories.copy()
        if categories is not None:
            cat_before_after = dict()
            for i in set(categories):
                cat_filt = categories == i
                cat_before_after[i] = x[cat_filt].sum(), y[cat_filt].sum()
            cat_changes = pd.DataFrame(columns=['unbiased', 'biased'])
            for key, val in cat_before_after.iteritems():
                cat_changes.loc[key, 'unbiased'] = val[0]
                cat_changes.loc[key, 'biased'] = val[1]
            cat_changes.plot(kind='bar')
            plt.xticks(rotation=0)
            plt.savefig(fname.rsplit('.', 1)[0] + '_cat_changes.pdf')
            plt.close('all')
        y /= x
        if filter_zeros:
            y_bf = np.zeros(len(filter_both))
            y_bf_idx = np.array(range(len(filter_both)))
            y_bf_idx = y_bf_idx[filter_both]
            y_bf[y_bf_idx] = y
            bias_factors_df[col] = y_bf
        else:
            bias_factors_df[col] = y
        if y_range is not None:
            min_y, max_y = y_range
        create_bf_scatter((baseline, x), (col, y), fname, legend=legend and idx == 0, filter_zeros=True,
                          min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x, categories=cat, **kwargs)
    return bias_factors_df


def create_bf_scatter(x, y, fname, min_y=None, max_y=None, min_x=None, max_x=None, filter_zeros=True,
                      legend=True, categories=None, **kwargs):
    plt.close('all')
    font_size = 22
    matplotlib.rcParams.update({'font.size': font_size})
    assert isinstance(x, tuple)
    assert isinstance(y, tuple)
    x_label, x_data = x
    y_label, y_data = y
    x_data = np.array(x_data).astype('float')
    y_data = np.array(y_data).astype('float')
    y_data_mod = False
    x_data_mod = False
    orig_len = len(y_data)
    if filter_zeros:
        filter_both = np.logical_and(y_data > 0, x_data > 0)
        y_data = y_data[filter_both]
        x_data = x_data[filter_both]
        if categories is not None:
            categories = categories[filter_both]
    print 'biasfactor dataset len:', len(y_data), '(', len(y_data) / orig_len * 100, '%)'
    alpha = min(1., 1 / np.log10(len(y_data)))
    f, ax = plt.subplots()
    # plt.axhline(1., color='red', alpha=1., lw=3, ls='--')
    if categories is None:
        for i in reversed(range(3)):
            if i == 0:
                filtered_y = y_data > 1
                label = 'Increase'
                marker = '^'
                c = 'red'
            elif i == 1:
                filtered_y = np.isclose(y_data, 1.)
                label = 'Neutral'
                c = 'gray'
                marker = 'o'
            else:
                filtered_y = y_data < 1
                label = 'Decrease'
                c = 'blue'
                marker = 'v'
            x_filt, y_filt = x_data[filtered_y], y_data[filtered_y]
            # print '\tfiltered len', i, len(x_filt)
            ax.scatter(x=x_filt, y=y_filt, alpha=alpha, s=90, color=c, lw=1, label=label,
                       marker=marker, facecolors='none', **kwargs)
    else:
        category_dist = Counter(categories)
        colors = ['blue', 'red']
        markers = ['o', '^']
        assert isinstance(categories, np.ndarray)
        for i, c, m in zip(zip(*sorted(category_dist.iteritems(), key=lambda x: x[1], reverse=True))[0], colors,
                           markers):
            belong_to_cat = categories == i
            x_filt = x_data[belong_to_cat]
            y_filt = y_data[belong_to_cat]
            print 'category:', i, 'pages:', len(x_filt)
            ax.scatter(x=x_filt, y=y_filt, alpha=alpha, s=90, color=c, lw=1, label=i,
                       marker=m, facecolors='none', **kwargs)
        plt.legend(loc='lower center')
        y_label = 'Bias Factor'

    if min_y is None or max_y is None:
        min_y, max_y = y_data.min(), y_data.max()
    if min_x is None or max_x is None:
        min_x, max_x = x_data.min(), x_data.max()
    min_y = min(min_y, 0.1)
    max_y = max(max_y, 10)
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ranges = [min_x, max_x, min_y, max_y]
    plt.xlabel(x_label + (' (shifted)' if x_data_mod else ''))
    plt.ylabel(y_label + (' (shifted)' if y_data_mod else ''))
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid()
    plt.tight_layout()
    try:
        plt.savefig(fname, dpi=150)
    except:
        pass
    # plt.show()

    if legend:
        print 'plot legend'
        legend_fname = fname.rsplit('/', 1)[0] + '/bf_scatter_legend.pdf'
        mpl_tools.plot_legend(ax=ax, filename=legend_fname, font_size=12, figsize=(16, 3), ncols=4)
    plt.close('all')

    plot_scatter_heatmap(x_data, y_data, logy=True, logx=True, logbins=True, bins=100,
                         axis_range=[[min_x, max_x], [min_y, max_y]])
    plt.xlabel(x_label + (' (shifted)' if x_data_mod else ''))
    plt.ylabel(y_label + (' (shifted)' if y_data_mod else ''))
    # plt.axhline(np.log10(1.), color='white', alpha=1., lw=3, ls='--')
    plt.plot(np.log10(np.array([min_x, max_x])), np.log10(np.array([min_x, max_x])), color='white', alpha=6., lw=1,
             ls='-')

    # categorize nodes depending on their stat. val.
    x_data = np.array(x_data)
    x_data.sort()
    x_cum = x_data.cumsum()
    v_lines = list()
    v_lines.append(x_data[np.searchsorted(x_cum, .33 * x_data.sum()) - 1])
    v_lines.append(x_data[np.searchsorted(x_cum, .67 * x_data.sum()) - 1])

    for val in v_lines:
        plt.axvline(np.log10(val), color='green', alpha=6., lw=1, ls='-')

    plt.tight_layout()
    fname = fname.rsplit('.', 1)[0] + '_heatmap.pdf'
    mpl_tools.save_n_crop(fname)
    plt.close('all')
    # plt.show()
    return ranges


def create_scatters_from_df(df, columns, output_folder='./', filter_zeros=True, file_ending='.png', **kwargs):
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(filter(lambda x: x != 'category', columns))
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for idx, col in enumerate(columns):
        for jdx, col2 in enumerate(columns):
            if jdx > idx:
                print 'scatter', col, col2
                fname = output_folder + 'scatter_' + col.replace(' ', '_') + '_' + col2.replace(' ', '_') + file_ending
                create_scatter(df, col, col2, fname, filter_zeros=filter_zeros)


def create_scatter(df, x, y, fname, filter_zeros=True):
    matplotlib.rcParams.update({'font.size': 14})
    x_data = np.array(df[x]).astype('float')
    y_data = np.array(df[y]).astype('float')
    if filter_zeros:
        filter_both = np.logical_and(y_data > 0, x_data > 0)
        y_data = y_data[filter_both]
        x_data = x_data[filter_both]
        x_data /= x_data.sum()
        y_data /= y_data.sum()

    corr_df = pd.DataFrame()
    corr_df[x] = x_data
    corr_df[y] = y_data
    print '\tpearson:',  float(corr_df.corr(method='pearson').at[x, y])
    #print '\tlogpearson:', float(corr_df.apply(np.log10).corr(method='pearson').at[x, y])
    #print '\tspearman:', float(corr_df.corr(method='spearman').at[x, y])
    # print '\tkendall:', float(corr_df.corr(method='kendall').at[x, y])

    alpha = 1 / np.log2(len(x_data))
    f, ax = plt.subplots()
    c = 'black'
    marker = 'x'
    ax.scatter(x=x_data, y=y_data, alpha=alpha, s=90, color=c, lw=1, marker=marker, facecolors='none')
    min_x, max_x = x_data.min(), x_data.max()
    min_y, max_y = y_data.min(), y_data.max()
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    plt.xlabel(x)
    plt.ylabel(y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    plt.close('all')

    plot_scatter_heatmap(x_data, y_data, logx=True, logy=True, logbins=True, bins=100)
    x_data_log = np.log10(x_data)
    y_data_log = np.log10(y_data)
    plt.plot(np.linspace(x_data_log.min(), x_data_log.max()), np.linspace(y_data_log.min(), y_data_log.max()), lw=2,
             c='black', alpha=.7)  # , ls='--')
    plt.xlabel(x)
    plt.ylabel(y)
    fname = fname.replace('.png', '') + '_heatmap.png'
    print('plot line on:', fname)
    plt.savefig(fname, dpi=150)
    plt.show()
    plt.close('all')


def create_ginis_from_df(df, columns=None, output_folder='./', zoom=None, filter_zeros=True, legend=True, font_size=16,
                         ms=5, out_fn=None, **kwargs):
    if columns is None:
        columns = list(df.select_dtypes(include=[np.float, np.int]).columns)
    if isinstance(columns, str):
        columns = [columns]
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    matplotlib.rcParams.update({'font.size': font_size})
    df = df.copy()[columns]
    if filter_zeros:
        for i in columns:
            df = df[df[i] > 0]
        df /= df.sum()

    for i in columns:
        df[i] = sorted(df[i])
    df = df.cumsum()
    df_len = len(df)
    df['idx'] = (np.array(range(df_len)).astype('float') + 1.) / df_len * 100
    #mark_every = int(len(df)/20)
    colors = ['red', 'green', 'blue', 'cyan']
    marker = ['o', 'v', '^', 's', '+', 'D', '<', '>', 'p', '*', 'x']
    ax = df.plot(x='idx', alpha=0.9, legend=False, color=colors, **kwargs)
    if df_len > 100:
        #if zoom is not None:
        #    df['idx'] = df['idx'].apply(lambda x: x * 10 if x >= zoom else x)
        #else:
        scatter_df = df.copy()
        scatter_df.at[scatter_df.index[-1], 'idx'] = 200
        scatter_df['idx'] = scatter_df['idx'].astype('int')
        scatter_df['idx'] = scatter_df['idx'].apply(lambda x: int(x / 5) * 5)
        scatter_df.drop_duplicates(subset=['idx'], inplace=True)
        #if zoom is not None:
        #    df['idx'] = df['idx'].apply(lambda x: x / 10 if x >= zoom else x)
        #else:
        scatter_df.at[scatter_df.index[-1], 'idx'] = 100
    else:
        scatter_df = df
    for i, m, c in zip(df.columns, marker, colors):
        if i != 'idx':
            ax = scatter_df.plot(x='idx', y=i, alpha=.9, ms=ms, legend=False, style=m, markerfacecolor=c,
                         ax=ax, **kwargs)
            # ax = df.plot(x='idx', alpha=0.9, marker=,
            # legend=False, ax=ax, **kwargs)
    print df.tail()
    ax.set_xlabel('nodes sorted by stat. prob.')
    ax.set_ylabel('cum. sum of stat. prob.')
    plt.yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
    plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    plt.plot([0, 100], [0, 1], ls='--', lw=1, label='Unif.', alpha=1.)
    if zoom is not None:
        axins = zoomed_inset_axes(ax, 3, loc=2) # zoom = 6
        axins = df.plot(x='idx', alpha=0.9, legend=False, color=colors, ax=axins, **kwargs)
        for i, m, c in zip(df.columns, marker, colors):
            if i != 'idx':
                axins = scatter_df.plot(x='idx', y=i, alpha=.9, ms=ms, legend=False, style=m, markerfacecolor=c,
                             ax=axins, **kwargs)
        axins.plot([0, 100], [0, 1], ls='--', lw=1, alpha=1.)
        axins.set_xlim(zoom, 100)
        axins.set_ylim(0, .25)
        axins.set_axis_bgcolor('lightgray')
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        axins.set_ylabel('')
        axins.set_xlabel('')
        mark_inset(ax, axins, loc1=1, loc2=3, fc="lightgray", ec=".75")
    plt.tight_layout()
    out_fn = out_fn if out_fn is not None else 'gini.pdf'
    plt.savefig(output_folder + out_fn)
    if legend:
        print 'plot legend'
        legend_ax = None
        for i, m, c in zip(df.columns, marker, colors):
            if i != 'idx':
                legend_ax = df.plot(x='idx', y=i, alpha=.9, marker=m, ms=ms, color=c, ax=legend_ax, **kwargs)
        legend_ax.plot([0, 100], [0, 1], ls='--', lw=1, alpha=1.)
        legend_fname = output_folder + 'gini_legend.pdf'
        mpl_tools.plot_legend(legend_ax, legend_fname, font_size=20, figsize=(30, 3), ncols=3,
                              labels=list(df.columns) + ['Uniform'])
    plt.close('all')


def plot_bar_plot(entropy_rates, filename, y_label):
    def_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 25})
    # entropy_rates = entropy_rates.T
    f, ax = plt.subplots(figsize=(20, 8))
    hatch = ['-', 'x', '\\', '*', 'o', 'O', '.', '/'] * 2
    #symbols = ['$\\clubsuit$', '$\\bigstar$', '$\\diamondsuit$', '$\\heartsuit', '$\\spadesuit$', '$\\blacksquare$']
    symbols = ['O', 'E', 'D', 'I', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    colors = ['blue', 'green', 'red', 'black', 'magenta', 'orange', 'gray'] * 2
    num_ds = len(entropy_rates.columns)
    num_ticks = len(entropy_rates)
    width = 0.7
    dataset_offset = width / num_ds
    space = 0.02
    rects = list()
    for idx, (i, h, c, s) in enumerate(zip(entropy_rates.columns, hatch, colors, symbols)):
        pos = 0. - (width / 2) + idx * dataset_offset + idx * (space/2)
        pos = np.array([pos + idx for idx in xrange(num_ticks)])
        # print idx, step, width
        #print pos
        #print width
        #print i
        rects.append(
            ax.bar(pos, entropy_rates[i], (width / num_ds - space), color='white', label=s + ': ' + i.decode('utf8'),
                   lw=2, alpha=1., hatch=h, edgecolor=c))
        autolabel(s, pos + (width / num_ds - space) / 2, entropy_rates[i], ax)
        # ax = entropy_rates[i].plot(position=pos,width=0.8, kind='bar',rot=20,ax=ax, alpha=1,lw=0.4,hatch=h,color=c)

    ax.set_position([0.1, 0.2, .8, 0.6])
    plt.xticks(np.array(range(len(entropy_rates))), entropy_rates.index, rotation=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linewidth=3, alpha=0.2, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tick_params(labelright=True)
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.35))
    plt.ylim([min(list(entropy_rates.min())) * .95, max(list(entropy_rates.max())) * 1.05])
    plt.ylabel(y_label)
    # plt.subplots_adjust(top=0.7)
    # plt.tight_layout()
    plt.savefig(filename, bbox_tight=True)
    plt.close('all')
    os.system('pdfcrop ' + filename + ' ' + filename)
    # plt.show()
    matplotlib.rcParams.update({'font.size': def_font_size})


def autolabel(symbol, x_pos, heights, ax):
    # attach some text labels
    for x, h in zip(x_pos, heights):
        ax.text(x, 1.01 * h, symbol, ha='center', va='bottom')


def plot_degree_distributions(filnames, output_dir):
    create_folder_structure(output_dir)
    matplotlib.rcParams.update({'font.size': 30})
    for idx, i in enumerate(filnames):
        net = load_graph(i)
        net_fn_name = i.rsplit('/', 1)[-1].replace('.gt', '')
        # net_name = i.rsplit('/', 1)[-1].replace('.gt', '').replace('_', ' ')
        for deg_type in ['in', 'out', 'total']:
            counts, bins = vertex_hist(net, deg_type)
            df = pd.DataFrame(index=bins[:-1], data=counts, columns=[deg_type])
            df.plot(loglog=True)
            # pl = powerlaw.Fit(np.array(deg_seq))
            #print '-' * 10
            #print i.rsplit('/', 1)[-1]
            #print 'powerlaw'
            #print '\txmin', pl.xmin
            #print '\talpha', pl.alpha
            #print '-' * 10
            plt.grid('on')
            plt.xlabel('degree')
            plt.ylabel('number of vertices')
            plt.tight_layout()
            plt.savefig(output_dir + net_fn_name + '_' + deg_type + '.pdf')
            plt.show()
            plt.close('all')

