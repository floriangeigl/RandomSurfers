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
                            file_ending='.png', common_range=True, **kwargs):
    if isinstance(columns, str):
        columns = [columns]
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    bias_factors_df = pd.DataFrame()
    x = df[baseline]
    min_x, min_y, max_x, max_y = None, None, None, None
    if common_range:
        for col in columns:
            x_data = np.array(x)
            y_data = np.array(df[col] / x_data)
            if filter_zeros:
                filter_both = np.logical_and(y_data > 0, x_data > 0)
                y_data = y_data[filter_both]
                x_data = x_data[filter_both]
            min_x = x_data.min() if min_x is None else min(x_data.min(), min_x)
            min_y = y_data.min() if min_y is None else min(y_data.min(), min_y)
            max_x = x_data.max() if max_x is None else min(x_data.max(), max_x)
            max_y = y_data.max() if max_y is None else min(y_data.max(), max_y)

    for idx,col in enumerate(columns):
        fname = output_folder + 'bf_' + baseline.replace(' ', '_') + '_' + col.replace(' ', '_') + file_ending
        y = df[col] / x
        bias_factors_df[col] = y
        create_bf_scatter((baseline, x), (col, y), fname, legend=legend and idx == 0, filter_zeros=filter_zeros,
                          min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x, )
    return bias_factors_df

def create_bf_scatter(x, y, fname, min_y=None, max_y=None, min_x=None, max_x=None, filter_zeros=True, legend=True, **kwargs):
    matplotlib.rcParams.update({'font.size': 14})
    assert isinstance(x, tuple)
    assert isinstance(y, tuple)
    x_label, x_data = x
    y_label, y_data = y
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_data_mod = False
    x_data_mod = False
    if filter_zeros:
        filter_both = np.logical_and(y_data > 0, x_data > 0)
        y_data = y_data[filter_both]
        x_data = x_data[filter_both]
        x_data /= x_data.sum()
        y_data /= y_data.sum()
    alpha = 1 / np.log10(len(y_data))
    f, ax = plt.subplots()
    plt.axhline(1., color='red', alpha=.5, lw=4, ls='--')
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
        ax.scatter(x=x_filt, y=y_filt, alpha=alpha, s=90, color=c, lw=1, label=label,
                   marker=marker, facecolors='none', **kwargs)

    if min_y is None or max_y is None:
        min_y, max_y = y_data.min(), y_data.max()
    if min_x is None or max_x is None:
        min_x, max_x = x_data.min(), x_data.max()
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    plt.xlabel(x_label + (' (shifted)' if x_data_mod else ''))
    plt.ylabel(y_label + (' (shifted)' if y_data_mod else ''))
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    plt.close('all')
    if legend:
        print 'plot legend'
        matplotlib.rcParams.update({'font.size': 12})
        f2 = plt.figure(figsize=(16, 3))
        f2.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
        legend_fname = fname.rsplit('/', 1)[0] + '/bf_scatter_legend.pdf'
        plt.savefig(legend_fname, bbox_tight=True)
        os.system('pdfcrop ' + legend_fname + ' ' + legend_fname)
        plt.show()
        plt.close('all')


def create_scatters_from_df(df, columns, output_folder='./', filter_zeros=True, file_ending='.png', **kwargs):
    if isinstance(columns, str):
        columns = [columns]
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
    if filter_zeros:
        data = df[df[x] > 0]
        data = data[data[y] > 0]
        data[[x, y]] /= data[[x, y]].sum()
        x_data = data[x]
        y_data = data[y]
    else:
        x_data = df[x]
        y_data = df[y]
    corr_df = pd.DataFrame()
    corr_df[x] = x_data
    corr_df[y] = y_data
    print '\tpearson:',  float(corr_df.corr(method='pearson').at[x, y])
    print '\tlogpearson:', float(corr_df.apply(np.log10).corr(method='pearson').at[x, y])
    print '\tspearman:', float(corr_df.corr(method='spearman').at[x, y])
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
    orig_xticks, orig_xtick_labels = plt.xticks()
    plt.savefig(fname, dpi=150)
    plt.show()
    plt.close('all')
    bins = 100
    heatmap, xedges, yedges = np.histogram2d(np.log10(x_data), np.log10(y_data), bins=bins)
    heatmap = np.log10(heatmap + 1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax = plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', interpolation='none', cmap='jet')

    ticks, lticks = plt.xticks()
    ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
    plt.xticks(ticks, map(lambda x: '$10^{' + x + '}$', map(str, map(int, ticks))))
    ticks, lticks = plt.yticks()
    ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
    plt.yticks(ticks, map(lambda x: '$10^{' + x + '}$', map(str, map(int, ticks))))
    plt.xlim([xedges[0], xedges[-1]])
    plt.ylim([yedges[0], yedges[-1]])
    plt.xlabel(x)
    plt.ylabel(y)
    cb = plt.colorbar(ax)
    vmin = int(cb.vmin)
    vmax = int(cb.vmax)
    ticks = range(vmin, vmax + 1)
    lin_ticks = np.linspace(0, 1., num=len(ticks), endpoint=True)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(map(lambda x: '$10^{' + x + '}$', map(str, map(int, ticks))))

    # plt.xticks(np.log10(orig_xticks), map(str, orig_xtick_labels))
    if False:
        ticks, ticklabels = plt.xticks()
        ticklabels = ['1e' + i for i in map(str, map(int, ticks))]
        plt.xticks(ticks, ticklabels)
        ticks, ticklabels = plt.yticks()
        ticklabels = ['1e' + i for i in map(str, map(int, ticks))]
        plt.yticks(ticks, ticklabels)

        ticks = range(int(cb.vmin), int(cb.vmax) + 2)
        cb.set_ticks(ticks)
        cb.set_ticklabels(['1e' + i for i in map(str, map(int, ticks))])

    fname = fname.replace('.png', '') + '_heatmap.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    plt.close('all')



def plot_gini(data, fname):

    data = np.array(data)
    data.sort()
    df = pd.DataFrame(columns=['idx', 'cum. sum of stat. prob.'], data=zip(data, range(len(data))))


def create_ginis_from_df(df, columns, output_folder='./', zoom=None, filter_zeros=True, legend=True, font_size=12,
                         ms=5, file_ending='.png',
                         **kwargs):
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
        if zoom is not None:
            df['idx'] = df['idx'].apply(lambda x: x * 10 if x >= zoom else x)
        else:
            df.at[df.index[-1], 'idx'] = 200
        df['idx'] = df['idx'].astype('int')
        df['idx'] = df['idx'].apply(lambda x: int(x / 5) * 5)
        df.drop_duplicates(subset=['idx'], inplace=True)
        if zoom is not None:
            df['idx'] = df['idx'].apply(lambda x: x / 10 if x >= zoom else x)
        else:
            df.at[df.index[-1], 'idx'] = 100
    for i, m, c in zip(df.columns, marker, colors):
        if i != 'idx':
            ax = df.plot(x='idx', y=i, alpha=.9, ms=ms, legend=False, style=m, markerfacecolor=c,
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
        print axins
        df.plot(x='idx', alpha=0.9, style=['-o', '-v', '-^', '-s', '-+', '-D', '-<', '->', '-p', '-*', '-x'],
                legend=False, ax=axins, **kwargs)
        axins.set_xlim(zoom, 100)
        axins.set_ylim(0, .25)
        axins.set_axis_bgcolor('lightgray')
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        axins.set_ylabel('')
        axins.set_xlabel('')
        mark_inset(ax, axins, loc1=1, loc2=3, fc="lightgray", ec=".75")
    plt.tight_layout()
    plt.savefig(output_folder + 'gini.pdf')
    if legend:
        print 'plot legend'
        legend_ax = None
        for i, m, c in zip(df.columns, marker, colors):
            if i != 'idx':
                legend_ax = df.plot(x='idx', y=i, alpha=.9, marker=m, ms=ms, legend=False, color=c, ax=legend_ax, **kwargs)
        plt.plot([0, 100], [0, 1], ls='--', lw=1, label='Unif.', alpha=1.)
        matplotlib.rcParams.update({'font.size': 20})
        f2 = plt.figure(figsize=(30,3))
        f2.legend(*legend_ax.get_legend_handles_labels(), loc='center', ncol=min(len(df.columns),8))
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.85)
        legend_fname = output_folder + 'gini_legend.pdf'
        plt.savefig(legend_fname, bbox_tight=True)
        plt.show()
        plt.close('all')
        os.system('pdfcrop ' + legend_fname + ' ' + legend_fname)

'''
data_dir = '/home/fgeigl/navigability_of_networks/output/iknow/'
base_dir = data_dir.rsplit('/',2)[0] + '/'
files = [data_dir + i for i in os.listdir(data_dir) if i.endswith('stationary_dist.df')]
print files
metrics = dict()
#metrics.add('betweenness')
#metrics.add('cosine')
metrics['eigenvector'] = 'Eigenvec. C.'
#metrics.add('inv_log_eigenvector')
#metrics.add('inv_sqrt_eigenvector')
#metrics.add('eigenvector_inverse')
metrics['deg'] = 'Degree'
#metrics.add('inv_log_deg')
metrics['inv_sqrt_deg'] = 'Inv. Degree'
metrics['sigma'] = 'Sigma'
metrics['sigma_sqrt_deg_corrected'] = 'Deg. Cor. Sigma'
metrics['adj'] = 'Unbiased'
#metrics['graph_with_props_text_sim.bias'] = 'TfIdf'
metrics['click_sub'] = 'ClickBias'
metrics['page_counts'] = 'PageViews'
#[u'inv_log_eigenvector', u'inv_sqrt_eigenvector', 
#u'inv_log_deg', u'adjacency', u'inv_sqrt_deg', u'eigenvector', u'sigma', u'betweenness', u'sigma_deg_corrected']
#'adjacency', u'betweenness', u'eigenvector', u'inv_log_deg', u'inv_log_eigenvector', 
#u'inv_sqrt_deg', u'inv_sqrt_eigenvector', u'sigma', u'sigma_deg_corrected', u'sigma_log_deg_corrected'
karate_metrics = set()
karate_metrics.add('adjacency')
karate_metrics.add('eigenvector')

for idx, file_name in enumerate(sorted(files)):
    print file_name
    
    data_set_name = file_name.split('/')[-1]
    data_set_name = data_set_name.replace('_stat_dists.df','')
    df = pd.read_pickle(file_name)
    df = df[df['adj']>0]
    df = df[df['click_sub']>0]
    df = df[df['page_counts']>0]
    print df[['adj', 'click_sub', 'page_counts']].sum()
    for i in ['adj', 'click_sub', 'page_counts']:
        df[i] /= df[i].sum()
    #filtered_nodes = np.load('/home/fgeigl/navigability_of_networks/preprocessing/data/af_clicked_nodes')
    print df.columns
    #drop_idx = set(df.index) - set(filtered_nodes)
    #df = df.drop(drop_idx)
    #assert set(df.index) == set(filtered_nodes)
    #df /= df.sum()
    print len(df), 'nodes'
    baseline_name = 'click_sub'
    baseline_name = 'adj'
    baseline = df[baseline_name]
    min_pfac = 1.
    max_pfac = 1.
    sorted_baseline = sorted(baseline)
    min_x = baseline[baseline > 0].min()
    x_perc = len(baseline[baseline > 0]) / len(baseline)
    max_x = baseline.max()
    #if abs(round(np.log10(min_x)) - round(np.log10(max_x))) < 1:
    min_x = 10 ** (np.log10(min_x)-.2)
    max_x = 10 ** (np.log10(max_x)+.2)
    while np.isclose(np.ceil(np.log10(min_x)), np.floor(np.log10(max_x))):
        min_x = 10 ** (np.log10(min_x)-.1)
        max_x = 10 ** (np.log10(max_x)+.1)
        
    prob_fac_df = pd.DataFrame()
    
    for col in sorted(filter(lambda x: (x in metrics) if 'karate' not in file_name else (x in karate_metrics),df.columns)):
        print col,
        prob_fac = df[col] / baseline
        pos_prob_fac = (prob_fac > 0) & (np.isfinite(prob_fac))
        #print prob_fac[pos_prob_fac].min()
        #print prob_fac[pos_prob_fac].max()
        #print np.nanmin(prob_fac[pos_prob_fac])
        #print np.nanmax(prob_fac[pos_prob_fac])
        min_pfac = min([prob_fac[pos_prob_fac].min(),min_pfac])
        max_pfac = max([prob_fac[pos_prob_fac].max(),max_pfac])            
        prob_fac_df[col] = prob_fac
    print ''
    print prob_fac_df.min(axis=0)
    print prob_fac_df.max(axis=0)
    for jdx,col in enumerate(prob_fac_df.columns):
        col_data = prob_fac_df[col]
        col_perc = len(col_data[col_data>0]) / len(col_data)
        col_neg = len(col_data[col_data<0]) / len(col_data)
        x = (baseline_name ,baseline)
        # x = ('baseline' +'('+'%.2f' % x_perc + ')' ,baseline)
        y_label_name = metrics[col] if col in metrics else (' '.join([i[:3] + '.' for i in col.split('_')]))
        # y = (y_label_name + ' fac.' +'('+'%.2f' % col_perc + '|n:' + '%.2f' % col_neg, col_data)
        y = (y_label_name + ' Bias Factor', col_data)
        if data_set_name.endswith('.df'):
            data_set_name = data_set_name[:-3]
        out_name = './scatters/' + data_set_name + '_scatter_' + col
        if not os.path.isdir('./scatters/'):
            os.mkdir('./scatters/')
        create_scatter(x, y, out_name, min_pfac, max_pfac, min_x=min_x, max_x = max_x, legend=jdx==idx==0)
'''
