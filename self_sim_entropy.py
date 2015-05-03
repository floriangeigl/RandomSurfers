from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt
import plotting
import os
import numpy as np
import pandas as pd
import operator
import utils
import network_matrix_tools
from collections import defaultdict
import scipy
import traceback
import datetime
import multiprocessing as mp
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

font_size = 12
matplotlib.rcParams.update({'font.size': font_size})
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=225)


def try_dump(data, filename):
    try:
        data.dump(filename)
        return True
    except SystemError:
        return False


def calc_bias(filename, biasname, data_dict, dump=True, verbose=1):
    dump_filename = filename + '_' + biasname
    name = filename.rsplit('/', 1)[-1].replace('.gt', '')
    if verbose > 0:
        print utils.color_string('[' + name + ']'), '[' + biasname + ']', '[' + str(
            datetime.datetime.now().replace(microsecond=0)) + ']', 'calc bias'
    loaded = False
    if biasname == 'adjacency':
        return None
    elif biasname == 'eigenvector':
        try:
            A_eigvector = np.load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvector = data_dict['eigvec']
            except KeyError:
                A_eigvalue, A_eigvector = eigenvector(data_dict['net'])
                A_eigvalue = np.float(A_eigvalue)
                A_eigvector = np.array(A_eigvector.a)
                data_dict['eigval'] = A_eigvalue
                data_dict['eigvec'] = A_eigvector
        if dump and not loaded:
            try_dump(A_eigvector, dump_filename)
        return A_eigvector
    elif biasname == 'eigenvector_inverse':
        try:
            A_eigvector_inf = np.load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvector = data_dict['eigvec']
            except KeyError:
                A_eigvector = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
            A_eigvector_inf = 1. / A_eigvector
        if dump and not loaded:
            try_dump(A_eigvector_inf, dump_filename)
        return A_eigvector_inf
    elif biasname == 'sigma':
        try:
            sigma = np.load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvalue = data_dict['eigval']
            except KeyError:
                _ = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
                A_eigvalue = data_dict['eigval']
            sigma = network_matrix_tools.katz_sim_network(data_dict['adj'], largest_eigenvalue=A_eigvalue)
        if dump and not loaded:
            try_dump(sigma, dump_filename)
        return sigma
    elif biasname == 'sigma_deg_corrected':
        try:
            sigma_deg_cor = np.load(dump_filename)
            loaded = True
        except IOError:
            sigma_deg_cor = calc_bias(filename, 'sigma', data_dict, dump=dump, verbose=verbose-1) / np.array(
                data_dict['net'].degree_property_map('total').a)
        if dump and not loaded:
            try_dump(sigma_deg_cor, dump_filename)
        return sigma_deg_cor
    elif biasname == 'cosine':
        try:
            cos = np.load(dump_filename)
            loaded = True
        except IOError:
            cos = network_matrix_tools.calc_cosine(data_dict['adj'], weight_direct_link=True)
        if dump and not loaded:
            try_dump(cos, dump_filename)
        return cos
    elif biasname == 'betweenness':
        try:
            bet = np.load(dump_filename)
            loaded = True
        except IOError:
            bet = np.array(betweenness(data_dict['net'])[0].a)
        if dump and not loaded:
            try_dump(bet, dump_filename)
        return bet
    elif biasname == 'deg':
        return np.array(data_dict['net'].degree_property_map('total').a)
    elif biasname == 'inv_deg':
        return 1. / calc_bias(filename, 'deg', data_dict, dump=dump, verbose=verbose-1)
    else:
        print 'unknown bias:', biasname
        exit()


def self_sim_entropy(network, name, out_dir, biases, error_q):
    try:
        if False:
            # network.set_directed(False)
            remove_parallel_edges(network)
            remove_self_loops(network)
        start_time = datetime.datetime.now()
        base_line_type = 'adjacency'
        out_data_dir = out_dir.rsplit('/', 2)[0] + '/data/'
        if not os.path.isdir(out_data_dir):
            os.mkdir(out_data_dir)
        print_prefix = utils.color_string('[' + name + ']')
        mem_cons = list()
        mem_cons.append(('start', utils.get_memory_consumption_in_mb()))
        try:
            com_prop = network.vp['com']
            mod = modularity(network, com_prop)
            print print_prefix + ' newman modularity:', mod
        except KeyError:
            print print_prefix + ' newman modularity:', 'no com mapping'
        adjacency_matrix = adjacency(network)
        name_to_legend = dict()
        name_to_legend['eigenvector'] = '$v_1$'
        name_to_legend['adjacency'] = 'A'
        name_to_legend['betweenness'] = 'B'
        name_to_legend['cosine'] = 'cos'
        name_to_legend['eigenvector_inverse'] = '$v_1^{-1}$'
        name_to_legend['inv_deg'] = '$k^{-1}$'
        name_to_legend['sigma'] = '$\\sigma$'
        name_to_legend['sigma_deg_corrected'] = '$\\sigma_{dc}$'

        deg_map = network.degree_property_map('total')
        if network.gp['type'] == 'empiric':
            dump_base_fn = network.gp['filename']
        else:
            dump_base_fn = 'synthetic'

        entropy_df = pd.DataFrame()
        sort_df = []

        corr_df = pd.DataFrame(columns=['deg'], data=deg_map.a)
        stat_distributions = {}
        network.save(out_dir+name+'.gt')
        data_dict = dict()
        data_dict['net'] = network
        data_dict['adj'] = adjacency(network)
        for bias_name in sorted(biases):
            # calc bias
            bias = calc_bias(dump_base_fn, bias_name, data_dict, dump=network.gp['type'] == 'empiric')

            print print_prefix, '[' + bias_name + ']', '['+str(datetime.datetime.now().replace(
                microsecond=0))+']', 'calc stat dist and entropy rate... ( #v:', network.num_vertices(), ', #e:', network.num_edges(), ')'

            # replace infs and nans with zero
            if bias is not None:
                num_nans = np.isnan(bias).sum()
                num_infs = np.isinf(bias).sum()
                if num_nans > 0 or num_infs > 0:
                    print print_prefix, '[' + bias_name + ']:', utils.color_string(
                        'shape:' + str(bias.shape) + '|replace nans(' + str(num_nans) + ') and infs (' + str(
                            num_infs) + ') of metric with zero', type=utils.bcolor.RED)
                    bias[np.isnan(bias) | np.isinf(bias)] = 0

            assert scipy.sparse.issparse(adjacency_matrix)
            ent, stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, bias)
            stat_distributions[bias_name] = stat_dist
            #print print_prefix, '[' + biasname + '] entropy rate:', ent
            entropy_df.at[0, bias_name] = ent
            sort_df.append((bias_name, ent))
            corr_df[bias_name] = stat_dist
            mem_cons.append(('after ' + bias_name, utils.get_memory_consumption_in_mb()))
        if base_line_type == 'adjacency':
            base_line_abs_vals = stat_distributions['adjacency']
        elif base_line_type == 'uniform':
            base_line_abs_vals = np.array([[1. / network.num_vertices()]])
        else:
            print print_prefix, '[' + bias_name + ']', utils.color_string(('unkown baseline type: ' + base_line_type).upper(),
                utils.bcolors.RED)
            exit()

        #save to df
        pd.DataFrame.from_dict(stat_distributions).to_pickle(out_data_dir + name + '_stat_dists.df')

        #base_line = base_line_abs_vals / 100  # /100 for percent
        base_line = base_line_abs_vals
        vertex_size = network.new_vertex_property('float')
        vertex_size.a = base_line
        min_stat_dist = min([min(i) for i in stat_distributions.values()])
        max_stat_dist = max([max(i) for i in stat_distributions.values()])
        min_val = min([min(i/base_line) for i in stat_distributions.values()])
        max_val = max([max(i/base_line) for i in stat_distributions.values()])
        trapped_df = pd.DataFrame(index=range(network.num_vertices()))

        # calc max vals for graph-coloring
        all_vals = [j for i in stat_distributions.values() for j in i / base_line]
        max_val = np.mean(all_vals) + (2 * np.std(all_vals))
        gini_coef_df = pd.DataFrame()

        pos = None
        # plot all biased graphs and add biases to trapped plot
        for bias_name, stat_dist in sorted(stat_distributions.iteritems(), key=operator.itemgetter(0)):
            stat_dist_diff = stat_dist / base_line
            stat_dist_diff[np.isclose(stat_dist_diff, 1.)] = 1.
            if False:
                if pos is None:
                    print print_prefix, '[' + str(datetime.datetime.now().replace(microsecond=0)) + ']', 'calc graph-layout'
                    try:
                        pos = sfdp_layout(network, groups=network.vp['com'], mu=3.0)
                    except KeyError:
                        pos = sfdp_layout(network)
                plotting.draw_graph(network, color=stat_dist_diff, min_color=min_val, max_color=max_val, sizep=vertex_size,
                                    groups='com', output=out_dir + name + '_graph_' + bias_name, pos=pos)
                plt.close('all')

            # create scatter plot
            if False:
                x = ('stationary value of adjacency', base_line_abs_vals)
                y = (name_to_legend[bias_name] + ' prob. ratio', stat_dist_diff)
                plotting.create_scatter(x=x, y=y, fname=out_dir + name + '_scatter_' + bias_name)

            # plot stationary distribution
            stat_dist_ser = pd.Series(data=stat_dist)

            # calc gini coef and trapped values
            stat_dist_ser.sort(ascending=True)
            stat_dist_ser.index = range(len(stat_dist_ser))
            gcoef = utils.gini_coeff(stat_dist_ser)
            gini_coef_df.at[bias_name, name] = gcoef
            bias_name = name_to_legend[bias_name]
            bias_name += ' $' + ('%.4f' % gcoef) + '$'
            trapped_df[bias_name] = stat_dist_ser.cumsum()
            trapped_df[bias_name] /= trapped_df[bias_name].max()
            mem_cons.append(('after ' + bias_name + ' scatter', utils.get_memory_consumption_in_mb()))

        # add uniform to trapped plot
        if False:
            bias_name = 'unif.'
            uniform = np.array([1]*len(trapped_df))
            gcoef = utils.gini_coeff(uniform)
            gini_coef_df.at[bias_name, name] = gcoef
            # key = name_to_legend[key]
            bias_name += ' $' + ('%.2f' % gcoef) + '$'
            trapped_df[bias_name] = uniform.cumsum()
            trapped_df[bias_name] /= trapped_df[bias_name].max()
        gini_coef_df.to_pickle(out_data_dir + name + '_gini.df')
        trapped_df.index += 1
        trapped_df['idx'] = np.round(np.array(trapped_df.index).astype('float') / len(trapped_df) * 100)

        if len(trapped_df) > 50:
            trapped_df['idx'] = trapped_df['idx'].astype('int')
            trapped_df['idx'] = trapped_df['idx'].apply(lambda x: int(x / 5) * 5)
            trapped_df.drop_duplicates(subset=['idx'], inplace=True)
        matplotlib.rcParams.update({'font.size': 15})
        trapped_df.plot(x='idx', lw=2, alpha=0.9, style=['-o', '-v', '-^', '-s', '-+', '-D', '-<', '->', '-p', '-*', '-x'])
        trapped_df.to_pickle(out_data_dir + name + '_trapped.df')
        plt.yticks([.25, .5, .75, 1], ['25%', '50%', '75%', '100%'])
        xticks = range(0, 101, 20)
        plt.plot([0, 100], [0, 1], ls='--', lw=1, label='unif.', alpha=1.)
        plt.xticks(xticks, map(lambda x: str(x) + '%', xticks))
        plt.xlabel('nodes sorted by stat. dist. value')
        plt.ylim([0, 1])
        plt.ylabel('cum. sum of stat. dist. values in pct. of max.')
        plt.tight_layout()
        plt.savefig(out_dir + name + '_trapped.png')
        plt.close('all')

        sorted_keys, sorted_values = zip(*sorted(sort_df, key=lambda x: x[1], reverse=True))
        if len(set(sorted_values)) == 1:
            sorted_keys = sorted(sorted_keys)
        entropy_df = entropy_df[list(sorted_keys)]
        bar_colors = defaultdict(lambda:'pink')
        bar_colors['adjacency'] = 'lightgray'
        bar_colors['betweenness'] = 'magenta'
        bar_colors['sigma'] = 'darkblue'
        bar_colors['sigma_deg_corrected'] = 'blue'
        bar_colors['cosine'] = 'green'
        bar_colors['eigenvector'] = 'darkred'
        bar_colors['eigenvector_inverse'] = 'red'
        bar_colors['inv_deg'] = 'yellow'
        bar_colors = {idx: bar_colors[key] for idx, key in enumerate(sorted_keys)}
        # print 'bar colors:', bar_colors

        print print_prefix, ' entropy rates:\n', entropy_df
        matplotlib.rcParams.update({'font.size': 15})
        #entropy_df.columns = [i.replace('_', ' ') for i in entropy_df.columns]
        ax = entropy_df.plot(kind='bar', color=bar_colors, alpha=0.9)
        entropy_df.to_pickle(out_data_dir + name + '_entropy.df')
        min_e, max_e = entropy_df.loc[0].min(), entropy_df.loc[0].max()
        ax.set_ylim([min_e * 0.95, max_e * 1.01])
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(True)
        #ax.spines['right'].set_visible(True)
        plt.ylabel('entropy rate')
        plt.legend(loc='upper left')
        plt.xlim([-1.1, 0.3])
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tight_layout()
        plt.savefig(out_dir + name + '_entropy_rates.png')
        plt.close('all')
        mem_df = pd.DataFrame(columns=['state', 'memory in MB'], data=mem_cons)
        mem_df.plot(x='state', y='memory in MB', rot=45, label='MB')
        plt.title('memory consumption')
        plt.tight_layout()
        plt.savefig(out_dir + name + '_mem_status.png')
        plt.close('all')
        print print_prefix, utils.color_string('>>all done<< duration: ' + str(datetime.datetime.now() - start_time),
                                               type=utils.bcolors.GREEN)
        results = dict()
        results['gini'] = gini_coef_df
        return results
    except:
        error_msg = str(traceback.format_exc())
        print error_msg
        if error_q is not None and isinstance(error_q, mp.Queue):
            error_q.put((name, error_msg))
        else:
            exit()
        with open(out_dir + name + '_error.log', 'w') as f:
            f.write(str(datetime.datetime.now()).center(100, '=') + '\n')
            f.write(error_msg + '\n')
        return None


if __name__ == '__main__':
    pass