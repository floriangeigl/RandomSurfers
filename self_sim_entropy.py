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
from data_io import *

def calc_bias(filename, biasname, data_dict, dump=True, verbose=1):
    dump_filename = filename + '_' + biasname + '.bias'
    name = filename.rsplit('/', 1)[-1].replace('.gt', '')
    if verbose > 0:
        print utils.color_string('[' + name + ']'), '[' + biasname + ']', '[' + str(
            datetime.datetime.now().replace(microsecond=0)) + ']', 'calc bias'
    loaded = False
    #################################
    if biasname == 'adjacency':
        return None
    #################################
    elif biasname == 'eigenvector':
        try:
            loaded_data = try_load(dump_filename)
            A_eigvalue = np.float64(loaded_data[0])
            A_eigvector = loaded_data[1:]
            data_dict['eigval'] = A_eigvalue
            data_dict['eigvec'] = A_eigvector
            loaded = True
        except IOError:
            try:
                A_eigvector = data_dict['eigvec']
                A_eigvalue = data_dict['eigval']
                loaded = True
            except KeyError:
                A_eigvalue, A_eigvector = eigenvector(data_dict['net'])
                A_eigvalue = np.float64(A_eigvalue)
                A_eigvector = np.array(A_eigvector.a)
                data_dict['eigval'] = A_eigvalue
                data_dict['eigvec'] = A_eigvector
        dump_data = np.concatenate((np.array([A_eigvalue]), A_eigvector))
        if dump and not loaded:
            try_dump(dump_data, dump_filename)
        return A_eigvector
    #################################
    elif biasname == 'eigenvector_inverse':
        try:
            A_eigvector_inf = try_load(dump_filename)
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
    #################################
    elif biasname == 'inv_log_eigenvector':
        try:
            A_inf_log_eigvector = try_load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvector = data_dict['eigvec']
            except KeyError:
                A_eigvector = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
            A_inf_log_eigvector = 1. / np.log(A_eigvector + 2)
        if dump and not loaded:
            try_dump(A_inf_log_eigvector, dump_filename)
        return A_inf_log_eigvector
    #################################
    elif biasname == 'inv_sqrt_eigenvector':
        try:
            A_sqrt_log_eigvector = try_load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvector = data_dict['eigvec']
            except KeyError:
                A_eigvector = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
            A_sqrt_log_eigvector = 1. / np.sqrt(A_eigvector)
            A_sqrt_log_eigvector[np.invert(np.isfinite(A_sqrt_log_eigvector))] = 0.
        if dump and not loaded:
            try_dump(A_sqrt_log_eigvector, dump_filename)
        return A_sqrt_log_eigvector
    #################################
    elif biasname == 'sigma':
        try:
            sigma = try_load(dump_filename)
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
    #################################
    elif biasname == 'sigma_deg_corrected':
        try:
            sigma_deg_cor = try_load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvalue = data_dict['eigval']
            except KeyError:
                _ = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
                A_eigvalue = data_dict['eigval']
            sigma_deg_cor = network_matrix_tools.katz_sim_network(data_dict['adj'], largest_eigenvalue=A_eigvalue,
                                                                  norm=np.array(
                                                                      data_dict['net'].degree_property_map('total').a))
        if dump and not loaded:
            try_dump(sigma_deg_cor, dump_filename)
        return sigma_deg_cor
    #################################
    elif biasname == 'sigma_log_deg_corrected':
        try:
            sigma_log_deg_cor = try_load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvalue = data_dict['eigval']
            except KeyError:
                _ = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
                A_eigvalue = data_dict['eigval']
            sigma_log_deg_cor = network_matrix_tools.katz_sim_network(data_dict['adj'], largest_eigenvalue=A_eigvalue,
                                                                      norm=np.log(np.array(
                                                                          data_dict['net'].degree_property_map(
                                                                              'total').a, dtype=np.float) + 2))
        if dump and not loaded:
            try_dump(sigma_log_deg_cor, dump_filename)
        return sigma_log_deg_cor
    #################################
    elif biasname == 'sigma_sqrt_deg_corrected':
        try:
            sigma_sqrt_deg_cor = try_load(dump_filename)
            loaded = True
        except IOError:
            try:
                A_eigvalue = data_dict['eigval']
            except KeyError:
                _ = calc_bias(filename, 'eigenvector', data_dict, dump=dump, verbose=verbose-1)
                A_eigvalue = data_dict['eigval']
            sigma_sqrt_deg_cor = network_matrix_tools.katz_sim_network(data_dict['adj'], largest_eigenvalue=A_eigvalue,
                                                                      norm=np.sqrt(np.array(
                                                                          data_dict['net'].degree_property_map(
                                                                              'total').a, dtype=np.float)))
        if dump and not loaded:
            try_dump(sigma_sqrt_deg_cor, dump_filename)
        return sigma_sqrt_deg_cor
    #################################
    elif biasname == 'cosine':
        try:
            cos = try_load(dump_filename)
            loaded = True
        except IOError:
            cos = network_matrix_tools.calc_cosine(data_dict['adj'], weight_direct_link=True)
        if dump and not loaded:
            try_dump(cos, dump_filename)
        return cos
    #################################
    elif biasname == 'betweenness':
        try:
            bet = try_load(dump_filename)
            loaded = True
        except IOError:
            bet = np.array(betweenness(data_dict['net'])[0].a)
        if dump and not loaded:
            try_dump(bet, dump_filename)
        return bet
    #################################
    elif biasname == 'deg':
        return np.array(data_dict['net'].degree_property_map('total').a, dtype=np.float)
    #################################
    elif biasname == 'inv_deg':
        return 1. / (calc_bias(filename, 'deg', data_dict, dump=dump, verbose=verbose - 1) + 1)
    #################################
    elif biasname == 'inv_log_deg':
        return 1. / np.log(calc_bias(filename, 'deg', data_dict, dump=dump, verbose=verbose - 1) + 2)
    #################################
    elif biasname == 'inv_sqrt_deg':
        return 1. / np.sqrt(calc_bias(filename, 'deg', data_dict, dump=dump, verbose=verbose - 1))
    #################################
    else:
        try:
            return try_load(biasname)
        except:
            try:
                return try_load(dump_filename)
            except:
                print 'unknown bias:', biasname
                exit()


def self_sim_entropy(network, name, out_dir, biases, error_q, method):
    try:
        if True:
            # network.set_directed(False)
            # remove_parallel_edges(network)
            remove_self_loops(network)
        start_time = datetime.datetime.now()
        base_line_type = 'adjacency'
        out_data_dir = out_dir.rsplit('/', 2)[0] + '/data/'
        if not os.path.isdir(out_data_dir):
            os.mkdir(out_data_dir)
        name = name.rsplit('/', 1)[-1]
        print_prefix = utils.color_string('[' + name + ']')
        # mem_cons = list()
        # mem_cons.append(('start', utils.get_memory_consumption_in_mb()))
        com_prop = None
        try:
            com_prop = network.vp['com']
        except KeyError:
            try:
                com_prop = network.vp['category']
            except KeyError:
                pass
        if com_prop is not None:
            if not isinstance(com_prop[network.vertex(0)], int):
                # convert categories-names to int
                tmp_prop = network.new_vertex_property('int')
                coms = dict()
                for v in network.vertices():
                    v_com = com_prop[v]
                    try:
                        com_id = coms[v_com]
                    except KeyError:
                        com_id = len(coms)
                        coms[v_com] = com_id
                    tmp_prop[v] = com_id
                com_prop = tmp_prop

            mod = modularity(network, com_prop)
            print print_prefix + ' newman modularity:', mod
        else:
            print print_prefix + ' newman modularity:', 'no com mapping (', sorted(network.vp.keys()), ')'
        adjacency_matrix = adjacency(network)

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
        skip_bias = None
        for bias in biases:
            if isinstance(bias, str):
                # calc bias
                bias_name = bias.rsplit('/', 1)[-1]
                bias = calc_bias(dump_base_fn, bias, data_dict, dump=network.gp['type'] == 'empiric')
            else:
                bias_name, bias = bias
            if skip_bias is not None:
                if bias_name.startswith(skip_bias):
                    print print_prefix, '[' + bias_name + ']', '[' + str(datetime.datetime.now().replace(
                        microsecond=0)) + ']', 'skip'
                    continue
                else:
                    skip_bias = None

            print print_prefix, '[' + bias_name + ']', '['+str(datetime.datetime.now().replace(
                microsecond=0))+']', 'calc stat dist and entropy rate... ( #v:', network.num_vertices(), ', #e:', network.num_edges(), ')'

            # replace infs and nans with zero
            if bias is not None:
                try:
                    num_nans = np.isnan(bias).sum()
                    num_infs = np.isinf(bias).sum()
                    if num_nans > 0 or num_infs > 0:
                        print print_prefix, '[' + bias_name + ']:', utils.color_string(
                            'shape:' + str(bias.shape) + '|replace nans(' + str(num_nans) + ') and infs (' + str(
                                num_infs) + ') of metric with zero', type=utils.bcolors.RED)
                        bias[np.isnan(bias) | np.isinf(bias)] = 0
                except TypeError:
                    assert scipy.sparse.issparse(bias)
                    num_nans = np.isnan(bias.data).sum()
                    num_infs = np.isinf(bias.data).sum()
                    if num_nans > 0 or num_infs > 0:
                        print print_prefix, '[' + bias_name + ']:', utils.color_string(
                            'shape:' + str(bias.shape) + '|replace nans(' + str(num_nans) + ') and infs (' + str(
                                num_infs) + ') of metric with zero', type=utils.bcolor.RED)
                        bias.data[np.isnan(bias.data) | np.isinf(bias.data)] = 0

            assert scipy.sparse.issparse(adjacency_matrix)
            try:
                ent, stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, bias, method=method,
                                                                                 print_prefix=print_prefix + ' [' + bias_name + '] ',
                                                                                 smooth_bias=False,
                                                                                 calc_entropy_rate=False)
                stat_distributions[bias_name] = stat_dist
                #print print_prefix, '[' + biasname + '] entropy rate:', ent
                entropy_df.at[0, bias_name] = ent
                sort_df.append((bias_name, ent))
                corr_df[bias_name] = stat_dist
            except:
                print traceback.format_exc()
                skip_bias = bias_name.split('_cs', 1)[0]
            del bias

            # mem_cons.append(('after ' + bias_name, utils.get_memory_consumption_in_mb()))
        if base_line_type == 'adjacency':
            base_line_abs_vals = stat_distributions['adjacency']
        elif base_line_type == 'uniform':
            base_line_abs_vals = np.array([[1. / network.num_vertices()]])
        else:
            print print_prefix, '[' + bias_name + ']', utils.color_string(('unkown baseline type: ' + base_line_type).upper(),
                utils.bcolors.RED)
            exit()
        if 'category' in network.vp.keys():
            print 'add categories to stationary distribution'
            cat_pmap = network.vp['category']
            stat_distributions['category'] = [cat_pmap[v] for v in network.vertices()]
        #save to df
        pd.DataFrame.from_dict(stat_distributions).to_pickle(out_data_dir + name + '_stat_dists.df')
        if 'category' in stat_distributions:
            del stat_distributions['category']
        trapped_df = pd.DataFrame(index=range(network.num_vertices()))
        gini_coef_df = pd.DataFrame()

        # base_line = base_line_abs_vals / 100  # /100 for percent
        base_line = base_line_abs_vals
        vertex_size = network.new_vertex_property('float')
        vertex_size.a = base_line
        # min_stat_dist = min([min(i) for i in stat_distributions.values()])
        # max_stat_dist = max([max(i) for i in stat_distributions.values()])
        min_val = min([min(i/base_line) for i in stat_distributions.values()])
        # max_val = max([max(i/base_line) for i in stat_distributions.values()])

        # calc max vals for graph-coloring
        all_vals = [j for i in stat_distributions.values() for j in i / base_line]
        max_val = np.mean(all_vals) + (2 * np.std(all_vals))

        pos = None
        # plot all biased graphs and add biases to trapped plot
        for bias_name, stat_dist in sorted(stat_distributions.iteritems(), key=operator.itemgetter(0)):
            stat_dist_diff = stat_dist / base_line
            stat_dist_diff[np.isclose(stat_dist_diff, 1.)] = 1.
            if True and network.num_vertices() < 100:
                if pos is None:
                    print print_prefix, '[' + str(datetime.datetime.now().replace(microsecond=0)) + ']', 'calc graph-layout'
                    try:
                        pos = sfdp_layout(network, groups=network.vp['com'], mu=3.0)
                    except KeyError:
                        pos = sfdp_layout(network)
                plotting.draw_graph(network, color=stat_dist_diff, min_color=min_val, max_color=max_val,
                                    sizep=vertex_size,
                                    groups='com', output=out_dir + name + '_graph_' + bias_name.split('/')[-1], pos=pos)
                plt.close('all')
            else:
                print print_prefix, 'skip draw graph', '#v:', network.num_vertices()

            # plot stationary distribution
            stat_dist_ser = pd.Series(data=stat_dist)

            # calc gini coef and trapped values
            stat_dist_ser.sort(ascending=True)
            stat_dist_ser.index = range(len(stat_dist_ser))
            gcoef = utils.gini_coeff(stat_dist_ser)
            gini_coef_df.at[bias_name, name] = gcoef
            # bias_name = name_to_legend[bias_name]
            bias_name += ' $' + ('%.4f' % gcoef) + '$'
            trapped_df[bias_name] = stat_dist_ser.cumsum()
            trapped_df[bias_name] /= trapped_df[bias_name].max()
            # mem_cons.append(('after ' + bias_name + ' scatter', utils.get_memory_consumption_in_mb()))

        gini_coef_df.to_pickle(out_data_dir + name + '_gini.df')
        trapped_df.to_pickle(out_data_dir + name + '_trapped.df')

        sorted_keys, sorted_values = zip(*sorted(sort_df, key=lambda x: x[1], reverse=True))
        if len(set(sorted_values)) == 1:
            sorted_keys = sorted(sorted_keys)
        entropy_df = entropy_df[list(sorted_keys)]

        print print_prefix, ' entropy rates:\n', entropy_df
        entropy_df.to_pickle(out_data_dir + name + '_entropy.df')

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