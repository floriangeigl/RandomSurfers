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