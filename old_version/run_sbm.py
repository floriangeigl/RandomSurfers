from __future__ import division
from sys import platform as _platform

import matplotlib


if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from old_version import optimizer, moves, cost_function
from tools.basics import *
from utils import *
import datetime
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool as Pool
from collections import defaultdict
import traceback


def main():
    # total number of nodes per network
    nodes = 100

    # divide network into x groups
    groups = 3

    # stepsize between different block configurations
    step = 0.25

    # max connection between blocks
    max_con = 1

    # reduce targets of missions by
    target_reduce = 1

    # reduce source per target by
    source_reduce = 1

    # max runs for optimizer
    runs_per_temp = 1000

    # init beta
    beta = 1000

    # percentages of which to calc the correlation between rankings (top x)
    correlation_perc = np.arange(0.1, 1.1, 0.1)

    # cpus used for cost calculations
    cpus = 10

    correlation_results = dict()
    for perc in correlation_perc:
        correlation_results[perc] = pd.DataFrame()
        correlation_results[perc].index = correlation_results[perc].index.astype('float')
    assert groups > 0
    process_results = []
    my_pool = Pool(10)
    num_results = 0
    try:
        for network_num, self_con in enumerate(np.arange(0, max_con + (step * 0.9), step)):
            num_results += 1
            my_pool.apply_async(func=run_optimization, args=(network_num, self_con, max_con, groups, nodes, target_reduce, source_reduce, ranking_weights_func, runs_per_temp, beta), callback=process_results.append)
            # run_optimization
    except KeyboardInterrupt:
        my_pool.terminate()
    my_pool.close()
    my_pool.join()
    if len(process_results) != num_results or any(j is None for i in process_results for j in i):
        print 'multiprocessing error. exit'
        exit()

    print 'calc correlations'
    for self_con, df in process_results:
        for perc in correlation_perc:
            print perc,
            tmp_df = df.loc[0:int(round((len(df) - 1) * perc))]
            correlations = tmp_df.corr(method='spearman')
            correlation_results[perc].at[self_con, 'deg'] = correlations['deg']['ranked_vertex']
            correlation_results[perc].at[self_con, 'pagerank'] = correlations['pagerank']['ranked_vertex']
            correlation_results[perc].at[self_con, 'betweeness'] = correlations['betweeness']['ranked_vertex']
    print 'plot correlations'
    for perc in correlation_perc:
        df = correlation_results[perc]
        df.sort(inplace=True)
        df.plot(lw=3)
        plt.xlabel('self connectivity')
        plt.savefig('output/sbm_correlation_top_' + str(perc).ljust(4, '0').replace('.','_') + '.png', dpi=300)
        plt.close('all')


class Printer():
    def __init__(self, self_con):
        self.self_con = self_con

    def print_f(self, *args):
        print '[self-con: ' + str(self.self_con) + ']',
        print ' '.join(map(str, args))


def run_optimization(network_num, self_con, max_con, groups, nodes, target_reduce, source_reduce, ranking_weights_func, runs_per_temp, beta):
    # generate network
    try:
        p = Printer(self_con)
        self_con = max_con - self_con
        other_con = (max_con - self_con) / (groups - 1)
        p.print_f('start optimization with self-con:', self_con)
        p.print_f('gen graph with ', nodes, 'nodes and', groups, 'groups.(self:', self_con, ',other:', other_con, ')')
        network, bm_groups = graph_gen(self_con, other_con, nodes, groups)
        network.vp['groups'] = bm_groups
        create_folder_structure('output/data/')
        network.save('output/data/network_sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.gt')

        # init cost function and print configuration details
        cf = cost_function.CostFunction(network, target_reduce=target_reduce, source_reduce=source_reduce, ranking_weights=[ranking_weights_func(i) for i in reversed(range(network.num_vertices()))], verbose=0)  # , cpus=cpus)
        if network_num == 0:
            plt.clf()
            plt.plot(cf.ranking_weights, lw=4, label='ranking weights')
            plt.xlabel('ranking position')
            plt.ylabel('weight')
            plt.title('weights of ranking')
            plt.savefig('output/sbm_results_rank_weights.png')
            plt.close('all')

            # create dict to compare costs
        measurements_costs = dict()
        measurements = dict()

        # get ranking of different measurements
        p.print_f('calc cost of measurements rankings')
        start = datetime.datetime.now()
        df = pd.DataFrame()
        deg_pmap = network.degree_property_map('total')
        measurements['deg'] = deg_pmap
        df['deg'] = get_ranking(deg_pmap)
        measurements_costs['deg'] = cf.calc_cost(list(df['deg']))
        if network.is_directed():
            vp_map = network.degree_property_map('in')
            df['in-deg'] = get_ranking(vp_map)
            vp_map = network.degree_property_map('out')
            df['out-deg'] = get_ranking(vp_map)
        pr_pmap = pagerank(network)
        measurements['page rank'] = pr_pmap
        df['pagerank'] = get_ranking(pr_pmap)
        measurements_costs['pagerank'] = cf.calc_cost(list(df['pagerank']))
        bw_pmap, _ = betweenness(network)
        measurements['betweeness'] = bw_pmap
        df['betweeness'] = get_ranking(bw_pmap)
        measurements_costs['betweeness'] = cf.calc_cost(list(df['betweeness']))
        clust = local_clustering(network)
        measurements['clustering coef.'] = clust
        df['clustering coef.'] = get_ranking(clust)
        measurements_costs['clustering coef.'] = cf.calc_cost(list(df['clustering coef.']))
        avg_time_per_cf = datetime.timedelta(microseconds=(datetime.datetime.now() - start).microseconds / 4)
        p.print_f('avg calc cost time:', avg_time_per_cf)
        p.print_f('calc cost time per', runs_per_temp, ':', datetime.timedelta(microseconds=avg_time_per_cf.microseconds * runs_per_temp))

        # take best of measurement rankings as init ranking
        init_ranking, init_cost = max(measurements_costs.iteritems(), key=lambda x: x[1])
        init_ranking = list(df[init_ranking])
        for key, val in sorted(measurements_costs.iteritems(), key=lambda x: x[1]):
            p.print_f('cost', key, ':', val)

        mover = moves.MoveTravelSM(verbose=0)

        # optimize ranking
        p.print_f('optimizing...')
        opt = optimizer.SimulatedAnnealing(cf, mover, init_ranking, runs_per_temp=runs_per_temp, beta=beta, verbose=0)
        ranking, cost = opt.optimize()

        # print accept prob
        accept_prob_df = pd.DataFrame(columns=['accept prob'], data=opt.prob_history)
        accept_prob_df['rolling mean'] = pd.rolling_mean(accept_prob_df['accept prob'], window=100)
        create_folder_structure('output/prob/')
        accept_prob_df.plot(lw=1)
        prob_min, prob_max = accept_prob_df['accept prob'].min(), accept_prob_df['accept prob'].max()
        prob_quad = prob_min + ((prob_max - prob_min) / 4)
        for run, betaval in opt.beta_history.iteritems():
            plt.annotate(str(betaval), xy=(run, prob_quad), rotation=90)
        plt.legend()
        plt.savefig('output/prob/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.png', dpi=150)
        plt.close('all')

        # plot measurements of ranking
        create_folder_structure('output/measurements_of_ranking/')
        plot_measurements_of_ranking(ranking, measurements, filename='output/measurements_of_ranking/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.png', logx=False)

        # make sure move did not exclude some vals
        assert len(ranking) == network.num_vertices()
        assert set(ranking) == set(map(int, network.vertices()))

        # create ranking dataframe
        create_folder_structure('output/data/')
        ranking_df = get_ranking_df(ranking, cf.ranking_weights)
        ranking_df.to_pickle('output/data/ranking_sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.df')
        for key in ranking_df.columns:
            df[key] = ranking_df[key]
        test_dict = {network.vertex(v): idx for idx, v in enumerate(reversed(ranking))}
        vp_map = network.new_vertex_property('int')
        for v in network.vertices():
            vp_map[v] = test_dict[v]
        df['test'] = get_ranking(vp_map)
        df.to_pickle('output/data/ranking_and_cmeasures_sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.df')

        # print some stats
        p.print_f('runs:', opt.runs)
        p.print_f('cost:', cost)
        p.print_f('cost improvement:', ((cost / init_cost) - 1) * 100, '%')
        # print df.head()
        # print 'ranking top 50', ranking[:50]

        # plot cost history
        create_folder_structure('output/cost/')
        opt.draw_cost_history(filename='output/cost/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.png', compare_dict=measurements_costs)

        # plot networks
        create_folder_structure('output/graph_plots/')
        eweight = network.new_edge_property('float')
        for e in network.edges():
            eweight[e] = 1 if bm_groups[e.source()] == bm_groups[e.target()] else 0
        pos = sfdp_layout(network, groups=bm_groups, mu=1)
        graph_draw(network, pos=pos, vertex_size=prop_to_size(deg_pmap), vertex_fill_color=bm_groups, output='output/graph_plots/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '.png')
        graph_draw(network, pos=pos, vertex_size=prop_to_size(deg_pmap), vertex_fill_color=deg_pmap, output='output/graph_plots/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '_deg.png')
        graph_draw(network, pos=pos, vertex_size=prop_to_size(deg_pmap), vertex_fill_color=bw_pmap, output='output/graph_plots/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '_betwe.png')

        # plot degree distribution
        deg_dist = defaultdict(float)
        for v in network.vertices():
            deg_dist[deg_pmap[v]] += 1
        x_axis = range(max(deg_dist.keys()) + 1)
        plt.plot([deg_dist[i] for i in x_axis], lw=2)
        plt.title('degree distribution')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('degree')
        plt.ylabel('# nodes')
        plt.savefig('output/graph_plots/sbm_' + str(self_con).ljust(4, '0').replace('.','_') + '_degdist.png')
        plt.close('all')
        return self_con, df
    except:
        print traceback.print_exc()
        return None, None


def ranking_weights_func(x):
    return np.power(x, 4)


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

