import pandas as pd
from plotting import create_bf_scatters_from_df, create_scatters_from_df, create_ginis_from_df
pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)

base_dir = '/home/fgeigl/navigability_of_networks/output/iknow/'
stationary_dist_fn = base_dir + 'stationary_dist.df'
entropy_rate_fn = base_dir + 'entropy_rate.df'

stat_dist_df = pd.read_pickle(stationary_dist_fn)
entropy_rate_df = pd.read_pickle(entropy_rate_fn)
print stat_dist_df.columns
adj_name = 'Uniform Random Surfer'
clickstream_name = 'Pragmatic Random Surfer'
page_views_name = 'Lateral Random Surfer'
url_name = 'URL'
stat_dist_df.columns = [adj_name, clickstream_name, page_views_name, url_name]
print stat_dist_df.columns
print entropy_rate_df.columns

bias_factors_df = create_bf_scatters_from_df(stat_dist_df, clickstream_name, [page_views_name], output_folder=base_dir + 'bf_scatter/')
min_y, max_y = bias_factors_df[bias_factors_df > 0].min().min(), bias_factors_df.max().max()
bias_factors_df = create_bf_scatters_from_df(stat_dist_df, adj_name, [clickstream_name, page_views_name], output_folder=base_dir + 'bf_scatter/')
min_y, max_y = min(min_y, bias_factors_df[bias_factors_df > 0].min().min()), max(max_y, bias_factors_df.max().max())

create_bf_scatters_from_df(stat_dist_df, clickstream_name, [page_views_name], output_folder=base_dir + 'bf_scatter/',
                           y_range=[min_y, max_y])
bias_factors_df = create_bf_scatters_from_df(stat_dist_df, adj_name, [clickstream_name, page_views_name],
                                             output_folder=base_dir + 'bf_scatter/', y_range=[min_y, max_y])

create_scatters_from_df(stat_dist_df, [adj_name, clickstream_name, page_views_name], output_folder=base_dir + 'scatter/')
create_ginis_from_df(stat_dist_df, [adj_name, clickstream_name, page_views_name], output_folder=base_dir + 'gini/', lw=3, ms=15,
                     font_size=15)

bias_factors_df[url_name] = stat_dist_df[url_name]
print bias_factors_df.sort(clickstream_name, ascending=False)[[clickstream_name, url_name]].head()
print bias_factors_df.sort(page_views_name, ascending=False)[[page_views_name, url_name]].head()


