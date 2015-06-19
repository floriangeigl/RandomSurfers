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

#bias_factors_df = create_bf_scatters_from_df(stat_dist_df, 'adj', ['click_sub', 'page_counts'], output_folder=base_dir + 'bf_scatter/')
#create_scatters_from_df(stat_dist_df, ['adj', 'click_sub', 'page_counts'], output_folder=base_dir + 'scatter/')
create_ginis_from_df(stat_dist_df, ['adj', 'click_sub', 'page_counts'], output_folder=base_dir + 'gini/', lw=3, ms=15,
                     font_size=15)

bias_factors_df['url'] = stat_dist_df['url']
print bias_factors_df.sort('click_sub', ascending=False)[['click_sub','url']].head()
print bias_factors_df.sort('page_counts', ascending=False)[['page_counts','url']].head()
