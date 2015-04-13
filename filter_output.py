__author__ = 'Florian Geigl'
import os
input_dir = './output/'
output_dir = './output/paper/'
cmd = 'cp'
steady_state_dir = output_dir + 'steady_state/'
ent_rate_dir = output_dir + 'entropy_rates/'
steady_state_hist_dir = output_dir + 'steady_state_hist_dir/'
os.system('mkdir -p ' + input_dir)
os.system('mkdir -p ' + output_dir)
os.system('mkdir -p ' + steady_state_dir)
os.system('mkdir -p ' + ent_rate_dir)
os.system('mkdir -p ' + steady_state_hist_dir)

graphs = ['weak', 'strong', 'price_net', 'facebook', 'wiki4schools']
properties = ['adjacency', 'betweenness', 'eigenvector', 'eigenvector_inverse', 'sigma', 'sigma_deg_corrected']
for g in graphs:
    for p in properties:
        os.system(cmd + ' ' + input_dir + '*' + g + '*graph_' + p + '* ' + steady_state_dir)
        os.system(cmd + ' ' + input_dir + '*' + g + '*stat_dist_' + p + '* ' + steady_state_hist_dir)
    os.system(cmd + ' ' + input_dir + '*' + g + '*entropy_rate* ' + ent_rate_dir)