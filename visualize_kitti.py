from lib import plot_precision_recall_curves, analyze_by_pair
import numpy as np
import matplotlib.pyplot as plt

method_names = []
stats = np.empty((0, 555, 5))
# DGR
DGR_data = [
    ('DGR', 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp.npz'),
    # ('DGR w/o safeguard', 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard.npz'),
    ('DGR w/o safeguard, w/o optim', 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard_optim.npz'),
    ('DGR + ICP', 'data/dgr_results_for_cvpr/dgr_kitti_with_icp.npz'),
]


for name, path in DGR_data:
    method_names.append(name)
    f = np.load(path)
    stats = np.concatenate( (stats, f['stats'][np.newaxis, ]), axis=0)

# Our method
our_methods_data = [
    ('Ours', 'data/ours_recall_curves_kitti/recall_curves_kitti_no_icp.npz'),
    ('Ours + ICP', 'data/ours_recall_curves_kitti/recall_curves_kitti_icp.npz'),
]

our_method_rs, our_method_ts = [], []
our_method_recall_rs, our_method_recall_ts = [], []
our_method_names = []

for name, path in our_methods_data:
    our_method_names.append(name)
    g = np.load(path)

    our_method_ts.append(g['t_list'])
    our_method_recall_ts.append(g['recall_vs_t'].mean(1))

    our_method_rs.append(g['r_list'])
    our_method_recall_rs.append(g['recall_vs_r'].mean(1))



cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, len(method_names) + len(our_method_names))]
colors.reverse()
plot_precision_recall_curves(stats,
                             method_names,
                             rre_precisions=np.arange(0, 5, 0.05),
                             rte_precisions=np.arange(0, 0.6, 0.005),
                             output_postfix='kitti',
                             cmap=colors,
                             our_method_names=our_method_names,
                             our_method_rs=our_method_rs, our_method_ts=our_method_ts,
                             our_methods_recall_rs=our_method_recall_rs, our_methods_recall_ts=our_method_recall_ts)
