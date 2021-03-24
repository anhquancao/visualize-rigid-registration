from lib import plot_precision_recall_curves, analyze_by_pair
import numpy as np
import matplotlib.pyplot as plt

method_names = []
stats = np.empty((0, 555, 5))
# DGR
DGR_data = [
    # ('DGR w/o safeguard, w/o optim', 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard_optim.npz'),
    (r"DGR + $\phi$", 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard_optim.npz'),
    (r"DGR + $\phi$ + Opt.", 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp.npz'),
    # ('DGR w/o safeguard', 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard.npz'),
    #('DGR + ICP', 'data/dgr_results_for_cvpr/dgr_kitti_with_icp.npz'),
]


for name, path in DGR_data:
    method_names.append(name)
    f = np.load(path)
    stats = np.concatenate( (stats, f['stats'][np.newaxis, ]), axis=0)


# Our method
our_methods_data = [
    ('PCAM-Sparse', 'data/recall_curves_kitti_PCAM_sparse.npz'),
    ('PCAM-Soft', 'data/recall_curves_PCAM_soft.npz'),
]

for name, path in our_methods_data:
    method_names.append(name)
    g = np.load(path)

    # print(stats.shape)
    # print(g['results'].mean(0, keepdims=True).shape)
    # print(g['results'].reshape(-1, 555, 5).mean(1).mean(0))
    stats = np.concatenate((stats, g['results'].reshape(-1, 555, 5).mean(0, keepdims=True)), axis=0)

    # stats = np.concatenate((stats, g['results'][1, 1, :, :][np.newaxis, ]), axis=0)



cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, len(method_names))]
colors.reverse()
plot_precision_recall_curves(stats,
                             method_names,
                             rre_precisions=np.arange(0, 4.9, 0.05),
                             rte_precisions=np.arange(0, 0.6, 0.005),
                             output_postfix='kitti',
                             cmap=colors)
