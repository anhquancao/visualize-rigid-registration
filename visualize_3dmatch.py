from lib import plot_precision_recall_curves, analyze_by_pair
import numpy as np
import matplotlib.pyplot as plt

DGR_3DMatch = np.load("results.npz")
method_names = DGR_3DMatch['names']
stats = DGR_3DMatch['stats']
selected_method_ids = [0, 1, 6, 7]
method_names = list(method_names[selected_method_ids])
stats = stats[selected_method_ids]

# DGR
DGR_rerun_names = [
    # ('DGR', 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp.npz'),
    ('DGR w/o safeguard', 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard.npz'),
    (r"DGR$^\dagger$", 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard_optim.npz')
]
for name, path in DGR_rerun_names:
    method_names.append(name)
    f = np.load(path)
    stats = np.concatenate( (stats, f['stats']), axis=0)

# Our method
our_methods_data = [
    ('Ours', 'data/recall_curves_3dmatch_no_icp_threshold/results.npy'),
]

for name, path in our_methods_data:
    method_names.append(name)
    g = np.load(path)

    # print(stats.shape)
    # print(g.shape)
    # print(g['results'].reshape(-1, 555, 5).mean(1).mean(0))
    stats = np.concatenate((stats, g.reshape(-1, 1623, 5).mean(0, keepdims=True)), axis=0)



cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, len(method_names))]
colors.reverse()
plot_precision_recall_curves(stats,
                             method_names,
                             rre_precisions=np.arange(0, 15, 0.05),
                             rte_precisions=np.arange(0, 0.3, 0.005),
                             output_postfix='3dmatch',
                             cmap=colors, figsize=(9, 3.3), aspect=4.0)
