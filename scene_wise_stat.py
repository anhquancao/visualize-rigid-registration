from lib import plot_scene_wise_stats, analyze_by_scene
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

ours_1_data = np.load("data/scene_stats/recall_curves_3dmatch_no_icp_245_None_1_with_scene_id.npz")['results']
d = np.load("data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp.npz")
methods = [
    # {
    #     "name": "DGR",
    #     "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp.npz',
    #     "is_our": False,
    # },
    {
        "name": r"DGR + $\phi$",
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard_optim.npz',
        "is_our": False,
    },
    {
        "name": r'DGR + $\phi$ + Opt.',
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard.npz',
        "is_our": False,
    },
    {
        "name": r'DGR + $\phi$ + Opt. + Saf.',
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp.npz',
        "is_our": False,
    },

    {
        "name": r'PCAM-Sparse + $\phi$',
        "path_3d": 'data/recall_curves_3dmatch_PCAM_Sparse_Phi.npz',
        "is_our": True,
    },
    {
        "name": r'PCAM-Sparse + $\phi$ + Opt.',
        "path_3d": 'data/recall_curves_3dmatch_PCAM_Sparse_Phi_Optim.npz',
        "is_our": True,
    },
    {
        "name": r'PCAM-Sparse + $\phi$ + Opt. + Saf.',
        "path_3d": 'data/recall_curves_3dmatch_PCAM_Sparse_Phi_Optim_Safeguard.npz',
        "is_our": True,
    },

]

scene_names = [
    'Kitchen', 'Home1', 'Home2', 'Hotel1', 'Hotel2', 'Hotel3', 'Study', 'Lab'
]
list_stats = []
method_names = []
for method in methods:
    if method['is_our']:
        g = np.load(method['path_3d'])
        # print(g['results'].reshape(-1, 1623, 5).shape)
        stats = g['results'].reshape(-1, 1623, 5).mean(0, keepdims=False)
    else:
        f = np.load(method['path_3d'])
        stats = np.squeeze(f['stats'])
    # print(method['name']) 
    # print(stats.mean(0))
    list_stats.append(stats[np.newaxis,])
    method_names.append(method['name'])

stats = np.concatenate(list_stats, axis=0)

cmap = plt.get_cmap('tab20').colors
colors = [cmap[0], cmap[8], cmap[4], cmap[12], cmap[6], cmap[2]]
scene_wise_stats = analyze_by_scene(stats,
                                    range(len(scene_names)),
                                    rte_thresh=0.3,
                                    rre_thresh=15)
plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'Recall', (0.0, 1.0), "_scene_wise", colors, "Recall")
plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'TE (m)', (0.0, 0.12), "_scene_wise", colors, 'TE (m)')
plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'RE (deg)', (0.0, 4.0), "_scene_wise", colors, 'RE (deg)')

scene_wise_stats = analyze_by_scene(stats,
                                    range(len(scene_names)),
                                    rte_thresh=np.inf,
                                    rre_thresh=np.inf)
plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'TE (m)', (0.0, 0.9), "(all)_scene_wise", colors, r"$\rm TE_{all}$ (m)")
plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, 'RE (deg)', (0.0, 30.0), "(all)_scene_wise", colors, r"$\rm RE_{all}$ (deg)")