from lib import plot_precision_recall_curves, analyze_by_pair, analyze_by_pair_single
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

DGR_3DMatch = np.load("results.npz")
method_names = DGR_3DMatch['names']
stats = DGR_3DMatch['stats']
selected_method_ids = [0, 1, 6, 7]
method_names = list(method_names[selected_method_ids])
stats = stats[selected_method_ids]



# DGR
methods = [
    # {
    #     "name": method_names[0],
    #     "data": stats[0],
    #     "is_our": False,
    #     "color": colors[0]
    # },
    # {
    #     "name": method_names[1],
    #     "data": stats[1],
    #     "is_our": False,
    #     "color": colors[1]
    # },
    # {
    #     "name": method_names[2],
    #     "data": stats[2],
    #     "is_our": False,
    #     "color": colors[2]
    # },
    # {
    #     "name": method_names[3],
    #     "data": stats[3],
    #     "is_our": False,
    #     "color": colors[3]
    # },
    # {
    #     "name": "DGR + ICP",
    #     "path_3d": '',
    #     "path_kitti": 'data/dgr_results_for_cvpr/dgr_kitti_with_icp.npz',
    #     "is_our": False,
    # },
    {
        "name": "DGR",
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp.npz',
        "path_kitti": 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp.npz',
        "is_our": False,
    },
    {
        "name": "DGR w/o safeguard",
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard.npz',
        "path_kitti": '',
        "is_our": False,
    },
    {
        "name": r"DGR$^\dagger$",
        "path_3d": 'data/dgr_results_for_cvpr/dgr_3dmatch_original_wo_icp_safeguard_optim.npz',
        "path_kitti": 'data/dgr_results_for_cvpr/dgr_kitti_original_wo_icp_safeguard_optim.npz',
        "is_our": False,
    },
    {
        "name": "Ours",
        "path_3d": 'data/recall_curves_3dmatch_no_icp_245_None_1.npz',
        "path_kitti": 'data/recall_curves_kitti/recall_curves_kitti_no_icp.npz',
        "is_our": True,
    },
    # {
    #     "name": "Ours + ICP",
    #     "path_3d": '',
    #     "path_kitti": 'data/recall_curves_kitti/recall_curves_kitti_icp.npz',
    #     "is_our": True,
    # },
    {
        "name": "Ours-5",
        "path_3d": 'data/recall_curves_3dmatch_no_icp_245_None_5.npz',
        "path_kitti": '',
        "is_our": True,
    },
]
# cmap = plt.get_cmap('tab20b').colors
# t = 2
# colors = [cmap[t + 0], cmap[t + 4], cmap[t + 8], cmap[t + 12], cmap[t + 16]]

# cmap = plt.get_cmap('tab20c').colors
# t = 1
# colors = [cmap[t + 0], cmap[t + 4], cmap[t + 8], cmap[t + 12], cmap[t + 16]]

cmap = plt.get_cmap('tab20').colors
t = 1
colors = [cmap[0], cmap[8], cmap[4], cmap[12], cmap[6]]
# colors = [cmap(i) for i in range(0, len(methods))]
# colors.reverse()

for i, color in enumerate(colors):
    methods[i]['color'] = colors[i]
# for name, path in DGR_rerun_names:
#     method_names.append(name)
#     f = np.load(path)
#     stats = np.concatenate((stats, f['stats']), axis=0)
#
#
# for name, path in our_methods_data:
#     method_names.append(name)
#     g = np.load(path)
#     stats = np.concatenate((stats, g.reshape(-1, 1623, 5).mean(0, keepdims=True)), axis=0)

def draw_curves(stats, ax1, ax2, rre_precisions, rte_precisions, name):
    rre_precision_curve = np.zeros(len(rre_precisions))
    rte_precision_curve = np.zeros(len(rte_precisions))
    for i in range(len(rre_precisions)):
        rre_thresh = rre_precisions[i]
        pairwise_stats = analyze_by_pair_single(stats, rte_thresh=np.inf, rre_thresh=rre_thresh)
        rre_precision_curve[i] = pairwise_stats[0]
    for i in range(len(rte_precisions)):
        rte_thresh = rte_precisions[i]
        pairwise_stats = analyze_by_pair_single(stats, rte_thresh=rte_thresh, rre_thresh=np.inf)
        rte_precision_curve[i] = pairwise_stats[0]

    line1, = ax1.plot(rre_precisions, rre_precision_curve, color=method['color'], alpha=1.0, linewidth=2.5, label=name)
    line2, = ax2.plot(rte_precisions, rte_precision_curve, color=method['color'], alpha=1.0, linewidth=2.5, label=name)

    return line1, line2

kitti_rre_precisions=np.arange(0, 5, 0.05)
kitti_rte_precisions=np.arange(0, 0.6, 0.005)
threedmatch_rre_precisions=np.arange(0, 15, 0.05)
threedmatch_rte_precisions=np.arange(0, 0.3, 0.005)

fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(1, 4, 1, aspect=2.8 / np.max(kitti_rte_precisions))
ax2 = fig.add_subplot(1, 4, 2, aspect=2.8 / np.max(kitti_rre_precisions))
ax3 = fig.add_subplot(1, 4, 3, aspect=4.0 / np.max(threedmatch_rte_precisions))
ax4 = fig.add_subplot(1, 4, 4, aspect=4.0 / np.max(threedmatch_rre_precisions))

lines = []
labels = []
for method in methods:
    kitti_stats = None
    threedmatch_stats = None
    if method['is_our']:
        if 'path_kitti' in method and method['path_kitti'] != '':
            print(method['name'])
            g = np.load(method['path_kitti'])
            kitti_stats = g['results'].reshape(-1, 555, 5).mean(0, keepdims=False)
        if 'path_3d' in method and method['path_3d'] != '':
            g = np.load(method['path_3d'])
            threedmatch_stats = g['results'].reshape(-1, 1623, 5).mean(0, keepdims=False)
    else:
        if 'path_kitti' in method and method['path_kitti'] != '':
            f = np.load(method['path_kitti'])
            kitti_stats = f['stats']
        if 'path_3d' in method and method['path_3d'] != '':
            f = np.load(method['path_3d'])
            threedmatch_stats = np.squeeze(f['stats'])
        if 'data' in method:
            threedmatch_stats = method['data']

    line = None
    if kitti_stats is not None:
        line1, line2 = draw_curves(kitti_stats, ax1, ax2, kitti_rre_precisions, kitti_rte_precisions, method['name'])
        line = line1
        # lines += [line1, line2]
        # labels += [method['name'], method['name']]
    if threedmatch_stats is not None:
        line3, line4 = draw_curves(threedmatch_stats, ax3, ax4, threedmatch_rre_precisions, threedmatch_rte_precisions, method['name'])
        line = line3
        # lines += [line3, line4]
        # labels += [method['name'], method['name']]
    if line is not None:
        lines.append(line)
        labels.append(method['name'])


# rre_precision_curves = np.zeros((num_methods, len(rre_precisions)))
# rte_precision_curves = np.zeros((num_methods, len(rte_precisions)))
#
# for i, rre_thresh in enumerate(rre_precisions):
#     pairwise_stats = analyze_by_pair(stats, rte_thresh=np.inf, rre_thresh=rre_thresh)
#     rre_precision_curves[:, i] = pairwise_stats[:, 0]
#
# for i, rte_thresh in enumerate(rte_precisions):
#     pairwise_stats = analyze_by_pair(stats, rte_thresh=rte_thresh, rre_thresh=np.inf)
#     rte_precision_curves[:, i] = pairwise_stats[:, 0]



# for m, name in enumerate(method_names):
#     alpha = rre_precision_curves[m].mean()
#     alpha = 1.0 if alpha > 0 else 0.0
#
#     ax1.plot(rre_precisions, rre_precision_curves[m], color=cmap[m], alpha=alpha, linewidth=1.5)
#     ax2.plot(rte_precisions, rte_precision_curves[m], color=cmap[m], alpha=alpha, linewidth=1.5)


ax1.set_ylabel('Recall')
ax1.set_xlabel('Rotation (deg)')
ax1.set_ylim((0.0, 1.0))
ax1.set_title("KITTI")
ax1.grid()

ax2.set_xlabel('Translation (m)')
ax2.set_ylim((0.0, 1.0))
ax2.set_title("KITTI")
ax2.grid()

ax3.set_xlabel('Rotation (deg)')
ax3.set_ylim((0.0, 1.0))
ax3.set_title("3DMatch")
ax3.grid()

ax4.set_xlabel('Translation (m)')
ax4.set_title("3DMatch")
ax4.set_ylim((0.0, 1.0))

# for method in methods:
#     names.append(method['name'])
# ax4.legend(names, loc='center left', bbox_to_anchor=(1, 0.5))
ax4.grid()
# ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
print(len(lines))
plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig('{}_{}.png'.format('precision_recall', "both"))

plt.close(fig)
