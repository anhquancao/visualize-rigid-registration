import matplotlib.pyplot  as plt
import numpy as np

PROPERTY_IDX_MAP = {
    'Recall': 0,
    'TE (m)': 1,
    'RE (deg)': 2,
    'log Time (s)': 3,
    'Scene ID': 4
}

def analyze_by_scene(stats, scene_id_list, rte_thresh=0.3, rre_thresh=15):
  '''
  \input stats: (num_methods, num_pairs, num_pairwise_stats=5)
  \return scene_wise mean stats: (num_methods, num_scenes, 4)
  4 properties: recall, rte, rre, time
  '''
  num_methods, num_pairs, num_pairwise_stats = stats.shape
  num_scenes = len(scene_id_list)

  scene_wise_stats = np.zeros((num_methods, len(scene_id_list), 4))

  for m in range(num_methods):
    # Filter valid registrations by rte / rre thresholds
    mask_rte = stats[m, :, 1] < rte_thresh
    mask_rre = stats[m, :, 2] < rre_thresh
    mask_valid = mask_rte * mask_rre

    for s in scene_id_list:
      mask_scene = stats[m, :, 4] == s

      # Valid registrations in the scene
      mask = mask_scene * mask_valid

      # Recall, RTE, RRE, Time
      scene_wise_stats[m, s, 0] = 0 if np.sum(mask_scene) == 0 else float(
          np.sum(mask)) / float(np.sum(mask_scene))
      scene_wise_stats[m, s, 1] = stats[m, mask, 1].mean()
      scene_wise_stats[m, s, 2] = stats[m, mask, 2].mean()
      scene_wise_stats[m, s, 3] = stats[m, mask, 3].mean()

  return scene_wise_stats

def plot_scene_wise_stats(scene_wise_stats, method_names, scene_names, property_name,
                          ylim, output_postfix, cmap, y_label):
  '''
  \input scene_wise_stats: (num_methods, num_scenes, 4)
  \input method_names:  (num_methods) string, shown as xticks
  \input scene_names:   (num_scenes) string, shown as legends
  \input property_name: string, shown as ylabel
  '''
  num_methods, num_scenes, _ = scene_wise_stats.shape
  assert len(method_names) == num_methods
  assert len(scene_names) == num_scenes

  # Initialize figure
  fig = plt.figure(figsize=(14, 3))
  ax = fig.add_subplot(1, 1, 1)

  # Add some paddings
  w = 1.0 / (num_methods + 2)

  # Rightmost bar
  x = np.arange(0, num_scenes) - 0.5 * w * num_methods

  for m in range(num_methods):
    m_stats = scene_wise_stats[m, :, PROPERTY_IDX_MAP[property_name]]
    valid = not (np.logical_and.reduce(np.isnan(m_stats))
                 or np.logical_and.reduce(m_stats == 0))
    alpha = 1.0 if valid else 0.0
    ax.bar(x + m * w, m_stats, w, color=cmap[m], alpha=alpha)

  plt.ylim(ylim)
  plt.xlim((0 - w * num_methods, num_scenes))
  plt.ylabel(y_label)
  plt.xticks(np.arange(0, num_scenes), tuple(scene_names))
  ax.legend(method_names, loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()
  plt.grid()
  plt.savefig('{}_{}.png'.format(property_name, output_postfix))
  plt.close(fig)


def plot_precision_recall_curves(stats, method_names, rte_precisions, rre_precisions,
                                 output_postfix, cmap, figsize=(8, 3.3), aspect=3.0, title='KITTI'):
    '''
    \input stats: (num_methods, num_pairs, 5)
    \input method_names:  (num_methods) string, shown as xticks
    '''
    num_methods, num_pairs, _ = stats.shape
    rre_precision_curves = np.zeros((num_methods, len(rre_precisions)))
    rte_precision_curves = np.zeros((num_methods, len(rte_precisions)))

    for i, rre_thresh in enumerate(rre_precisions):
        pairwise_stats = analyze_by_pair(stats, rte_thresh=np.inf, rre_thresh=rre_thresh)
        rre_precision_curves[:, i] = pairwise_stats[:, 0]

    for i, rte_thresh in enumerate(rte_precisions):
        pairwise_stats = analyze_by_pair(stats, rte_thresh=rte_thresh, rre_thresh=np.inf)
        rte_precision_curves[:, i] = pairwise_stats[:, 0]

    fig = plt.figure(figsize=figsize)
    # ax1 = fig.add_subplot(1, 2, 1, aspect=3.0 / np.max(rte_precisions))
    # ax2 = fig.add_subplot(1, 2, 2, aspect=3.0 / np.max(rre_precisions))
    plt.rcParams.update({'font.size': 11})

    ax1 = fig.add_subplot(1, 2, 1, aspect=aspect / np.max(rte_precisions))
    ax2 = fig.add_subplot(1, 2, 2, aspect=aspect / np.max(rre_precisions))

    for m, name in enumerate(method_names):
        alpha = rre_precision_curves[m].mean()
        alpha = 1.0 if alpha > 0 else 0.0

        ax1.plot(rre_precisions, rre_precision_curves[m], color=cmap[m], alpha=alpha, linewidth=1.5)
        ax2.plot(rte_precisions, rte_precision_curves[m], color=cmap[m], alpha=alpha, linewidth=1.5)

    # for m, name in enumerate(our_method_names):
    #     ax1.plot(our_method_rs[m], our_methods_recall_rs[m], color=cmap[m + len(method_names)])
    #     ax2.plot(our_method_ts[m], our_methods_recall_ts[m], color=cmap[m + len(method_names)])

    ax1.set_ylabel('Recall')
    ax1.set_xlabel('Rotation (deg)')
    ax1.set_ylim((0.0, 1.0))

    ax2.set_xlabel('Translation (m)')
    ax2.set_ylim((0.0, 1.0))
    ax2.legend(method_names, loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid()
    ax2.grid()

    fig.suptitle(title, y=0.93)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig('{}_{}.png'.format('precision_recall', output_postfix))

    plt.close(fig)

def analyze_by_pair_single(stats, rte_thresh, rre_thresh):
    '''
    \input stats: (num_methods, num_pairs, num_pairwise_stats=5)
    \return valid mean_stats: (num_methods, 4)
    4 properties: recall, rte, rre, time
    '''
    pairwise_stats = np.zeros(4)

    # Filter valid registrations by rte / rre thresholds
    mask_rte = stats[:, 1] < rte_thresh
    mask_rre = stats[:, 2] < rre_thresh
    mask_valid = mask_rte * mask_rre

    # Recall, RTE, RRE, Time
    pairwise_stats[0] = mask_valid.mean()
    pairwise_stats[1] = stats[mask_valid, 1].mean()
    pairwise_stats[2] = stats[mask_valid, 2].mean()
    pairwise_stats[3] = stats[mask_valid, 3].mean()


    return pairwise_stats


def analyze_by_pair(stats, rte_thresh, rre_thresh):
    '''
    \input stats: (num_methods, num_pairs, num_pairwise_stats=5)
    \return valid mean_stats: (num_methods, 4)
    4 properties: recall, rte, rre, time
    '''
    num_methods, num_pairs, num_pairwise_stats = stats.shape
    pairwise_stats = np.zeros((num_methods, 4))

    for m in range(num_methods):
        # Filter valid registrations by rte / rre thresholds
        mask_rte = stats[m, :, 1] < rte_thresh
        mask_rre = stats[m, :, 2] < rre_thresh
        mask_valid = mask_rte * mask_rre

        # Recall, RTE, RRE, Time
        pairwise_stats[m, 0] = mask_valid.mean()
        pairwise_stats[m, 1] = stats[m, mask_valid, 1].mean()
        pairwise_stats[m, 2] = stats[m, mask_valid, 2].mean()
        pairwise_stats[m, 3] = stats[m, mask_valid, 3].mean()


    return pairwise_stats
