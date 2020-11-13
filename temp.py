# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import matplotlib.pyplot as plt

f = np.load('Y:/rigid_registration/DeepGlobalRegistration/results_for_cvpr/dgr_kitti_original_wo_icp.npz')

threshold_te = .6
threshold_re = 5
t_list = np.linspace(0, threshold_te, 100)
r_list = np.linspace(0, threshold_re, 100)

recall_vs_t = np.zeros((len(t_list),))
recall_vs_r = np.zeros((len(r_list),))

# Recall vs t
where = f['stats'][:, 2] < threshold_re
for j, th in enumerate(t_list):
    recall_vs_t[j] = np.logical_and(f['stats'][:, 1] < th, where).mean()
# Recall vs r
where = f['stats'][:, 1]  < threshold_te
for j, th in enumerate(r_list):
    recall_vs_r[j] = np.logical_and(f['stats'][:, 2] < th, where).mean()


g = np.load('Y:/rigid_registration/FLOW/scripts/results_cvpr_submission/recall_curves_kitti_no_icp.npz')
    
plt.figure(1); plt.clf()
plt.plot(t_list, recall_vs_t)
plt.plot(g['t_list'], g['recall_vs_t'].mean(1))

plt.figure(2); plt.clf()
plt.plot(r_list, recall_vs_r)
plt.plot(g['r_list'], g['recall_vs_r'].mean(1))
    
