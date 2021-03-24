import numpy as np
from numpy.core.fromnumeric import squeeze
import open3d as o3d
import matplotlib.pyplot as plt
from operator import itemgetter
from heapq import nlargest
import os
import ntpath
import pickle


# green_colors = np.repeat()
# pcd.colors = o3d.utility.Vector3dVector(np_colors)


def extract_data(path):
    with open(path, "rb") as input_file:
        e = pickle.load(input_file)
    return e


def draw_scans(src, tgt, image_name, save_dir, point_size=3, view=False):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src)
    pcd1.estimate_normals()
    # pcd1.paint_uniform_color(np.array([50 / 255, 205 / 255, 50 / 255]).reshape(3, 1))  # green
    # pcd1.paint_uniform_color(np.array([230 / 255, 20 / 255, 50 / 255]).reshape(3, 1))  # red
    pcd1.paint_uniform_color(np.array([255 / 255, 255 / 255, 30 / 255]).reshape(3, 1))  # yellow

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(tgt)
    pcd2.estimate_normals()

    # pcd2.paint_uniform_color(np.array([0 / 255, 0 / 255, 205 / 255]).reshape(3, 1))  # blue
    pcd2.paint_uniform_color(np.array([135 / 255, 206 / 255, 255 / 255]).reshape(3, 1))  # lightblue
    # vis = o3d.visualization.draw_geometries([pcd1, pcd2])
    # vis.capture_screen_image('test.png', True)

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=1000, height=1000, left=50, top=50)
    vis.get_render_option().load_from_json("o3d_config/3dmatch.json")
    ctr = vis.get_view_control()
    # traj = o3d.io.read_pinhole_camera_trajectory("o3d_config/kitti_fov.json")
    # print(traj.parameters)

    vis.add_geometry(pcd1, reset_bounding_box=True)
    vis.add_geometry(pcd2, reset_bounding_box=False)

    param = o3d.io.read_pinhole_camera_parameters("o3d_config/3dmatch_camera.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    if view:
        vis.run()
    else:
        vis.poll_events()
        vis.update_renderer()

    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('o3d_config/3dmatch_camera.json', param)
    image = vis.capture_screen_float_buffer(False)
    plt.imsave(os.path.join(save_dir, image_name + ".png"), np.asarray(image), dpi=1)
    vis.destroy_window()

def draw_origin_quant_sampled(src_orig, src_quant, src, image_name, save_dir, point_size=3, view=False):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src_orig)
    pcd1.estimate_normals()

    pcd2 = o3d.geometry.PointCloud()
    src_quant += np.array([4, 0, 0]).reshape(1, 3)
    pcd2.points = o3d.utility.Vector3dVector(src_quant)

    # pcd2.estimate_normals()

    pcd3 = o3d.geometry.PointCloud()
    src += np.array([8, 0, 0]).reshape(1, 3)
    pcd3.points = o3d.utility.Vector3dVector(src)

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=2000, height=900, left=50, top=50)
    vis.get_render_option().load_from_json("o3d_config/3dmatch_quant.json")
    ctr = vis.get_view_control()
    # traj = o3d.io.read_pinhole_camera_trajectory("o3d_config/kitti_fov.json")
    # print(traj.parameters)

    vis.add_geometry(pcd1, reset_bounding_box=True)
    vis.add_geometry(pcd2, reset_bounding_box=False)
    vis.add_geometry(pcd3, reset_bounding_box=False)

    param = o3d.io.read_pinhole_camera_parameters("o3d_config/3dmatch_quant_camera.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    if view:
        vis.run()
    else:
        vis.poll_events()
        vis.update_renderer()

    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('o3d_config/3dmatch_quant_camera.json', param)
    image = vis.capture_screen_float_buffer(False)
    plt.imsave(os.path.join(save_dir, image_name + ".png"), np.asarray(image), dpi=1)
    vis.destroy_window()

def draw_mask(src, weights, image_name, save_dir, point_size=3):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src)
    low = np.array([0.283072, 0.130895, 0.449241]).reshape(1, 3)
    # high = np.array([0.993248, 0.906157, 0.143936]).reshape(1, 3)
    # low = np.array([0, 0, 0]).reshape(1, 3)
    high = np.array([1.0, 0.0, 0.0]).reshape(1, 3)

    t = np.repeat(weights.reshape(-1, 1), 3, 1)
    colors = low + t * (high - low)
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()

    vis.create_window(width=1000, height=1000, left=50, top=50)
    vis.get_render_option().load_from_json("o3d_config/3dmatch.json")
    ctr = vis.get_view_control()

    vis.add_geometry(pcd1, reset_bounding_box=True)

    param = o3d.io.read_pinhole_camera_parameters("o3d_config/3dmatch_camera.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    # vis.poll_events()
    # vis.update_renderer()
    vis.run()

    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('o3d_config/3dmatch_camera.json', param)
    image = vis.capture_screen_float_buffer(False)
    plt.imsave(os.path.join(save_dir, image_name + ".png"), np.asarray(image), dpi=1)
    vis.destroy_window()


def draw_correspondences(src, src_orig, tgt, tgt_orig, weights, attn, image_name, save_dir, point_size=3, k=256,
                         view=False):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src)
    pcd1.paint_uniform_color(np.array([255 / 255, 255 / 255, 30 / 255]).reshape(3, 1))  # yellow

    pcd1_orig = o3d.geometry.PointCloud()
    pcd1_orig.points = o3d.utility.Vector3dVector(src_orig)
    pcd1_orig.estimate_normals()
    pcd1_orig.paint_uniform_color(np.array([255 / 255, 255 / 255, 30 / 255]).reshape(3, 1))  # yellow
    # pcd1.paint_uniform_color(np.array([50 / 255, 205 / 255, 50 / 255]).reshape(3, 1))  # green
    # pcd1.paint_uniform_color(np.array([230 / 255, 20 / 255, 50 / 255]).reshape(3, 1))  # red

    pcd2 = o3d.geometry.PointCloud()
    # print(tgt.shape)
    tgt += np.array([4, 0, 0]).reshape(1, 3)
    tgt_orig += np.array([4, 0, 0]).reshape(1, 3)

    pcd2.points = o3d.utility.Vector3dVector(tgt)
    pcd2.paint_uniform_color(np.array([135 / 255, 206 / 255, 255 / 255]).reshape(3, 1))  # lightblue

    pcd2_orig = o3d.geometry.PointCloud()
    pcd2_orig.points = o3d.utility.Vector3dVector(tgt_orig)
    pcd2_orig.estimate_normals()
    pcd2_orig.paint_uniform_color(np.array([135 / 255, 206 / 255, 255 / 255]).reshape(3, 1))  # lightblue
    # pcd2.paint_uniform_color(np.array([0 / 255, 0 / 255, 205 / 255]).reshape(3, 1))  # blue
    # vis = o3d.visualization.draw_geometries([pcd1, pcd2])
    # vis.capture_screen_image('test.png', True)

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=2000, height=1000, left=50, top=50)
    vis.get_render_option().load_from_json("o3d_config/3dmatch.json")
    ctr = vis.get_view_control()

    # vis.add_geometry(pcd1, reset_bounding_box=True)

    idx = np.argmax(np.squeeze(attn), axis=1)
    corr_idx = list(zip(range(src.shape[0]), idx))
    klargest = nlargest(k, enumerate(np.squeeze(weights)), itemgetter(1))
    klargest_idx = [i for i, v in klargest]

    corrs = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1, pcd2,
                                                                         [corr_idx[i] for i in klargest_idx])
    corrs.paint_uniform_color(np.array([50 / 255, 205 / 255, 50 / 255]).reshape(3, 1))

    # vis.add_geometry(pcd2, reset_bounding_box=False)

    vis.add_geometry(pcd1_orig, reset_bounding_box=True)
    vis.add_geometry(pcd2_orig, reset_bounding_box=False)
    vis.add_geometry(corrs)

    param = o3d.io.read_pinhole_camera_parameters("o3d_config/3dmatch_camera_corr.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    if view:
        vis.run()
    else:
        vis.poll_events()
        vis.update_renderer()

    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('o3d_config/3dmatch_camera_corr.json', param)

    image = vis.capture_screen_float_buffer(False)
    plt.imsave(os.path.join(save_dir, image_name + ".png"), np.asarray(image), dpi=1)
    vis.destroy_window()


# =========================
# for i in range(10):
#     name = "kitti_scans_{}".format(i)
#     scan = extract_data("data/kitti_scans/{}.npz".format(name))
#
#     # draw_scans(scan['src'], scan['tgt'], image_name=name, point_size=3 )
#     src_est = (scan['Rest'] @ scan['src'].T).T + scan['Test'].T
#     # draw_scans(src_est, scan['tgt'], image_name= name + "_est", point_size=3)
#
#     # draw_mask(scan['src'], scan['w_src'], image_name=name + "_mask")
#     draw_correspondences(src_est, scan['tgt'], scan['w_src'], scan['log_attn_row'], image_name=name + "_corres", k=256)
# ========================


# scan0['src'].reshape()
base_dir = "data/multi_attns"
save_dir = "images/multi_attns"
paths = os.listdir(base_dir)
for path in paths:
    # head, tail = os.path.splitext(path)
    scan = extract_data(os.path.join(base_dir, path))
    src = scan['src'].squeeze()
    n_points = src.shape[1]
    tgt = scan['tgt'].squeeze()
    est_trans = scan['est_trans'].squeeze()
    R_est = est_trans[:3, :3]
    t_est = est_trans[:3, 3].reshape(3, 1)
    src_est = R_est @ src + t_est 
    gt_trans = scan['gt_trans']
    log_attn_rows = [t.squeeze() for t in scan['log_attn_rows']]
    log_attn_cols = [t.squeeze() for t in scan['log_attn_cols']]
    print(src.shape, est_trans.shape, gt_trans.shape)
    print(len(log_attn_rows), log_attn_rows[0].shape)
    # src_true = (scan['Rtrue'] @ scan['src'].T).T + scan['Ttrue'].T
    # src_est_orig = (scan['Rest'] @ scan['src_orig'].T).T + scan['Test'].T
    # src_true_orig = (scan['Rtrue'] @ scan['src_orig'].T).T + scan['Ttrue'].T
    # draw_scans(scan['src_orig'], scan['tgt_orig'], image_name= head + "_init", save_dir=save_dir, point_size=3, view=False)
    # draw_scans(src_true_orig, scan['tgt_orig'], image_name=head + "_gt", point_size=3, save_dir=save_dir)
    # draw_scans(src_est_orig, scan['tgt_orig'], image_name=head + "_est", point_size=3, save_dir=save_dir)

    w = np.ones(n_points)/n_points
    draw_correspondences(src_est.T, src_est.T,
                          tgt.T, tgt.T,
                          w, log_attn_rows[0],
                          image_name=  "attn_{}".format(0), save_dir=save_dir,
                          k=500, view=True)

    # draw_origin_quant_sampled(scan['src_orig'], scan['src_quant'], scan['src'], image_name=head + "_quant", save_dir=save_dir, point_size=3, view=True)
    # break

# for i in range(5):
#     name = "3dmatch_scans_{}".format(i)
#     scan = extract_data("data/3dmatch_scans/{}.npz".format(name))
#     src_true = (scan['Rtrue'] @ scan['src_orig'].T).T + scan['Ttrue'].T
#     src_est_orig = (scan['Rest'] @ scan['src_orig'].T).T + scan['Test'].T
#     draw_scans(scan['src_orig'], scan['tgt_orig'], image_name=name+"_init", point_size=3, view=False)
#     draw_scans(src_true, scan['tgt_orig'], image_name=name+"_gt", point_size=3)
#     draw_scans(src_est_orig, scan['tgt_orig'], image_name=name+"_est", point_size=3)
#
#     src_est = (scan['Rest'] @ scan['src'].T).T + scan['Test'].T
#     draw_correspondences(src_est, src_est_orig,
#                          scan['tgt'], scan['tgt_orig'],
#                          scan['w_src'], scan['log_attn_row'], image_name=name + "_corres", k=256, view=False)
