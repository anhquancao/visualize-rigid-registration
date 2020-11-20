import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from operator import itemgetter
from heapq import nlargest
import os
import ntpath


# green_colors = np.repeat()
# pcd.colors = o3d.utility.Vector3dVector(np_colors)


def extract_data(path):
    scan = np.load(path)
    print(list(scan.keys()))
    # ['src', 'tgt', 'src_quant', 'tgt_quant', 'src_orig', 'tgt_orig', 'Rtrue', 'Ttrue', 'Rest', 'Test', 'corres_pts_for_src', 'log_attn_row', 'w_src', 'filename']
    # print(scan['src_orig'].shape)
    src = np.squeeze(scan['src']).T
    tgt = np.squeeze(scan['tgt']).T
    src_quant = np.squeeze(scan['src_quant'])
    tgt_quant = np.squeeze(scan['tgt_quant'])
    src_orig = np.squeeze(scan['src_orig'])
    tgt_orig = np.squeeze(scan['tgt_orig'])

    R_est = np.squeeze(scan['Rest'])
    T_est = np.squeeze(scan['Test'], 0)

    return {
        'src_orig': src_orig,
        'tgt_orig': tgt_orig,
        'src': src,
        'tgt': tgt,
        'src_quant': src_quant,
        'tgt_quant': tgt_quant,
        'Rest': R_est,
        'Test': T_est,
        'Rtrue': scan['Rtrue'],
        'Ttrue': scan['Ttrue'],
        'w_src': scan['w_src'],
        'corres_pts_for_src': scan['corres_pts_for_src'],
        'log_attn_row': scan['log_attn_row']
    }


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
base_dir = "data/supp/3dmatch"
save_dir = "images/3dmatch"
paths = os.listdir(base_dir)
for path in paths:
    head, tail = os.path.splitext(path)
    scan = extract_data(os.path.join(base_dir, path))
    src_true = (scan['Rtrue'] @ scan['src'].T).T + scan['Ttrue'].T
    src_est_orig = (scan['Rest'] @ scan['src_orig'].T).T + scan['Test'].T
    src_true_orig = (scan['Rtrue'] @ scan['src_orig'].T).T + scan['Ttrue'].T
    draw_scans(scan['src_orig'], scan['tgt_orig'], image_name= head + "_init", save_dir=save_dir, point_size=3, view=False)
    draw_scans(src_true_orig, scan['tgt_orig'], image_name=head + "_gt", point_size=3, save_dir=save_dir)
    draw_scans(src_est_orig, scan['tgt_orig'], image_name=head + "_est", point_size=3, save_dir=save_dir)

    # src_est = (scan['Rest'] @ scan['src'].T).T + scan['Test'].T
    draw_correspondences(src_true, src_true_orig,
                         scan['tgt'], scan['tgt_orig'],
                         scan['w_src'], scan['log_attn_row'],
                         image_name= head + "_corres", save_dir=save_dir,
                         k=256, view=False)
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
