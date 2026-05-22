import genesis as gs
import json
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

# =====================================================================
# 1. 初期設定とシーン構築
# =====================================================================
def load_config(config_path):
    """JSONから設定を読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    env_cfg = config['environment']
    dataset_id = 'dataset_02_multimodal'
    cam_ids = config['datasets'][dataset_id]['camera_ids']
    return env_cfg, config['cameras'][cam_ids[0]], config['cameras'][cam_ids[1]]

def create_base_scene(env_cfg, urdf_path):
    """Genesisの初期化と、机・マット・ロボットの配置"""
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=True)

    # 机
    desk_size_x = env_cfg['desk_x_max'] - env_cfg['desk_x_min']
    desk_size_y = env_cfg['desk_y_max'] - env_cfg['desk_y_min']
    desk_thickness = 0.05 
    desk_center = (
        (env_cfg['desk_x_max'] + env_cfg['desk_x_min']) / 2,
        (env_cfg['desk_y_max'] + env_cfg['desk_y_min']) / 2,
        env_cfg['desk_z_offset'] - (desk_thickness / 2) 
    )
    scene.add_entity(gs.morphs.Box(size=(desk_size_x, desk_size_y, desk_thickness), pos=desk_center, fixed=True), surface=gs.surfaces.Rough(color=(0.85, 0.76, 0.65)))

    # マット
    mat_thickness = 0.002
    mat_center = (
        env_cfg['mat_x_start'] + (env_cfg['mat_length'] / 2),
        env_cfg['mat_y_start'] + (env_cfg['mat_width'] / 2),
        env_cfg['desk_z_offset'] + env_cfg['mat_z_offset']
    )
    scene.add_entity(gs.morphs.Box(size=(env_cfg['mat_length'], env_cfg['mat_width'], mat_thickness), pos=mat_center, fixed=True), surface=gs.surfaces.Rough(color=(0.15, 0.15, 0.15)))

    # ロボット
    robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, pos=(0, 0, 0)))
    
    return scene, robot

# =====================================================================
# 2. カメラと可視化オブジェクトのセットアップ
# =====================================================================
def setup_fixed_camera_and_visuals(scene, env_cfg, cam_cfg):
    """固定カメラの配置と、その視野枠（緑線）の生成"""
    cam_pos = np.array([cam_cfg['x'], cam_cfg['y'], cam_cfg['z']])
    pitch_rad = np.deg2rad(cam_cfg['pitch_deg'])
    
    cam_dir = np.array([np.cos(pitch_rad), 0, np.sin(pitch_rad)])
    cam_right = np.array([0, -1, 0])
    cam_up = np.array([-np.sin(pitch_rad), 0, np.cos(pitch_rad)])

    w = np.tan(np.deg2rad(cam_cfg['fov_h_deg']) / 2)
    h = np.tan(np.deg2rad(cam_cfg['fov_v_deg']) / 2)

    # 視野の4隅の計算
    rays = np.array([
        cam_dir + w * cam_right + h * cam_up,
        cam_dir - w * cam_right + h * cam_up,
        cam_dir - w * cam_right - h * cam_up,
        cam_dir + w * cam_right - h * cam_up
    ])
    
    z_target = env_cfg['desk_z_offset'] + env_cfg['mat_z_offset']
    t = (z_target - cam_pos[2]) / rays[:, 2]
    pts = cam_pos + t[:, np.newaxis] * rays

    # カメラ自体の召喚
    lookat_target = (cam_pos[0] + np.cos(pitch_rad), cam_pos[1], cam_pos[2] + np.sin(pitch_rad))
    cam_fixed = scene.add_camera(res=(640, 480), pos=tuple(cam_pos), lookat=lookat_target, fov=cam_cfg['fov_v_deg'], GUI=False)

    # OBJファイルの書き出し
    with open('temp_frustum.obj', 'w') as f:
        f.write(f"v {cam_pos[0]} {cam_pos[1]} {cam_pos[2]}\n")
        for p in pts: f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        f.write("f 1 2 3\n"); f.write("f 1 3 4\n"); f.write("f 1 4 5\n"); f.write("f 1 5 2\n")

    with open('temp_tape.obj', 'w') as f:
        v_idx = 1; tape_width = 0.005; tape_thickness = 0.001; z_tape = z_target + 0.002
        for i in range(4):
            p_start = pts[i]; p_end = pts[(i+1)%4]
            d = p_end - p_start; L = np.linalg.norm(d); d = d / L
            u = np.cross(d, np.array([0, 0, 1])); u = u / np.linalg.norm(u); v = np.array([0, 0, 1])
            for off in [u*tape_width + v*tape_thickness, -u*tape_width + v*tape_thickness, -u*tape_width - v*tape_thickness, u*tape_width - v*tape_thickness]:
                f.write(f"v {p_start[0] + off[0]} {p_start[1] + off[1]} {z_tape + off[2]}\n")
            for off in [u*tape_width + v*tape_thickness, -u*tape_width + v*tape_thickness, -u*tape_width - v*tape_thickness, u*tape_width - v*tape_thickness]:
                f.write(f"v {p_end[0] + off[0]} {p_end[1] + off[1]} {z_tape + off[2]}\n")
            f.write(f"f {v_idx} {v_idx+1} {v_idx+5} {v_idx+4}\n"); f.write(f"f {v_idx+1} {v_idx+2} {v_idx+6} {v_idx+5}\n")
            f.write(f"f {v_idx+2} {v_idx+3} {v_idx+7} {v_idx+6}\n"); f.write(f"f {v_idx+3} {v_idx} {v_idx+4} {v_idx+7}\n")
            f.write(f"f {v_idx} {v_idx+3} {v_idx+2} {v_idx+1}\n"); f.write(f"f {v_idx+4} {v_idx+5} {v_idx+6} {v_idx+7}\n")
            v_idx += 8

    with open('temp_edges.obj', 'w') as f:
        v_idx = 1; thickness = 0.002 
        for p_end in pts:
            d = p_end - cam_pos; L = np.linalg.norm(d); d = d / L
            u = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
            u = u - np.dot(u, d) * d; u = u / np.linalg.norm(u); v = np.cross(d, u)
            for off in [u, v, -u, -v]: f.write(f"v {cam_pos[0] + off[0]*thickness} {cam_pos[1] + off[1]*thickness} {cam_pos[2] + off[2]*thickness}\n")
            for off in [u, v, -u, -v]: f.write(f"v {p_end[0] + off[0]*thickness} {p_end[1] + off[1]*thickness} {p_end[2] + off[2]*thickness}\n")
            f.write(f"f {v_idx} {v_idx+1} {v_idx+5} {v_idx+4}\n"); f.write(f"f {v_idx+1} {v_idx+2} {v_idx+6} {v_idx+5}\n")
            f.write(f"f {v_idx+2} {v_idx+3} {v_idx+7} {v_idx+6}\n"); f.write(f"f {v_idx+3} {v_idx} {v_idx+4} {v_idx+7}\n")
            v_idx += 8

    # エンティティの追加
    scene.add_entity(gs.morphs.Mesh(file='temp_frustum.obj', fixed=True, collision=False), surface=gs.surfaces.Rough(color=(0.0, 0.8, 1.0, 0.15)))
    scene.add_entity(gs.morphs.Mesh(file='temp_tape.obj', fixed=True, collision=False), surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0)))
    scene.add_entity(gs.morphs.Mesh(file='temp_edges.obj', fixed=True, collision=False), surface=gs.surfaces.Rough(color=(0.0, 0.8, 1.0)))
    
    # 後の計算で使い回す数学データを辞書にして返す
    f_math = {
        'pos': cam_pos, 'dir': cam_dir, 'right': cam_right, 'up': cam_up,
        'w': w, 'h': h, 'pts': pts, 'desk_z': z_target
    }
    return cam_fixed, f_math

def setup_hand_camera_and_visuals(scene, cam_cfg):
    """手先カメラの召喚と、追従するマーカー群の準備（美しい実線アウトライン版）"""
    cam_hand = scene.add_camera(res=(640, 480), pos=(0,0,0), lookat=(1,0,0), fov=cam_cfg['fov_v_deg'], GUI=False)
    cam_marker = scene.add_entity(gs.morphs.Sphere(radius=0.005, fixed=True, collision=False), surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0)))

    L = 1.2
    w_h = L * np.tan(np.deg2rad(cam_cfg['fov_h_deg']) / 2)
    h_h = L * np.tan(np.deg2rad(cam_cfg['fov_v_deg']) / 2)

    with open('temp_hand_frustum.obj', 'w') as f:
        f.write("v 0 0 0\n")
        for pt in [[L, -w_h, h_h], [L, w_h, h_h], [L, w_h, -h_h], [L, -w_h, -h_h]]:
            f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
        f.write("f 1 2 3\n"); f.write("f 1 3 4\n"); f.write("f 1 4 5\n"); f.write("f 1 5 2\n")

    with open('temp_hand_frustum_edges.obj', 'w') as f:
        v_idx = 1; thickness = 0.002; cam_origin = np.array([0,0,0])
        ends = [np.array([L, -w_h, h_h]), np.array([L, w_h, h_h]), np.array([L, w_h, -h_h]), np.array([L, -w_h, -h_h])]
        for p_end in ends:
            d = p_end - cam_origin; L_dist = np.linalg.norm(d); d = d / L_dist
            u = np.array([0, 1, 0]) if abs(d[2]) < 0.9 else np.array([1, 0, 0])
            u = u - np.dot(u, d) * d; u = u / np.linalg.norm(u); v = np.cross(d, u)
            for off in [u, v, -u, -v]: f.write(f"v {cam_origin[0] + off[0]*thickness} {cam_origin[1] + off[1]*thickness} {cam_origin[2] + off[2]*thickness}\n")
            for off in [u, v, -u, -v]: f.write(f"v {p_end[0] + off[0]*thickness} {p_end[1] + off[1]*thickness} {p_end[2] + off[2]*thickness}\n")
            f.write(f"f {v_idx} {v_idx+1} {v_idx+5} {v_idx+4}\n"); f.write(f"f {v_idx+1} {v_idx+2} {v_idx+6} {v_idx+5}\n")
            f.write(f"f {v_idx+2} {v_idx+3} {v_idx+7} {v_idx+6}\n"); f.write(f"f {v_idx+3} {v_idx} {v_idx+4} {v_idx+7}\n")
            v_idx += 8

    # 手先カメラから伸びる光の筋
    cam_hand_frustum = scene.add_entity(gs.morphs.Mesh(file='temp_hand_frustum.obj', fixed=True, collision=False), surface=gs.surfaces.Rough(color=(1.0, 0.8, 0.2, 0.25)))
    cam_hand_frustum_edges = scene.add_entity(gs.morphs.Mesh(file='temp_hand_frustum_edges.obj', fixed=True, collision=False), surface=gs.surfaces.Rough(color=(1.0, 0.5, 0.0, 0.9)))

    # 手先カメラが机とぶつかる4隅
    hand_fov_markers = [scene.add_entity(gs.morphs.Sphere(radius=0.005, fixed=True, collision=False), surface=gs.surfaces.Rough(color=(1.0, 0.5, 0.0))) for _ in range(4)]
    
    # ==========================================================
    # 💡 [NEW] 「完全な実線の枠線」を描くための細いラインパーツ（長さ5cm）を100本用意！
    # 激重だったタイル（400枚）は完全に消去しました。
    # ==========================================================
    outline_segments = [scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.004, 0.004), fixed=True, collision=False), 
        surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.5, 1.0)) # 濃いエメラルドグリーンの実線
    ) for _ in range(100)]

    h_visuals = {
        'cam_marker': cam_marker, 'frustum': cam_hand_frustum, 'frustum_edges': cam_hand_frustum_edges,
        'fov_markers': hand_fov_markers,
        'outline_segments': outline_segments,  # 実線パーツのみを管理
        'is_visible': True
    }
    return cam_hand, h_visuals

def attach_hand_camera(robot, cam_hand, cam_cfg):
    """手先カメラをアームの指定リンクに溶接する"""
    hand_link = robot.get_link(cam_cfg['attached_link'])

    r = np.deg2rad(cam_cfg.get('roll_deg', 0.0))
    p = np.deg2rad(cam_cfg.get('pitch_deg', 0.0))
    y = np.deg2rad(cam_cfg.get('yaw_deg', 0.0))

    R_json = R.from_euler('ZYX', [y, p, r], degrees=False).as_matrix()
    R_optical_fix = R.from_euler('XYZ', [90, -90, 0], degrees=True).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R_json @ R_optical_fix  
    T[0, 3] = cam_cfg['offset_x']
    T[1, 3] = cam_cfg['offset_y']
    T[2, 3] = cam_cfg['offset_z']

    cam_hand.attach(rigid_link=hand_link, offset_T=T)
    return hand_link

# =====================================================================
# 3. 毎フレームの計算ロジック（3D / 2D）
# =====================================================================
def update_3d_visuals(hand_link, cam_cfg, f_math, h_visuals):
    """アームの動きに合わせて3D空間の実線枠を生成する"""
    link_pos = np.array(hand_link.get_pos().tolist())
    quat_wxyz = hand_link.get_quat().tolist()
    
    r_link = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R_link = r_link.as_matrix()

    cam_local_offset = np.array([cam_cfg['offset_x'], cam_cfg['offset_y'], cam_cfg['offset_z']])
    global_cam_pos = link_pos + R_link @ cam_local_offset

    h_visuals['cam_marker'].set_pos(global_cam_pos.tolist())
    h_visuals['frustum'].set_pos(global_cam_pos.tolist())
    h_visuals['frustum_edges'].set_pos(global_cam_pos.tolist())

    r_rad = np.deg2rad(cam_cfg.get('roll_deg', 0.0))
    p_rad = np.deg2rad(cam_cfg.get('pitch_deg', 0.0))
    y_rad = np.deg2rad(cam_cfg.get('yaw_deg', 0.0))
    
    r_local = R.from_euler('ZYX', [y_rad, p_rad, r_rad], degrees=False)
    r_global = r_link * r_local
    
    q = r_global.as_quat()
    global_quat_wxyz = [q[3], q[0], q[1], q[2]]

    h_visuals['frustum'].set_quat(global_quat_wxyz)
    h_visuals['frustum_edges'].set_quat(global_quat_wxyz)

    desk_z = f_math['desk_z']
    w_h = np.tan(np.deg2rad(cam_cfg['fov_h_deg']) / 2)
    h_h = np.tan(np.deg2rad(cam_cfg['fov_v_deg']) / 2)
    
    local_rays = np.array([[1, -w_h, h_h], [1, w_h, h_h], [1, w_h, -h_h], [1, -w_h, -h_h]])
    global_rays = local_rays @ r_global.as_matrix().T
    
    hand_pts_3d = []
    for ray in global_rays:
        if ray[2] < -1e-4:
            t = (desk_z - global_cam_pos[2]) / ray[2]
            if t > 0: hand_pts_3d.append(global_cam_pos + t * ray)
    
    marker_z = desk_z + 0.015 
    if len(hand_pts_3d) == 4:
        h_visuals['is_visible'] = True
        for i in range(4):
            h_visuals['fov_markers'][i].set_pos([hand_pts_3d[i][0], hand_pts_3d[i][1], marker_z])

        poly_hand = cv2.convexHull(np.array([[p[0], p[1]] for p in hand_pts_3d], dtype=np.float32))
        pts = f_math['pts']
        poly_fixed = cv2.convexHull(np.array([[pts[1][0], pts[1][1]], [pts[0][0], pts[0][1]], [pts[3][0], pts[3][1]], [pts[2][0], pts[2][1]]], dtype=np.float32))
        
        intersect_area, poly_intersect = cv2.intersectConvexConvex(poly_hand, poly_fixed)
        
        if intersect_area > 0 and poly_intersect is not None:
            num_pts = len(poly_intersect)
            
            # ==========================================================
            # 🎯 長さ5cmのパーツを多角形の辺に沿って並べ、1本の「実線」に錬成！
            # ==========================================================
            seg_idx = 0
            for i in range(num_pts):
                p1 = poly_intersect[i].ravel()
                p2 = poly_intersect[(i+1)%num_pts].ravel()
                dist = np.linalg.norm(p2 - p1)
                if dist < 1e-4: continue
                
                # 辺の向きを計算してパーツを回転させる
                dir_v = (p2 - p1) / dist
                yaw = np.arctan2(dir_v[1], dir_v[0])
                quat_wxyz = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                
                # 隙間ができないように、少しずつ重ねながら配置
                n_segs = int(np.ceil(dist / 0.05))
                step = (dist - 0.05) / max(1, n_segs - 1) if n_segs > 1 else 0
                for k in range(n_segs):
                    if seg_idx >= 100: break
                    pos = p1 + dir_v * (0.025 + k * step)
                    h_visuals['outline_segments'][seg_idx].set_pos([pos[0], pos[1], marker_z])
                    h_visuals['outline_segments'][seg_idx].set_quat(quat_wxyz)
                    seg_idx += 1
            
            # 使わなかった残りのパーツは地下に隠す
            while seg_idx < 100:
                h_visuals['outline_segments'][seg_idx].set_pos([0, 0, -1.0])
                seg_idx += 1
        else:
            for m in h_visuals['outline_segments']: m.set_pos([0, 0, -1.0])
    else:
        # FPS低下を防ぐフラグ処理
        if h_visuals['is_visible'] == True:
            for m in h_visuals['fov_markers']: m.set_pos([0, 0, -1.0])
            for m in h_visuals['outline_segments']: m.set_pos([0, 0, -1.0])
            h_visuals['is_visible'] = False

    return global_cam_pos, r_global

def apply_2d_ar_overlay(img_bgr, cam_cfg, global_cam_pos, r_global, f_math):
    """固定カメラの映像に手先カメラの枠をAR合成する"""
    w_h = np.tan(np.deg2rad(cam_cfg['fov_h_deg']) / 2)
    h_h = np.tan(np.deg2rad(cam_cfg['fov_v_deg']) / 2)
    local_rays = np.array([[1, -w_h, h_h], [1, w_h, h_h], [1, w_h, -h_h], [1, -w_h, -h_h]])
    global_rays = local_rays @ r_global.as_matrix().T
    
    def project_to_2d(p3d):
        V = p3d - f_math['pos']
        z_c = np.dot(V, f_math['dir'])
        if z_c > 0:
            u = int(320 + (np.dot(V, f_math['right']) / z_c) * (320 / f_math['w']))
            v = int(240 - (np.dot(V, f_math['up']) / z_c) * (240 / f_math['h']))
            return [u, v]
        return None

    desk_z = f_math['desk_z']
    hand_pts_3d = []
    for ray in global_rays:
        if ray[2] < -1e-4:
            t = (desk_z - global_cam_pos[2]) / ray[2]
            if t > 0: hand_pts_3d.append(global_cam_pos + t * ray)
    
    if len(hand_pts_3d) == 4:
        hand_pts_2d = [project_to_2d(p) for p in hand_pts_3d]
        fixed_pts_2d = [project_to_2d(p) for p in f_math['pts']]

        if all(p is not None for p in hand_pts_2d) and all(p is not None for p in fixed_pts_2d):
            poly_hand = np.array(hand_pts_2d, dtype=np.float32)
            poly_fixed = np.array(fixed_pts_2d, dtype=np.float32)

            intersect_area, poly_intersect = cv2.intersectConvexConvex(poly_hand, poly_fixed)

            overlay = img_bgr.copy()
            cv2.polylines(overlay, [np.int32(poly_hand)], True, (0, 165, 255), 1)

            if intersect_area > 0 and poly_intersect is not None:
                poly_intersect_int = np.int32(poly_intersect)
                cv2.fillPoly(overlay, [poly_intersect_int], (0, 255, 120))
                cv2.polylines(overlay, [poly_intersect_int], True, (0, 255, 255), 2)

            cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, img_bgr)

    return img_bgr

def cleanup_temp_files():
    """終了時にゴミファイルをお掃除"""
    files = ['temp_frustum.obj', 'temp_tape.obj', 'temp_edges.obj', 'temp_hand_frustum.obj', 'temp_hand_frustum_edges.obj']
    for f in files:
        if os.path.exists(f): os.remove(f)

# =====================================================================
# 4. メインループ
# =====================================================================
def main():
    # 1. 設定の読み込みとシーン構築
    config_path = '/home/hogehoge/Documents/so101_MATLAB/env_config.json'
    env_cfg, cam_fixed_cfg, cam_hand_cfg = load_config(config_path)
    
    urdf_path = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf'
    scene, robot = create_base_scene(env_cfg, urdf_path)

    # 2. カメラと可視化オブジェクトの準備
    cam_fixed, f_math = setup_fixed_camera_and_visuals(scene, env_cfg, cam_fixed_cfg)
    cam_hand, h_visuals = setup_hand_camera_and_visuals(scene, cam_hand_cfg)

    # 3. シーンのビルドと溶接
    scene.build()
    hand_link = attach_hand_camera(robot, cam_hand, cam_hand_cfg)

    # 4. GUIの準備
    target_pos = np.zeros(robot.n_dofs)
    print("🚀 マルチモーダル環境を起動しました！")
    
    window_name = "Genesis Dual Camera View (1280x480)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    def on_change(val): pass 
    joint_names = ["1: Base (Pan)", "2: Shoulder", "3: Elbow", "4: Wrist Flex", "5: Wrist Roll", "6: Gripper"]
    for name in joint_names: cv2.createTrackbar(name, window_name, 180, 360, on_change)

    is_topmost_set = False 

    # 5. 実行ループ
    try:
        while True:
            # スライダーからアームを動かす
            for i, name in enumerate(joint_names):
                if i < robot.n_dofs:
                    val = cv2.getTrackbarPos(name, window_name)
                    target_pos[i] = np.deg2rad(val - 180)
            robot.control_dofs_position(target_pos)
            scene.step()

            # 💡 [重要] try-except を廃止し、値がある時だけ安全に実行（例外オーバーヘッド撲滅）
            global_cam_pos, r_global = update_3d_visuals(hand_link, cam_hand_cfg, f_math, h_visuals)

            # 固定カメラ映像の取得（毎フレーム1回のみ）
            render_f = cam_fixed.render(rgb=True)
            img_f = render_f[0] if isinstance(render_f, tuple) else render_f
            img_fixed_bgr = cv2.cvtColor(img_f, cv2.COLOR_RGB2BGR)

            # 2D映像にAR合成（値の有無を明示的にチェック）
            if global_cam_pos is not None:
                if r_global is not None:
                    img_fixed_bgr = apply_2d_ar_overlay(img_fixed_bgr, cam_hand_cfg, global_cam_pos, r_global, f_math)

            # 手先カメラ映像の取得（毎フレーム1回のみ）
            render_h = cam_hand.render(rgb=True)
            img_h = render_h[0] if isinstance(render_h, tuple) else render_h
            img_hand_bgr = cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR)

            # 描画
            combined_img = np.hstack((img_fixed_bgr, img_hand_bgr))
            cv2.imshow(window_name, combined_img)

            if not is_topmost_set:
                import subprocess
                subprocess.Popen(f'wmctrl -r "{window_name}" -b add,above', shell=True)
                is_topmost_set = True

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C が押されました。安全に終了します...")
    finally:
        cv2.destroyAllWindows()
        cleanup_temp_files()

if __name__ == "__main__":
    main()