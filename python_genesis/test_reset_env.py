import genesis as gs
import numpy as np
import yaml
import os
import time
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces

# 紙コップとスポンジのランダム配置をテストするコード

class SO101GraspEnvTest(gym.Env):
    """
    ランダム配置の動作確認に特化した環境クラス（ビューワー有効化）
    """
    def __init__(self):
        super().__init__()
        # Genesisの初期化（ビューワー用のinfoレベル）
        try:
            gs.init(backend=gs.cpu, logging_level="info")
        except Exception:
            pass

        # YAMLとURDFのパス設定
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        self.config_path = os.path.join(parent_dir, "config", "env_config.yaml")
        self.cup_urdf_path = os.path.join(parent_dir, "assets", "papercup.urdf")
        self.robot_urdf_path = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf'

        # パラメータのロードと幾何学計算
        self._load_config()
        self._build_scene()

    def _load_config(self):
        """YAMLから設定を読み込み、出現可能エリアのポリゴンを計算する"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f)

        env_data = env_config["environment"]
        cam_cfg = env_config["cameras"]["cam_fixed_side"]

        # マットの寸法
        self.mat_x_start = env_data["mat_x_start"]
        self.mat_y_start = env_data["mat_y_start"]
        self.mat_length  = env_data["mat_length"]
        self.mat_width   = env_data["mat_width"]
        self.mat_z       = env_data.get("desk_z_offset", -0.0054) + env_data.get("mat_z_offset", 0.001)

        # アームの可動域と死角
        self.arm_reach_min = env_data.get("arm_reach_min", 0.10)
        self.arm_reach_max = env_data.get("arm_reach_max", 0.30)
        self.arm_yaw_min   = np.deg2rad(env_data.get("arm_yaw_min_deg", -90.0))
        self.arm_yaw_max   = np.deg2rad(env_data.get("arm_yaw_max_deg", 90.0))
        
        self.shoulder_x = 0.04 + 0.0388
        self.shoulder_y = 0.04 + 0.0
        
        self.deadzone_x_min, self.deadzone_x_max = 0.0, 0.25
        self.deadzone_y_min, self.deadzone_y_max = 0.0, 0.15

        # カメラ視野ポリゴンの計算
        cam_x, cam_y, cam_z = cam_cfg["x"], cam_cfg["y"], cam_cfg["z"]
        pitch = np.deg2rad(cam_cfg["pitch_deg"])
        cam_dir   = np.array([np.cos(pitch), 0, np.sin(pitch)])
        cam_right = np.array([0, -1, 0])
        cam_up    = np.array([-np.sin(pitch), 0, np.cos(pitch)])

        w = np.tan(np.deg2rad(cam_cfg["fov_h_deg"]) / 2)
        h = np.tan(np.deg2rad(cam_cfg["fov_v_deg"]) / 2)

        rays = np.array([
            cam_dir + w * cam_right + h * cam_up,
            cam_dir - w * cam_right + h * cam_up,
            cam_dir - w * cam_right - h * cam_up,
            cam_dir + w * cam_right - h * cam_up
        ])

        t = (self.mat_z - cam_z) / rays[:, 2]
        intersect_pts = np.array([cam_x, cam_y, cam_z]) + t[:, np.newaxis] * rays
        self.fov_poly = Path(intersect_pts[:, :2])

        self.home_qpos = np.array(env_data.get("home_qpos", [0.0]*6), dtype=np.float32)

    def _is_valid_spawn_area(self, x, y):
        """出現条件の判定"""
        if not (self.mat_x_start <= x <= (self.mat_x_start + self.mat_length)): return False
        if not (self.mat_y_start <= y <= (self.mat_y_start + self.mat_width)):  return False
        if not self.fov_poly.contains_point((x, y)): return False

        dx = x - self.shoulder_x
        dy = y - self.shoulder_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        if not (self.arm_reach_min <= r <= self.arm_reach_max): return False
        if not (self.arm_yaw_min <= theta <= self.arm_yaw_max): return False
        if (self.deadzone_x_min <= x <= self.deadzone_x_max) and (self.deadzone_y_min <= y <= self.deadzone_y_max):
            return False
            
        return True

    def _build_scene(self):
        # ★検証用に show_viewer=True に設定
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.8, -0.8, 0.6), camera_lookat=(0.15, 0.0, 0.1)
            ),
            show_viewer=True
        )
        self.scene.add_entity(gs.morphs.Plane())

        # マット生成
        mat_cx = self.mat_x_start + (self.mat_length / 2.0)
        mat_cy = self.mat_y_start + (self.mat_width / 2.0)
        self.scene.add_entity(
            morph=gs.morphs.Box(size=(self.mat_length, self.mat_width, 0.002), pos=(mat_cx, mat_cy, 0.001), fixed=True),
            surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15, 1.0))
        )

        # ロボット
        try:
            self.robot = self.scene.add_entity(
                material=gs.materials.Rigid(friction=2.0, coup_friction=3.0),
                morph=gs.morphs.URDF(file=self.robot_urdf_path, pos=(0, 0, 0), fixed=True)
            )
        except Exception:
            self.robot = self.scene.add_entity(morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

        # スポンジと紙コップ
        self.sponge = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500, friction=5.0, coup_friction=5.0),
            morph=gs.morphs.Box(size=(0.04, 0.03, 0.03), pos=(0.1, 0, 0.1), fixed=False),
            surface=gs.surfaces.Default(color=(0.7, 1.0, 0.4, 1.0))
        )
        self.cup = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500, friction=2.0, coup_friction=3.0),
            morph=gs.morphs.URDF(file=self.cup_urdf_path, pos=(0.1, 0.1, 0.1), fixed=False)
        )

        self.scene.build()
        self.n_dofs = self.robot.n_dofs

    def reset(self, seed=None, options=None):
        """ランダム再配置を実行するメソッド"""
        rng = np.random.default_rng(seed)

        # アームを初期姿勢へ
        self.robot.set_dofs_position(self.home_qpos)
        self.robot.set_dofs_velocity(np.zeros(self.n_dofs, dtype=np.float32))

        x_min, x_max = self.mat_x_start, self.mat_x_start + self.mat_length
        y_min, y_max = self.mat_y_start, self.mat_y_start + self.mat_width

        # 1. 紙コップの配置
        while True:
            cup_x = rng.uniform(x_min, x_max)
            cup_y = rng.uniform(y_min, y_max)
            if self._is_valid_spawn_area(cup_x, cup_y):
                break
        
        self.cup.set_pos(np.array([cup_x, cup_y, 0.015], dtype=np.float32))
        self.cup.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # self.cup.set_vel(np.zeros(6, dtype=np.float32)) # 落下速度をリセット

        # 2. スポンジの配置（コップと重ならないように）
        MIN_DIST = 0.06
        while True:
            sp_x = rng.uniform(x_min, x_max)
            sp_y = rng.uniform(y_min, y_max)
            if not self._is_valid_spawn_area(sp_x, sp_y): continue
            
            dist = np.sqrt((sp_x - cup_x)**2 + (sp_y - cup_y)**2)
            if dist >= MIN_DIST:
                break
                
        self.sponge.set_pos(np.array([sp_x, sp_y, 0.02], dtype=np.float32))
        self.sponge.set_quat(np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32))
        # self.sponge.set_vel(np.zeros(6, dtype=np.float32)) # 落下速度をリセット

        # 配置直後の物理演算を安定させるための空回し
        for _ in range(10):
            self.scene.step()

        print(f"↺ 配置更新 -> コップ:({cup_x:.3f}, {cup_y:.3f}) | スポンジ:({sp_x:.3f}, {sp_y:.3f})")
        # 本来は _get_obs() を返しますが、今回はテスト用なので None を返します
        return None, {}

if __name__ == "__main__":
    env = SO101GraspEnvTest()
    
    print("\n--- ランダム配置テストを開始します ---")
    print("2秒ごとに配置がリセットされます。ビューワーで位置を確認してください。")
    print("終了するには Ctrl+C を押してください。\n")

    try:
        while True:
            env.reset()
            # 2秒間シミュレーションを回す（1ステップ 0.01秒 × 200 = 2.0秒）
            for _ in range(200):
                env.scene.step()
    except KeyboardInterrupt:
        print("\nテスト終了")