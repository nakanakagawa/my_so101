# ============================================================
# Genesis SO-101アーム - 強化学習（PPO）Mac ビューワー版
# リアルタイムビューワーでシミュレーションを可視化
# ============================================================

# ============================================================
# STEP 0: 依存ライブラリのインストール（初回のみ）
# ============================================================
# pip install genesis-world stable-baselines3 gymnasium opencv-python


# so101_URDF みたいなフォルダが自動で作られ、URDFとSTLがダウンロードされる．

# ============================================================
# STEP 1: インポート
# ============================================================
import genesis as gs
import numpy as np
import json
import torch
import torch.nn as nn
import os
import re
import urllib.request
import matplotlib
matplotlib.use("Agg")   # ビューワーと競合しないバックエンド
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


# ============================================================
# STEP 2: SO-101 URDFをダウンロード（初回のみ）
# ============================================================
# URDF_DIR  = os.path.join(os.path.dirname(__file__), "so101_urdf")
# URDF_PATH = os.path.join(URDF_DIR, '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf')
URDF_PATH = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf'
def download_so101():
    os.makedirs(URDF_DIR, exist_ok=True)
    os.makedirs(os.path.join(URDF_DIR, "assets"), exist_ok=True)
    urdf_url = "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF/so101.urdf"
    print("SO-101 URDFをダウンロード中...")
    urllib.request.urlretrieve(urdf_url, URDF_PATH)
    print("✓ URDF ダウンロード完了")
    with open(URDF_PATH, "r") as f:
        content = f.read()
    stl_files = set(
        os.path.basename(m)
        for m in re.findall(r'filename="([^"]+\.stl)"', content, re.IGNORECASE)
    )
    base_url = "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF/assets/"
    for stl in stl_files:
        dest = os.path.join(URDF_DIR, "assets", stl)
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(base_url + stl, dest)
                print(f"  ✓ {stl}")
            except Exception as e:
                print(f"  ✗ {stl}: {e}")

if not os.path.exists(URDF_PATH):
    # download_so101()
    pass
else:
    print("✓ URDFはすでに存在します")

# ============================================================
# STEP 3: 学習用環境クラス（show_viewer=False、高速化優先）
# ============================================================
CAM_W, CAM_H = 320, 240   # 学習時は小さめで省メモリ

class SO101GraspEnv(gym.Env):
    """
    SO-101ロボットアームがキューブを把持して持ち上げるRL環境
    学習用（show_viewer=False）
    """
    metadata = {"render_modes": ["rgb_array"]}

    N_CUBES          = 5
    MAX_STEPS        = 500
    CUBE_LIFT_HEIGHT = 0.07
    ACTION_SCALE     = 0.05
    DT               = 0.01

    CUBE_X_MIN, CUBE_X_MAX = 0.05, 0.18
    CUBE_Y_MIN, CUBE_Y_MAX = -0.25, -0.02

    def __init__(self, render_mode=None, env_id=0):
        super().__init__()
        self.render_mode = render_mode
        self.env_id      = env_id
        self._step_count = 0

        try:
            gs.init(backend=gs.cpu, logging_level="warning")
        except Exception:
            pass

        self._build_scene()

        n = self.n_dofs
        obs_dim = 2 * n + 3 + self.N_CUBES * 3 + self.N_CUBES + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def _build_scene(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.DT, substeps=5),
            show_viewer=False,   # 学習中はビューワー非表示
        )
        self.scene.add_entity(gs.morphs.Plane())

        try:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 0), fixed=True)
            )
        except Exception as e:
            print(f"URDF読み込みエラー ({e})、Pandaで代替")
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            )

        default_positions = [
            (0.12, -0.08, 0.02), (0.12,  0.08, 0.02),
            (0.16,  0.00, 0.02), (0.18, -0.08, 0.02),
            (0.18,  0.08, 0.02),
        ]
        self.cubes = [
            self.scene.add_entity(
                gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=p, fixed=False)
            )
            for p in default_positions
        ]

        self.scene.build()
        self.n_dofs = self.robot.n_dofs
        print(f"✓ 学習環境構築完了 (DoF={self.n_dofs}, キューブ={self.N_CUBES}個)")

    def _get_eef_pos(self) -> np.ndarray:
        try:
            return self.robot.get_link("moving_jaw_so101_v1").get_pos().cpu().numpy()
        except Exception:
            return np.array([0.10, -0.15, 0.28], dtype=np.float32)

    def _get_cube_positions(self) -> np.ndarray:
        return np.stack([c.get_pos().cpu().numpy() for c in self.cubes])

    def _get_obs(self) -> np.ndarray:
        qpos      = self.robot.get_dofs_position().cpu().numpy()
        qvel      = self.robot.get_dofs_velocity().cpu().numpy()
        eef_pos   = self._get_eef_pos()
        cube_poss = self._get_cube_positions()
        dists     = np.linalg.norm(cube_poss - eef_pos, axis=1)
        nearest   = np.array([float(np.argmin(dists))])
        return np.concatenate([qpos, qvel, eef_pos, cube_poss.flatten(), dists, nearest]).astype(np.float32)

    def _compute_reward(self):
        eef_pos      = self._get_eef_pos()
        cube_poss    = self._get_cube_positions()
        dists        = np.linalg.norm(cube_poss - eef_pos, axis=1)
        nearest_idx  = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]
        nearest_z    = float(cube_poss[nearest_idx, 2])

        reward_reach   = -nearest_dist * 1.5
        reward_lift    = max(0.0, nearest_z - 0.02) * 8.0
        success        = nearest_z > self.CUBE_LIFT_HEIGHT
        reward_success = 30.0 if success else 0.0

        return float(reward_reach + reward_lift + reward_success - 0.01), success

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        rng = np.random.default_rng(seed)

        self.robot.set_dofs_position(np.zeros(self.n_dofs, dtype=np.float32))
        self.robot.set_dofs_velocity(np.zeros(self.n_dofs, dtype=np.float32))

        for cube in self.cubes:
            cx = rng.uniform(self.CUBE_X_MIN, self.CUBE_X_MAX)
            cy = rng.uniform(self.CUBE_Y_MIN, self.CUBE_Y_MAX)
            cube.set_pos(np.array([cx, cy, 0.02], dtype=np.float32))
            cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        for _ in range(10):
            self.scene.step()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        current_qpos = self.robot.get_dofs_position().cpu().numpy()
        delta        = np.clip(action, -1.0, 1.0) * self.ACTION_SCALE
        target_qpos  = np.clip(current_qpos + delta, -np.pi, np.pi)
        self.robot.set_dofs_position(target_qpos.astype(np.float32))
        self.scene.step()

        obs = self._get_obs()
        reward, success = self._compute_reward()
        terminated = success
        truncated  = self._step_count >= self.MAX_STEPS
        return obs, reward, terminated, truncated, {"success": success}

    def close(self):
        pass


# ============================================================
# STEP 4: リアルタイム・デモ環境クラス（show_viewer=True）
# ============================================================
class SO101ViewerEnv:
    """
    Macのビューワーでリアルタイム確認用。
    学習済みモデルを使ってデモ再生する。
    """
    N_CUBES          = 5
    CUBE_LIFT_HEIGHT = 0.07
    ACTION_SCALE     = 0.05
    DT               = 0.01

    CUBE_X_MIN, CUBE_X_MAX = 0.05, 0.18
    CUBE_Y_MIN, CUBE_Y_MAX = -0.25, -0.02

    def __init__(self):
        try:
            gs.init(backend=gs.cpu, logging_level="info")
        except Exception:
            pass

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.DT, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.8, -0.8, 0.6),
                camera_lookat=(0.15, 0.0, 0.1),
                camera_fov=50,
                max_FPS=60,
            ),
            show_viewer=True,    # ← Macビューワーを有効化
            show_FPS=True,
        )
        self.scene.add_entity(gs.morphs.Plane())

        try:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 0), fixed=True)
            )
        except Exception as e:
            print(f"URDF読み込みエラー ({e})、Pandaで代替")
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            )

        colors = [
            (1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0),
            (1.0, 1.0, 0.3), (1.0, 0.5, 0.0),
        ]
        self.cubes = [
            self.scene.add_entity(
                material=gs.materials.Rigid(rho=300),
                morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=p, fixed=False),
                surface=gs.surfaces.Default(color=(*c, 1.0)),
            )
            for p, c in zip(
                [(0.12,-0.08,0.02),(0.12,0.08,0.02),(0.16,0.0,0.02),(0.18,-0.08,0.02),(0.18,0.08,0.02)],
                colors,
            )
        ]

        self.scene.build()
        self.n_dofs = self.robot.n_dofs
        print(f"✓ ビューワー環境構築完了 (DoF={self.n_dofs})")

    def _get_eef_pos(self) -> np.ndarray:
        try:
            return self.robot.get_link("moving_jaw_so101_v1").get_pos().cpu().numpy()
        except Exception:
            return np.array([0.10, -0.15, 0.28], dtype=np.float32)

    def _get_cube_positions(self) -> np.ndarray:
        return np.stack([c.get_pos().cpu().numpy() for c in self.cubes])

    def _get_obs(self) -> np.ndarray:
        qpos      = self.robot.get_dofs_position().cpu().numpy()
        qvel      = self.robot.get_dofs_velocity().cpu().numpy()
        eef_pos   = self._get_eef_pos()
        cube_poss = self._get_cube_positions()
        dists     = np.linalg.norm(cube_poss - eef_pos, axis=1)
        nearest   = np.array([float(np.argmin(dists))])
        return np.concatenate([qpos, qvel, eef_pos, cube_poss.flatten(), dists, nearest]).astype(np.float32)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.robot.set_dofs_position(np.zeros(self.n_dofs, dtype=np.float32))
        self.robot.set_dofs_velocity(np.zeros(self.n_dofs, dtype=np.float32))

        for cube in self.cubes:
            cx = rng.uniform(self.CUBE_X_MIN, self.CUBE_X_MAX)
            cy = rng.uniform(self.CUBE_Y_MIN, self.CUBE_Y_MAX)
            cube.set_pos(np.array([cx, cy, 0.02], dtype=np.float32))
            cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        for _ in range(10):
            self.scene.step()

        return self._get_obs()

    def step(self, action: np.ndarray):
        current_qpos = self.robot.get_dofs_position().cpu().numpy()
        delta        = np.clip(action, -1.0, 1.0) * self.ACTION_SCALE
        target_qpos  = np.clip(current_qpos + delta, -np.pi, np.pi)
        self.robot.set_dofs_position(target_qpos.astype(np.float32))
        self.scene.step()

        obs       = self._get_obs()
        cube_poss = self._get_cube_positions()
        dists     = np.linalg.norm(cube_poss - self._get_eef_pos(), axis=1)
        max_z     = float(cube_poss[:, 2].max())
        success   = max_z > self.CUBE_LIFT_HEIGHT
        return obs, success


# ============================================================
# STEP 5: カスタムネットワーク
# ============================================================
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, features_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ============================================================
# STEP 6: 学習
# ============================================================
def train():
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "genesis_so101_rl")
    LOG_DIR  = os.path.join(SAVE_DIR, "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    N_ENVS = 4   # Mac CPUに合わせて並列数を削減（Colabの64→4）
    print(f"\n並列環境数: {N_ENVS}  保存先: {SAVE_DIR}\n")

    def make_env(env_id):
        def _init():
            return SO101GraspEnv(env_id=env_id)
        return _init

    envs = VecMonitor(DummyVecEnv([make_env(i) for i in range(N_ENVS)]), LOG_DIR)

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy", env=envs,
        learning_rate=lambda p: 3e-4 * p,
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR, verbose=1, device="cpu",
    )

    checkpoint_path = os.path.join(SAVE_DIR, "best_model.zip")
    if os.path.exists(checkpoint_path):
        try:
            model = PPO.load(checkpoint_path, env=envs, device="cpu")
            print(f"✓ チェックポイントをロード: {checkpoint_path}")
        except ValueError as e:
            print(f"⚠️ 観測空間不一致のため新規学習: {e}")
            import shutil
            shutil.move(checkpoint_path, checkpoint_path.replace(".zip", "_old.zip"))

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000, save_path=SAVE_DIR, name_prefix="ppo_so101"
    )
    eval_env = VecMonitor(DummyVecEnv([make_env(99)]))
    eval_cb  = EvalCallback(
        eval_env, best_model_save_path=SAVE_DIR, log_path=LOG_DIR,
        eval_freq=20_000, n_eval_episodes=10, deterministic=True, render=False,
    )

    TOTAL_TIMESTEPS = 60_000
    print(f"学習開始: {TOTAL_TIMESTEPS:,} ステップ\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_so101",
        reset_num_timesteps=False,
    )

    final_path = os.path.join(SAVE_DIR, "ppo_so101_final")
    model.save(final_path)
    print(f"\n✓ 学習完了! モデル保存: {final_path}.zip")

    envs.close()
    eval_env.close()
    return model, SAVE_DIR


def is_valid_spawn_area(x, y):
    """座標(x, y)が出現条件を満たしているか判定する関数"""
    # 条件: 黒マット・カメラ視野・リーチの共通領域（暫定の大枠範囲）
    if not (0.05 <= x <= 0.18 and -0.25 <= y <= -0.02):
        return False
        
    # ※ 後ほどここに「アームの陰になるエリアを除外する条件」を追加します
    return True

# ============================================================
# STEP 7: キーボードテレオペモード（環境確認用）
# ============================================================
"""
キーボード操作マッピング（SO-101 各関節）:
  F1 / F2   : Joint 0  (ベース回転)     +/-
  F3 / F4   : Joint 1  (肩ピッチ)       +/-
  F5 / F6   : Joint 2  (肘)             +/-
  F7 / F8   : Joint 3  (手首ピッチ)     +/-
  F9 / F10  : Joint 4  (手首ロール)     +/-
  - / =  : Joint 5  (グリッパー)     開/閉
  U         : リセット（キューブをランダム再配置）
  ESC       : 終了

  ※ viewer デフォルトキー（使わない）:
     i=ヘルプ  r=動画録画  s=画像保存  z=カメラリセット
"""

def run_keyboard_teleop():
    from genesis.vis.keybindings import Key, KeyAction, Keybind
    from matplotlib.path import Path  # 視野ポリゴンの判定に使用

    try:
        gs.init(backend=gs.cpu, logging_level="info")
    except Exception:
        pass

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=5),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.8, 0.6),
            camera_lookat=(0.15, 0.0, 0.1),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
        show_FPS=True,
    )

    scene.add_entity(gs.morphs.Plane())

    try:
        robot = scene.add_entity(
            # ここを修正：摩擦係数を強化
            material=gs.materials.Rigid(
                friction=2.0,      # アーム表面の摩擦
                coup_friction=3.0, # 把持力を強める (2.0 → 3.0 へ変更)
            ),
            morph=gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 0), fixed=True),
        )
        print("✓ SO-101 URDFをロード (把持強化設定)")
    except Exception as e:
        print(f"URDF読み込みエラー ({e})、Pandaで代替")
        # 代替のPandaロボットを使う場合も同様に摩擦を強化しておくと安心です
        robot = scene.add_entity(
            material=gs.materials.Rigid(
                friction=2.0,
                coup_friction=3.0,
            ),
            morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )

    # ──────────────────────────────────────────
    # 🧽 スポンジ（1個）の初期化
    # ──────────────────────────────────────────
    # スポンジ用の高摩擦マテリアル
    sponge_material = gs.materials.Rigid(
        rho=500, friction=2.0, coup_friction=5.0, coup_restitution=0.1
    )
    sponge = scene.add_entity(
        material=sponge_material,
        morph=gs.morphs.Box(size=(0.025, 0.025, 0.025), pos=(0.12, 0.08, 0.0125), fixed=False),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)), # 見分けやすいようにオレンジ色に設定
    )

    # ──────────────────────────────────────────
    # 🗑️ 紙コップ（1個）の初期化
    # ──────────────────────────────────────────
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    cup_urdf_path = os.path.join(parent_dir, "assets", "papercup.urdf")

    cup_material = gs.materials.Rigid(
        rho=500, friction=2.0, coup_friction=3.0, coup_restitution=0.1
    )
    cup = scene.add_entity(
        material=cup_material,
        morph=gs.morphs.URDF(
            file=cup_urdf_path,
            pos=(0.15, -0.15, 0.0),
            fixed=False
        )
    )

    # ──────────────────────────────────────────
    # カメラ追加（build()の前に追加する必要がある）
    # ──────────────────────────────────────────
    CAM_W, CAM_H = 400, 300

    # カメラ1: 真上からアーム全体を俯瞰
    cam_overhead = scene.add_camera(
        res=(CAM_W, CAM_H),
        pos=(0.15, 0.0, 1.2),
        lookat=(0.15, 0.0, 0.0),
        fov=55,
    )

    # カメラ2: グリッパー追従（毎ステップset_poseで更新）
    cam_wrist = scene.add_camera(
        res=(CAM_W, CAM_H),
        pos=(0.15, 0.0, 0.4),
        lookat=(0.15, 0.0, 0.0),
        fov=60,
    )

# ──────────────────────────────────────────
    # 📂 config JSONの読み込み と 幾何学計算
    # ──────────────────────────────────────────
    config_path = os.path.join(parent_dir, "config", "env_config.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            env_config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {config_path} が見つかりません。デフォルト値を使用します。")
        env_config = {
            "environment": {"mat_x_start": 0.13, "mat_y_start": -0.405, "mat_length": 0.30, "mat_width": 0.69, "mat_z_offset": 0.001, "desk_z_offset": -0.0054},
            "cameras": {"cam_fixed_side": {"x": 0.025, "y": -0.12, "z": 0.4546, "pitch_deg": -56.2, "fov_h_deg": 54.7, "fov_v_deg": 42.1}}
        }

    env_data = env_config["environment"]
    cam_data = env_config["cameras"]["cam_fixed_side"]

    mat_x_start = env_data["mat_x_start"]
    mat_y_start = env_data["mat_y_start"]
    mat_length  = env_data["mat_length"]
    mat_width   = env_data["mat_width"]
    mat_z       = env_data.get("desk_z_offset", -0.0054) + env_data.get("mat_z_offset", 0.001)

    # --- アームの把持可能範囲（極座標パラメータ） ---
    arm_reach_min = env_data.get("arm_reach_min", 0.10)
    arm_reach_max = env_data.get("arm_reach_max", 0.30)
    arm_yaw_min   = np.deg2rad(env_data.get("arm_yaw_min_deg", -90.0))
    arm_yaw_max   = np.deg2rad(env_data.get("arm_yaw_max_deg", 90.0))

    # --- shoulder_panのワールド座標を計算 ---
    # Genesisでのロボット配置座標 + URDFから読み取ったオフセット
    robot_x, robot_y = 0.04, 0.04
    shoulder_offset_x = 0.0388
    shoulder_offset_y = 0.0
    
    shoulder_x = robot_x + shoulder_offset_x
    shoulder_y = robot_y + shoulder_offset_y


    cam_x, cam_y, cam_z = cam_data["x"], cam_data["y"], cam_data["z"]

    # --- 固定カメラの視野（FOV）ポリゴン計算 ---
    pitch = np.deg2rad(cam_data["pitch_deg"])
    cam_dir   = np.array([np.cos(pitch), 0, np.sin(pitch)])
    cam_right = np.array([0, -1, 0])
    cam_up    = np.array([-np.sin(pitch), 0, np.cos(pitch)])

    w = np.tan(np.deg2rad(cam_data["fov_h_deg"]) / 2)
    h = np.tan(np.deg2rad(cam_data["fov_v_deg"]) / 2)

    rays = np.array([
        cam_dir + w * cam_right + h * cam_up,
        cam_dir - w * cam_right + h * cam_up,
        cam_dir - w * cam_right - h * cam_up,
        cam_dir + w * cam_right - h * cam_up
    ])

    # Z = mat_z 平面との交点を計算
    t = (mat_z - cam_z) / rays[:, 2]
    cam_pos = np.array([cam_x, cam_y, cam_z])
    intersect_pts = cam_pos + t[:, np.newaxis] * rays
    fov_poly = Path(intersect_pts[:, :2]) # ポリゴン生成

    # --- アームの把持可能範囲（ユーザー設定） ---
    ws_x_min = 0.02
    ws_x_max = 0.45
    # ws_y_min, ws_y_max の制限も追加する場合はここに

    # ──────────────────────────────────────────
    # 📍 エリア判定関数（マット寸法 ＋ カメラ視野 ＋ 極座標アーム範囲）
    # ──────────────────────────────────────────
    def is_valid_spawn_area(x, y):
        # 条件1: 完全に黒マットの内側であること
        if not (mat_x_start <= x <= (mat_x_start + mat_length)): return False
        if not (mat_y_start <= y <= (mat_y_start + mat_width)):  return False

        # 条件2: カメラの視野ポリゴンの内側であること
        if not fov_poly.contains_point((x, y)): return False

        # 条件3: アームの把持可能範囲（shoulder_pan基準の極座標）
        dx = x - shoulder_x
        dy = y - shoulder_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        if not (arm_reach_min <= r <= arm_reach_max): 
            return False
        if not (arm_yaw_min <= theta <= arm_yaw_max): 
            return False
            
        return True

    # ──────────────────────────────────────────
    # ⬛ 黒マットの表示（JSONから自動計算）
    # ──────────────────────────────────────────
    mat_center_x = mat_x_start + (mat_length / 2.0)
    mat_center_y = mat_y_start + (mat_width / 2.0)
    
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(mat_length, mat_width, 0.002),       
            pos=(mat_center_x, mat_center_y, 0.001),
            fixed=True
        ),
        surface=gs.surfaces.Default(color=(0.15, 0.15, 0.15, 1.0))
    )

# ──────────────────────────────────────────
    # 🟩 【デバッグ用】出現エリアの可視化（ホログラム仕様）
    # ──────────────────────────────────────────
    print("有効エリアのマーカーを生成中...")
    for x in np.arange(mat_x_start, mat_x_start + mat_length + 0.03, 0.03):
        for y in np.arange(mat_y_start, mat_y_start + mat_width + 0.03, 0.03):
            if is_valid_spawn_area(x, y):
                scene.add_entity(
                    morph=gs.morphs.Box(
                        size=(0.008, 0.008, 0.001), 
                        pos=(x, y, 0.0025), 
                        fixed=True,
                        collision=False  # 物理演算から除外
                    ),
                    surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 0.7))
                )

    scene.build()
    n_dofs = robot.n_dofs
    print(f"✓ シーン構築完了 (DoF={n_dofs})")

    def get_eef_pos():
        try:
            return robot.get_link("moving_jaw_so101_v1").get_pos().cpu().numpy()
        except Exception:
            return np.array([0.15, 0.0, 0.25], dtype=np.float32)

    def update_wrist_camera():
        """グリッパー位置に追従してwristカメラを更新"""
        eef = get_eef_pos()
        # アームの根元方向（原点）を常に向く
        cam_pos  = eef + np.array([0.0, 0.0, 0.12])   # グリッパーの12cm真上
        lookat   = eef + np.array([0.0, 0.0, -0.08])  # グリッパー先端下向き
        try:
            cam_wrist.set_pose(pos=cam_pos, lookat=lookat)
        except Exception:
            pass

    def grab_frame(cam):
        try:
            frame = cam.render(rgb=True)
            if isinstance(frame, (list, tuple)):
                frame = frame[0]
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()
            return frame.astype(np.uint8)
        except Exception:
            return np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    # OpenCVウィンドウ初期配置（メインビューワーの右側を想定）
    import cv2
    MAIN_WIN_W = 1000   # メインビューワーの幅の目安
    cv2.namedWindow("overhead", cv2.WINDOW_NORMAL)
    cv2.namedWindow("wrist",    cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overhead", CAM_W, CAM_H)
    cv2.resizeWindow("wrist",    CAM_W, CAM_H)
    cv2.moveWindow("overhead", MAIN_WIN_W + 20, 50)
    cv2.moveWindow("wrist",    MAIN_WIN_W + 20, 50 + CAM_H + 40)

    # カメラ更新間隔（毎ステップだと重いので数ステップに1回）
    CAM_INTERVAL = 5
    step_count = 0

    print(__doc__)  # キー操作ガイドを表示

    dq = 0.03   # 1ステップあたりの関節角変化量

    # サーボ目標角度（現在位置で初期化）
    target_qpos = robot.get_dofs_position().cpu().numpy().astype(np.float32)

    # PD制御ゲイン（実機サーボの保持力をイメージ）
    KP = 200.0   # 位置ゲイン（大きいほど保持力強）
    KD = 20.0    # 速度ゲイン（振動を抑える）

    def move_joint(idx, delta):
        current = robot.get_dofs_position().cpu().numpy().astype(np.float32)
        target_qpos[idx] = float(np.clip(current[idx] + delta, -np.pi, np.pi))

    def apply_pd_control():
        """毎ステップ呼ぶ。目標角度に向けてトルクを出し続ける（サーボ保持）"""
        current_pos = robot.get_dofs_position().cpu().numpy().astype(np.float32)
        current_vel = robot.get_dofs_velocity().cpu().numpy().astype(np.float32)
        torque = KP * (target_qpos - current_pos) - KD * current_vel
        robot.control_dofs_force(torque)



    def reset_scene():
        target_qpos[:] = 0.0
        robot.set_dofs_position(target_qpos)
        rng = np.random.default_rng()
        
        MIN_DIST = 0.06

        # ★ 探索範囲をマットの寸法から動的に取得
        # X: mat_x_start 〜 (mat_x_start + mat_length)
        # Y: mat_y_start 〜 (mat_y_start + mat_width)
        x_min, x_max = mat_x_start, mat_x_start + mat_length
        y_min, y_max = mat_y_start, mat_y_start + mat_width

        # 1. 紙コップの配置座標を決定
        while True:
            cup_x = rng.uniform(x_min, x_max)
            cup_y = rng.uniform(y_min, y_max)
            if is_valid_spawn_area(cup_x, cup_y):
                break
        
        cup.set_pos(np.array([cup_x, cup_y, 0.015], dtype=np.float32))
        cup.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        # 2. スポンジの配置座標を決定
        while True:
            sp_x = rng.uniform(x_min, x_max)
            sp_y = rng.uniform(y_min, y_max)
            if not is_valid_spawn_area(sp_x, sp_y):
                continue
            dist = np.sqrt((sp_x - cup_x)**2 + (sp_y - cup_y)**2)
            if dist >= MIN_DIST:
                break
                
        sponge.set_pos(np.array([sp_x, sp_y, 0.0125], dtype=np.float32))
        sponge.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        
        print(f"↺ リセット完了 (生成範囲 X:{x_min:.2f}-{x_max:.2f}, Y:{y_min:.2f}-{y_max:.2f})")

    is_running = True
    def stop():
        nonlocal is_running
        is_running = False

    # DOFインデックスを安全に解決（DOF数に応じてクランプ）
    def dof(i):
        return min(i, n_dofs - 1)

    scene.viewer.register_keybinds(
        # Joint 0: 1(+) / 2(-)
        Keybind("j0_pos", Key._1, KeyAction.HOLD, callback=move_joint, args=(dof(0),  dq)),
        Keybind("j0_neg", Key._2, KeyAction.HOLD, callback=move_joint, args=(dof(0), -dq)),
        # Joint 1: 3(+) / 4(-)
        Keybind("j1_pos", Key._3, KeyAction.HOLD, callback=move_joint, args=(dof(1),  dq)),
        Keybind("j1_neg", Key._4, KeyAction.HOLD, callback=move_joint, args=(dof(1), -dq)),
        # Joint 2: 5(+) / 6(-)
        Keybind("j2_pos", Key._5, KeyAction.HOLD, callback=move_joint, args=(dof(2),  dq)),
        Keybind("j2_neg", Key._6, KeyAction.HOLD, callback=move_joint, args=(dof(2), -dq)),
        # Joint 3: 7(+) / 8(-)
        Keybind("j3_pos", Key._7, KeyAction.HOLD, callback=move_joint, args=(dof(3),  dq)),
        Keybind("j3_neg", Key._8, KeyAction.HOLD, callback=move_joint, args=(dof(3), -dq)),
        # Joint 4: 9(+) / 0(-)
        Keybind("j4_pos", Key._9, KeyAction.HOLD, callback=move_joint, args=(dof(4),  dq)),
        Keybind("j4_neg", Key._0, KeyAction.HOLD, callback=move_joint, args=(dof(4), -dq)),
        # Joint 5 (グリッパー): -(+) / =(-)
        Keybind("j5_pos", Key.MINUS, KeyAction.HOLD, callback=move_joint, args=(dof(5),  dq)),
        Keybind("j5_neg", Key.EQUAL, KeyAction.HOLD, callback=move_joint, args=(dof(5), -dq)),
        # リセット・終了
        Keybind("reset", Key.U,      KeyAction.PRESS, callback=reset_scene),
        Keybind("quit",  Key.ESCAPE, KeyAction.PRESS, callback=stop),
    )

    try:
        while is_running:
            apply_pd_control()
            scene.step()

            # カメラ更新（数ステップに1回）
            step_count += 1
            if step_count % CAM_INTERVAL == 0:
                update_wrist_camera()

                img_overhead = grab_frame(cam_overhead)
                img_wrist    = grab_frame(cam_wrist)

                # ラベルを描画
                def draw_label(img, text):
                    cv2.rectangle(img, (0, 0), (len(text) * 11 + 10, 28), (0, 0, 0), -1)
                    cv2.putText(img, text, (6, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)
                    return img

                draw_label(img_overhead, "OVERHEAD")
                draw_label(img_wrist,    "WRIST")

                # RGB→BGR変換してOpenCVで表示
                cv2.imshow("overhead", cv2.cvtColor(img_overhead, cv2.COLOR_RGB2BGR))
                cv2.imshow("wrist",    cv2.cvtColor(img_wrist,    cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("✓ キーボード操作モード終了")


# ============================================================
# STEP 8: リアルタイムビューワーでデモ再生
# ============================================================
def run_viewer_demo(model, n_episodes: int = 5):
    """
    学習済みモデルをMacビューワーでリアルタイム再生する。
    ビューワーウィンドウを閉じると終了。
    """
    print("\n--- Macビューワーでリアルタイムデモを開始 ---")
    print("ビューワーウィンドウを閉じると終了します。\n")

    env = SO101ViewerEnv()

    for ep in range(n_episodes):
        obs      = env.reset(seed=ep)
        done     = False
        total_r  = 0.0
        step     = 0
        MAX_STEP = 500

        print(f"Episode {ep + 1}/{n_episodes} 開始...")

        while not done and step < MAX_STEP:
            action, _ = model.predict(obs, deterministic=True)
            obs, success = env.step(action)
            step += 1

            if success:
                print(f"  ✓ SUCCESS! ({step}ステップ)")
                done = True

        if not done:
            print(f"  Episode {ep + 1}: {step}ステップ完了 (タイムアウト)")

    print("\n✓ デモ終了")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SO-101 RL Mac版",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "demo", "traindemo"],
        default=None,
        help=(
            "（省略）      : キーボードで環境を手動確認\n"
            "train        : 学習のみ\n"
            "demo         : 既存モデルでビューワーデモ\n"
            "traindemo    : 学習後にビューワーデモ"
        ),
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="demo モード時に使う .zip モデルパス（省略時は自動検索）",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="デモエピソード数（デフォルト: 5）",
    )
    args = parser.parse_args()

    SAVE_DIR = os.path.join(os.path.dirname(__file__), "genesis_so101_rl")

    # --------------------------------------------------
    # 引数なし → キーボードテレオペで環境確認
    # --------------------------------------------------
    if args.mode is None:
        print("=" * 50)
        print(" SO-101 キーボード操作モード（環境確認用）")
        print("=" * 50)
        run_keyboard_teleop()
        raise SystemExit

    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    if args.mode in ("train", "traindemo"):
        model, SAVE_DIR = train()

    # --------------------------------------------------
    # デモ
    # --------------------------------------------------
    if args.mode in ("demo", "traindemo"):
        if args.model_path:
            model_path = args.model_path
        else:
            candidates = [
                os.path.join(SAVE_DIR, "ppo_so101_final.zip"),
                os.path.join(SAVE_DIR, "best_model.zip"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

        if args.mode == "demo":
            if model_path and os.path.exists(model_path):
                print(f"モデルをロード: {model_path}")
                dummy_env = VecMonitor(DummyVecEnv([lambda: SO101GraspEnv()]))
                model = PPO.load(model_path, env=dummy_env, device="cpu")
                dummy_env.close()
            else:
                print("⚠️ モデルが見つかりません。先に --mode train を実行してください。")
                raise SystemExit

        run_viewer_demo(model, n_episodes=args.episodes)

    print("\n✅ 完了!")
    if args.mode in ("train", "traindemo"):
        print(f"\n📊 TensorBoard で学習曲線を確認:")
        print(f"  tensorboard --logdir {SAVE_DIR}/logs")
