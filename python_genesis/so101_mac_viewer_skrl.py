# ============================================================
# Genesis SO-101アーム - 強化学習（PPO）Mac ビューワー版
# リアルタイムビューワーでシミュレーションを可視化
# ============================================================

# ============================================================
# Genesis SO-101アーム - 強化学習（PPO） GPU並列・爆速版
# ============================================================
import genesis as gs
import numpy as np
import yaml
import torch
import torch.nn as nn
import os
import re
import urllib.request
import time
import gymnasium as gym
from scipy.spatial.transform import Rotation as R 

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.memories.torch import RandomMemory
import skrl.envs.wrappers.torch as skrl_wrappers

# ============================================================
# STEP 1: 設定とURDFダウンロード
# ============================================================
URDF_PATH = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf'

# 並列数をここで一元管理（VRAM 8GBなら16〜32が目安。まずは16で確実な爆速を体感してください）
N_ENVS = 16

# ============================================================
# STEP 2: GPUネイティブ環境クラス（Genesisバッチ処理対応版）
# ============================================================
class SO101GraspEnvBatched:
    """
    Genesisのバッチ機能（B_envs）を活用し、GPU上で全環境を並列計算するクラス。
    NumPyへの変換を排除し、すべてPyTorchテンソルで処理します。
    """
    MAX_STEPS = 500
    ACTION_SCALE = 0.05
    DT = 0.01

    def __init__(self, num_envs=N_ENVS, show_viewer=False):
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        
        # skrlが要求するデバイス
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # バッチ管理用のフラグとカウンター (すべてテンソル)
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.is_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_frame_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 把持した瞬間の相対位置・姿勢（バッチサイズ分保持）
        self.relative_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        try:
            gs.init(backend=gs.cuda, logging_level="warning")
        except Exception:
            pass

        self._load_config()
        self._build_scene()

        self.n_dofs = self.robot.n_dofs
        obs_dim = 2 * self.n_dofs + 14 
        
        # skrl用の空間定義
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_dofs,), dtype=np.float32)
        
        # skrlが直接要求するプロパティ群
        self.state_space = self.observation_space
        self.shared_observation_space = None
        self.unwrapped = self
        
        # 毎ステップ使い回すバッファ
        self.obs_buf = torch.zeros((self.num_envs, obs_dim), dtype=torch.float32, device=self.device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_config(self):
        import yaml
        from matplotlib.path import Path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, "config", "env_config.yaml")
        self.cup_urdf_path = os.path.join(parent_dir, "assets", "papercup.urdf")

        with open(config_path, "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f)

        env_data = env_config["environment"]
        cam_cfg = env_config["cameras"]["cam_fixed_side"]

        self.mat_x_start = env_data["mat_x_start"]
        self.mat_y_start = env_data["mat_y_start"]
        self.mat_length  = env_data["mat_length"]
        self.mat_width   = env_data["mat_width"]
        self.mat_z       = env_data.get("desk_z_offset", -0.0054) + env_data.get("mat_z_offset", 0.001)

        self.arm_reach_min = env_data.get("arm_reach_min", 0.10)
        self.arm_reach_max = env_data.get("arm_reach_max", 0.30)
        self.arm_yaw_min   = np.deg2rad(env_data.get("arm_yaw_min_deg", -90.0))
        self.arm_yaw_max   = np.deg2rad(env_data.get("arm_yaw_max_deg", 90.0))
        self.shoulder_x = 0.04 + 0.0388
        self.shoulder_y = 0.04 + 0.0
        self.deadzone_x_min, self.deadzone_x_max = 0.0, 0.25
        self.deadzone_y_min, self.deadzone_y_max = 0.0, 0.15

        # カメラ視野ポリゴンの計算 (CPU側で行う初期化処理なのでNumPyのままでOK)
        cam_x, cam_y, cam_z = cam_cfg["x"], cam_cfg["y"], cam_cfg["z"]
        pitch = np.deg2rad(cam_cfg["pitch_deg"])
        cam_dir   = np.array([np.cos(pitch), 0, np.sin(pitch)])
        cam_right = np.array([0, -1, 0])
        cam_up    = np.array([-np.sin(pitch), 0, np.cos(pitch)])

        w = np.tan(np.deg2rad(cam_cfg["fov_h_deg"]) / 2)
        h = np.tan(np.deg2rad(cam_cfg["fov_v_deg"]) / 2)
        rays = np.array([
            cam_dir + w * cam_right + h * cam_up, cam_dir - w * cam_right + h * cam_up,
            cam_dir - w * cam_right - h * cam_up, cam_dir + w * cam_right - h * cam_up
        ])
        t = (self.mat_z - cam_z) / rays[:, 2]
        intersect_pts = np.array([cam_x, cam_y, cam_z]) + t[:, np.newaxis] * rays
        self.fov_poly = Path(intersect_pts[:, :2])
        self.home_qpos = torch.tensor(env_data.get("home_qpos", [0.0]*6), dtype=torch.float32, device=self.device)

    def _is_valid_spawn_area(self, x, y):
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
        # 【超重要】Sceneに B (バッチサイズ) を渡すことで、GenesisがGPU上でnum_envs個の世界を同時生成する
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.DT, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.8, -0.8, 0.6), camera_lookat=(0.15, 0.0, 0.1),
            ) if self.show_viewer else None,
            show_viewer=self.show_viewer
        )
        self.scene.add_entity(gs.morphs.Plane())

        # ロボット
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 0), fixed=True)
        )
        # スポンジ
        self.sponge = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500, friction=5.0, coup_friction=5.0),
            morph=gs.morphs.Box(size=(0.04, 0.03, 0.03), pos=(0.1, 0, 0.1), fixed=False),
            surface=gs.surfaces.Default(color=(0.6, 1.0, 0.0, 1.0))
        )
        # コップ
        self.cup = self.scene.add_entity(
            material=gs.materials.Rigid(rho=500, friction=2.0, coup_friction=3.0),
            morph=gs.morphs.URDF(file=self.cup_urdf_path, pos=(0.1, 0.1, 0.1), fixed=False)
        )

        # GPU上でnum_envs個の世界を一斉にビルド
        self.scene.build(n_envs=self.num_envs)

        try:
            self.gripper_idx = self.robot.get_joint("gripper").dof_idx_local
        except Exception:
            self.gripper_idx = 5
            
        # Z方向のダウンベクトル (計算用に常に用意しておく)
        self.down_dir_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

    def _get_jaw_pos_batched(self):
        # テンソルのまま (N_ENVS, 3) と (N_ENVS, 4) を取得
        g_pos = self.robot.get_link("gripper_link").get_pos()
        g_quat = self.robot.get_link("gripper_link").get_quat()
        
        # PyTorch版のクォータニオン回転関数 (簡易実装: ベクトルをZ軸方向に回す)
        # クォータニオン q = [w, x, y, z] を想定
        w, x, y, z = g_quat[:, 0], g_quat[:, 1], g_quat[:, 2], g_quat[:, 3]
        
        # jaw_offset_local = [0.01, 0.0, -0.09] を回転させる
        offset_x, offset_y, offset_z = 0.01, 0.0, -0.09
        
        rot_x = (1 - 2*y**2 - 2*z**2)*offset_x + (2*x*y - 2*w*z)*offset_y + (2*x*z + 2*w*y)*offset_z
        rot_y = (2*x*y + 2*w*z)*offset_x + (1 - 2*x**2 - 2*z**2)*offset_y + (2*y*z - 2*w*x)*offset_z
        rot_z = (2*x*z - 2*w*y)*offset_x + (2*y*z + 2*w*x)*offset_y + (1 - 2*x**2 - 2*y**2)*offset_z
        
        jaw_rot = torch.stack([rot_x, rot_y, rot_z], dim=1)
        return g_pos + jaw_rot

    def _get_obs(self):
        qpos = self.robot.get_dofs_position()
        qvel = self.robot.get_dofs_velocity()
        jaw_pos = self._get_jaw_pos_batched()
        s_pos = self.sponge.get_pos()
        s_quat = self.sponge.get_quat()
        c_pos = self.cup.get_pos()
        
        grasp_flag = self.is_grasping.to(torch.float32).unsqueeze(1)
        
        # (N_ENVS, obs_dim) に沿って結合
        self.obs_buf = torch.cat([qpos, qvel, jaw_pos, s_pos, s_quat, c_pos, grasp_flag], dim=1)
        return self.obs_buf

    def _compute_reward(self, current_qpos):
        jaw_pos = self._get_jaw_pos_batched()
        s_pos = self.sponge.get_pos()
        c_pos = self.cup.get_pos()
        
        dx = jaw_pos[:, 0] - s_pos[:, 0]
        dy = jaw_pos[:, 1] - s_pos[:, 1]
        dz = jaw_pos[:, 2] - s_pos[:, 2]
        
        xy_dist = torch.sqrt(dx**2 + dy**2)
        z_dist = torch.abs(dz)
        dist_to_sponge = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # --- ペナルティ計算 (全環境を一括でTorchテンソル演算) ---
        not_grasping = ~self.is_grasping
        
        # 1. 接近ペナルティ
        reward[not_grasping] += -(xy_dist[not_grasping] * 2.0 + z_dist[not_grasping] * 1.0) - 1.2
        
        # 2. グリッパー開度ペナルティ
        gripper_angle = current_qpos[:, self.gripper_idx]
        
        far_mask = not_grasping & (xy_dist > 0.015)
        target_angle_far = 45.0 * np.pi / 180.0
        reward[far_mask] += -torch.abs(gripper_angle[far_mask] - target_angle_far) * 1.0
        
        close_mask = not_grasping & (xy_dist <= 0.015) & (dist_to_sponge <= 0.025)
        target_angle_close = 0.0
        reward[close_mask] += -torch.abs(gripper_angle[close_mask] - target_angle_close) * 1.0

        # 3. 持ち上げ & 4. 運搬ペナルティ
        grasping = self.is_grasping
        lift_penalty = torch.clamp(0.10 - s_pos[:, 2], min=0.0) * 2.0
        reward[grasping] += -lift_penalty[grasping]
        
        dist_to_cup = torch.sqrt(torch.sum((s_pos - c_pos)**2, dim=1))
        reward[grasping] += -(dist_to_cup[grasping] * 1.0) - 0.2

        # 4.5 速度ペナルティ
        qvel = self.robot.get_dofs_velocity()
        # グリッパー以外の速度の2乗和
        arm_qvel_sq = torch.sum(qvel**2, dim=1) - qvel[:, self.gripper_idx]**2
        reward += -arm_qvel_sq * 0.005

        # 5. 成功報酬
        reward[self._is_success] += 100.0
        
        self.reward_buf = reward
        return self.reward_buf

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        n_reset = len(env_ids)
        if n_reset == 0:
            return self._get_obs(), {}

        # 状態リセット
        self.step_counts[env_ids] = 0
        self.is_grasping[env_ids] = False
        self.success_frame_counts[env_ids] = 0
        self._is_success[env_ids] = False
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False
        
        # ロボットリセット
        qpos_reset = self.home_qpos.repeat(n_reset, 1)
        qvel_reset = torch.zeros((n_reset, self.n_dofs), device=self.device)

        # 【修正】 envs_idx= を明記して渡す
        self.robot.set_dofs_position(qpos_reset, envs_idx=env_ids)
        self.robot.set_dofs_velocity(qvel_reset, envs_idx=env_ids)
        
        # スポンジ・コップのランダム配置 (ここだけCPUで計算してGPUに送る)
        cup_pos_list, sponge_pos_list = [], []
        for _ in range(n_reset):
            while True:
                cup_x, cup_y = np.random.uniform(self.mat_x_start, self.mat_x_start + self.mat_length), np.random.uniform(self.mat_y_start, self.mat_y_start + self.mat_width)
                if self._is_valid_spawn_area(cup_x, cup_y): break
            cup_pos_list.append([cup_x, cup_y, 0.015])
            
            while True:
                sp_x, sp_y = np.random.uniform(self.mat_x_start, self.mat_x_start + self.mat_length), np.random.uniform(self.mat_y_start, self.mat_y_start + self.mat_width)
                if not self._is_valid_spawn_area(sp_x, sp_y): continue
                if np.sqrt((sp_x - cup_x)**2 + (sp_y - cup_y)**2) >= 0.06: break
            sponge_pos_list.append([sp_x, sp_y, 0.02])

        # 【修正】 envs_idx= を明記して渡す
        self.cup.set_pos(torch.tensor(cup_pos_list, dtype=torch.float32, device=self.device), envs_idx=env_ids)
        self.cup.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(n_reset, 1), envs_idx=env_ids)
        
        self.sponge.set_pos(torch.tensor(sponge_pos_list, dtype=torch.float32, device=self.device), envs_idx=env_ids)
        self.sponge.set_quat(torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=self.device).repeat(n_reset, 1), envs_idx=env_ids)
        
        # リセット時の安定化ステップ
        self.scene.step()
        
        return self._get_obs(), {}

    def step(self, actions: torch.Tensor):
        import time
        t_start = time.perf_counter()
        
        self.step_counts += 1
        
        # --------------------------------------------------
        # 1. アクション適用と物理演算の計測
        # --------------------------------------------------
        current_qpos = self.robot.get_dofs_position()
        delta = torch.clamp(actions, -1.0, 1.0) * self.ACTION_SCALE
        target_qpos = current_qpos + delta
        target_qpos[:, self.gripper_idx] = torch.clamp(target_qpos[:, self.gripper_idx], -0.174, 1.745)
        
        # ※ここに先ほどの `control_dofs_position` の修正を入れても入れなくても計測は可能です。
        # 現在のコードのまま (set_dofs_position) 計測します。
        self.robot.set_dofs_position(target_qpos)
        
        self.scene.step()
        
        # PyTorchにGPU計算の完了を待たせる（正確な時間計測のため）
        torch.cuda.synchronize()
        t_phys = time.perf_counter()

        # --------------------------------------------------
        # 2. 状態取得と仮想拘束・内外判定の計測
        # --------------------------------------------------
        g_pos = self.robot.get_link("gripper_link").get_pos()
        jaw_pos = self._get_jaw_pos_batched()
        s_pos = self.sponge.get_pos()
        
        dx = jaw_pos[:, 0] - s_pos[:, 0]
        dy = jaw_pos[:, 1] - s_pos[:, 1]
        dz = jaw_pos[:, 2] - s_pos[:, 2]
        xy_dist = torch.sqrt(dx**2 + dy**2)
        z_dist = torch.abs(dz)
        
        is_close = (xy_dist <= 0.020) & (z_dist <= 0.035)
        g_angle = target_qpos[:, self.gripper_idx]
        is_squeezing = g_angle <= 0.20
        
        new_grasp_mask = (~self.is_grasping) & is_close & is_squeezing
        self.is_grasping[new_grasp_mask] = True
        
        grasping_mask = self.is_grasping
        if grasping_mask.any():
            target_sponge_pos = jaw_pos.clone()
            target_sponge_pos[:, 2] -= 0.015
            current_s_pos = self.sponge.get_pos()
            current_s_pos[grasping_mask] = target_sponge_pos[grasping_mask]
            self.sponge.set_pos(current_s_pos)

        release_mask = self.is_grasping & (g_angle > 0.35)
        self.is_grasping[release_mask] = False

        c_pos = self.cup.get_pos()
        in_x = torch.abs(s_pos[:, 0] - c_pos[:, 0]) < 0.035
        in_y = torch.abs(s_pos[:, 1] - c_pos[:, 1]) < 0.035
        in_z = (s_pos[:, 2] - c_pos[:, 2] > -0.015) & (s_pos[:, 2] - c_pos[:, 2] < 0.060)
        
        is_in_cup = in_x & in_y & in_z & (~self.is_grasping)
        self.success_frame_counts[is_in_cup] += 1
        self.success_frame_counts[~is_in_cup] = 0
        self._is_success = self.success_frame_counts >= 50
        
        torch.cuda.synchronize()
        t_logic = time.perf_counter()

        # --------------------------------------------------
        # 3. 報酬計算とリセット処理の計測
        # --------------------------------------------------
        obs = self._get_obs()
        reward = self._compute_reward(target_qpos)
        
        self.terminated_buf = self._is_success.clone()
        self.truncated_buf = self.step_counts >= self.MAX_STEPS
        
        reset_ids = (self.terminated_buf | self.truncated_buf).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset(env_ids=reset_ids)
            obs = self._get_obs()
            
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        # --- 時間の集計と出力 ---
        if not hasattr(self, '_profile_stats'):
            self._profile_stats = {'phys': 0.0, 'logic': 0.0, 'reward_reset': 0.0}
            self._profile_count = 0
            
        self._profile_stats['phys'] += (t_phys - t_start)
        self._profile_stats['logic'] += (t_logic - t_phys)
        self._profile_stats['reward_reset'] += (t_end - t_logic)
        self._profile_count += 1
        
        if self._profile_count % 1000 == 0:
            print("\n=== GPUバッチ処理時間 (1ステップ平均 / 1000歩計測) ===")
            print(f"1. 物理演算 (scene.step 等) : {self._profile_stats['phys']/1000 * 1000:.3f} ms")
            print(f"2. 状態判定 (仮想拘束 等)   : {self._profile_stats['logic']/1000 * 1000:.3f} ms")
            print(f"3. 報酬・リセット処理       : {self._profile_stats['reward_reset']/1000 * 1000:.3f} ms")
            print("========================================================\n")
            for k in self._profile_stats: self._profile_stats[k] = 0.0
        
        info = {}
        return obs, reward.unsqueeze(1), self.terminated_buf.unsqueeze(1), self.truncated_buf.unsqueeze(1), info
    def state(self):
        return self.obs_buf
    def close(self): pass
    def render(self, *args, **kwargs): pass


# ============================================================
# STEP 3: skrl エージェント・ネットワーク設定
# ============================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = torch.zeros((1, self.num_observations), device=self.device)
        return self.net(states), {"log_std": self.log_std_parameter}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = torch.zeros((1, self.num_observations), device=self.device)
        return self.net(states), {}


# ============================================================
# STEP 4: 学習メイン関数
# ============================================================
def train():
    print(f"\n🚀 [GPU Native Mode] 爆速バッチ環境を構築中... (N_ENVS={N_ENVS})\n")
    
    # 完全にPyTorch/GPUネイティブなバッチ環境を1つインスタンス化
    env = SO101GraspEnvBatched(num_envs=N_ENVS, show_viewer=False)
    device = env.device

    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device)
    }

    cfg = PPO_CFG()
    cfg.rollouts = 1024  
    cfg.learning_epochs = 10
    cfg.mini_batches = 4
    cfg.discount_factor = 0.99
    cfg.gae_lambda = 0.95
    cfg.learning_rate = 3e-4
    cfg.random_timesteps = 0
    cfg.learning_starts = 0
    cfg.timesteps = 500000
    
    cfg.state_preprocessor = None
    cfg.value_preprocessor = None
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesis_so101_rl40", "logs")
    os.makedirs(log_dir, exist_ok=True)
    cfg.experiment.directory = log_dir
    cfg.experiment.experiment_name = "ppo_so101_skrl_gpu_native"

    memory = RandomMemory(memory_size=1024, num_envs=N_ENVS, device=device)

    agent = PPO(models=models, memory=memory, cfg=cfg,
                observation_space=env.observation_space, action_space=env.action_space, device=device)

    trainer_cfg = SequentialTrainerCfg()
    trainer_cfg.timesteps = 500000
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    
    print("\n⚡ シミュレーション開始: GPUの真の力を見せます ⚡\n")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    fps = 500000 / (end_time - start_time)
    print(f"\n🏁 学習完了: 500,000 steps in {end_time - start_time:.2f} sec (Average FPS: {fps:.2f})")
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesis_so101_rl40", "best_model.pt")
    agent.save(save_path)
    print(f"✅ モデルを保存しました: {save_path}")

    return agent, os.path.dirname(save_path)



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
        # gs.init(backend=gs.cpu, logging_level="info")
        gs.init(backend=gs.cuda, logging_level="warning")
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
    # 🧽 スポンジの初期化
    # ──────────────────────────────────────────
    sponge_material = gs.materials.Rigid(
        rho=500, friction=5.0, coup_friction=5.0, coup_restitution=0.0
    )
    
    # 直方体の場合、サイズは (length, width, height) で指定します
    # ここでは 4cm x 3cm x 2cm のスポンジを想定しています
    sponge_size = (0.04, 0.03, 0.03)
    
    sponge = scene.add_entity(
        material=sponge_material,
        morph=gs.morphs.Box(
            size=sponge_size,
            pos=(0.12, 0.08, sponge_size[2] / 2.0), # Z座標は高さの半分に設定
            fixed=False
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)),
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
    # 📂 config JSONの読み込み と 幾何学計算
    # ──────────────────────────────────────────
    config_path = os.path.join(parent_dir, "config", "env_config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            env_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ {config_path} が見つかりません。デフォルト値を使用します。")
        env_config = {
            "environment": {"mat_x_start": 0.13, "mat_y_start": -0.405, "mat_length": 0.30, "mat_width": 0.69, "mat_z_offset": 0.001, "desk_z_offset": -0.0054},
            "cameras": {"cam_fixed_side": {"x": 0.025, "y": -0.12, "z": 0.4546, "pitch_deg": -56.2, "fov_h_deg": 54.7, "fov_v_deg": 42.1},
                        "cam_hand": {"fov_v_deg": 49.1}} # フォールバック用
        }

# 各種設定データを抽出
    env_data = env_config["environment"]
    cam_fixed_cfg = env_config["cameras"]["cam_fixed_side"]
    cam_hand_cfg = env_config["cameras"]["cam_hand"]

    mat_x_start = env_data["mat_x_start"]
    mat_y_start = env_data["mat_y_start"]
    mat_length  = env_data["mat_length"]
    mat_width   = env_data["mat_width"]
    mat_z       = env_data.get("desk_z_offset", -0.0054) + env_data.get("mat_z_offset", 0.001)


    # --- アームの初期姿勢（ホームポジション）を読み込む ---
    home_qpos = np.array(env_data.get("home_qpos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)

    # ──────────────────────────────────────────
    # 📷 固定カメラの視線ベクトル（lookat）計算
    # ──────────────────────────────────────────
    fx, fy, fz = cam_fixed_cfg["x"], cam_fixed_cfg["y"], cam_fixed_cfg["z"]
    pitch = np.deg2rad(cam_fixed_cfg["pitch_deg"])
    
    # ピッチ角から向いている方向（ベクトル）を計算し、カメラ位置に足して注視点を作る
    cam_dir = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
    lookat_target = np.array([fx, fy, fz]) + cam_dir

    # ──────────────────────────────────────────
    # カメラ追加（build()の前に追加する必要がある）
    # ──────────────────────────────────────────
    CAM_W, CAM_H = 400, 300

    # カメラ1: YAML設定に基づいた固定カメラ
    cam_overhead = scene.add_camera(
        res=(CAM_W, CAM_H),
        pos=(fx, fy, fz),
        lookat=lookat_target,
        fov=cam_fixed_cfg["fov_v_deg"],
        near=0.001,
    )

    # カメラ2: グリッパー追従カメラ
    # ※位置と注視点はシミュレーション中に毎ステップ上書きされるため、ここでは初期値(ダミー)でOK。
    # ※ただし、視野角(fov)だけはここでYAMLの値を設定しておく必要があります。
    cam_wrist = scene.add_camera(
        res=(CAM_W, CAM_H),
        pos=(0.15, 0.0, 0.4),
        lookat=(0.15, 0.0, 0.0),
        fov=cam_hand_cfg["fov_v_deg"],
        near=0.001,
    )

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


# ★修正：古い cam_data を cam_fixed_cfg に変更
    cam_x, cam_y, cam_z = cam_fixed_cfg["x"], cam_fixed_cfg["y"], cam_fixed_cfg["z"]

    # --- 固定カメラの視野（FOV）ポリゴン計算 ---
    pitch = np.deg2rad(cam_fixed_cfg["pitch_deg"])
    cam_dir   = np.array([np.cos(pitch), 0, np.sin(pitch)])
    cam_right = np.array([0, -1, 0])
    cam_up    = np.array([-np.sin(pitch), 0, np.cos(pitch)])

    w = np.tan(np.deg2rad(cam_fixed_cfg["fov_h_deg"]) / 2)
    h = np.tan(np.deg2rad(cam_fixed_cfg["fov_v_deg"]) / 2)

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
    # ws_y_min, ws_y_max の制限も追加する場合はここ

    # 死角エリア
    deadzone_x_min = 0.0
    deadzone_x_max = 0.25
    deadzone_y_min = 0
    deadzone_y_max = 0.15 # 左端

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
        # ★ 追加条件: 死角（除外長方形）に入っていないこと
        if (deadzone_x_min <= x <= deadzone_x_max) and (deadzone_y_min <= y <= deadzone_y_max):
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
    # 🟥 【デバッグ用】除外エリア（死角）の可視化
    # ※1つの長方形を置くだけなので軽いです。調整が終わったらコメントアウト推奨。
    # ──────────────────────────────────────────
    dz_center_x = (deadzone_x_min + deadzone_x_max) / 2.0
    dz_center_y = (deadzone_y_min + deadzone_y_max) / 2.0
    dz_size_x = deadzone_x_max - deadzone_x_min
    dz_size_y = deadzone_y_max - deadzone_y_min

    scene.add_entity(
        morph=gs.morphs.Box(
            size=(dz_size_x, dz_size_y, 0.001), 
            pos=(dz_center_x, dz_center_y, 0.003), # マットより少しだけ浮かせる
            fixed=True,
            collision=False  # アームがぶつからないように物理判定をオフ
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.5)) # 赤色（半透明）
    )

    # ----------------------------------------------------

    # # ──────────────────────────────────────────
    # # 🟩 【デバッグ用】出現エリアの可視化（ホログラム仕様）
    # # ──────────────────────────────────────────
    # print("有効エリアのマーカーを生成中...")
    # for x in np.arange(mat_x_start, mat_x_start + mat_length + 0.03, 0.03):
    #     for y in np.arange(mat_y_start, mat_y_start + mat_width + 0.03, 0.03):
    #         if is_valid_spawn_area(x, y):
    #             scene.add_entity(
    #                 morph=gs.morphs.Box(
    #                     size=(0.008, 0.008, 0.001), 
    #                     pos=(x, y, 0.0025), 
    #                     fixed=True,
    #                     collision=False  # 物理演算から除外
    #                 ),
    #                 surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 0.7))
    #             )

    # ──────────────────────────────────────────
    # 🔴 アゴ先端位置の確認用マーカー（赤い小球）
    # ──────────────────────────────────────────
    # 衝突判定を持たない視覚的なダミーオブジェクトとして配置します
    marker = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.005,    # 半径5mmで視認しやすくする
            pos=(0, 0, -1),  # 初期位置は邪魔にならない地下へ
            fixed=True,
            collision=False  # 物理的な衝突判定を無効化
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 1.0)), # 赤色
    )


    scene.build()
    # ──────────────────────────────────────────
    # PD制御ゲインの設定（腕は柔らかく、グリッパーは強烈に）
    # ──────────────────────────────────────────
    # ロボットの関節数（DOF数）を取得。SO-101なら通常6個です。
    n_dofs = robot.n_dofs

    # ベースとなるゲイン（腕用）
    kp_array = np.ones(n_dofs) * 200.0
    kd_array = np.ones(n_dofs) * 20.0

    # ★ グリッパーの関節インデックスを取得
    try:
        all_dofs = np.arange(robot.n_dofs)
        gripper_idx = robot.get_joint("gripper").dof_idx_local
        all_dofs = np.arange(n_dofs)
        arm_idxs = np.delete(all_dofs, gripper_idx)
        
        MAX_GRASP_FORCE = 5.0
        
        # グリッパーの関節だけKPとKDを爆上げする
        kp_array[gripper_idx] = 300.0
        kd_array[gripper_idx] = 50.0
        print(f"グリッパー(idx:{gripper_idx})のゲインを強化しました。")
    except Exception as e:
        print("グリッパーの関節名が見つかりません。URDFを確認してください:", e)

    # ロボットにゲインを適用
    robot.set_dofs_kp(kp_array)
    robot.set_dofs_kv(kd_array)  # GenesisではKDをKV(Velocity Gain)と呼ぶことがあります
    print(f"✓ シーン構築完了 (DoF={n_dofs})")

    # def get_eef_pos():
    #     try:
    #         return robot.get_link("moving_jaw_so101_v1").get_pos().cpu().numpy()
    #     except Exception:
    #         return np.array([0.15, 0.0, 0.25], dtype=np.float32)

    # def update_wrist_camera():
    #     """グリッパー位置に追従してwristカメラを更新"""
    #     eef = get_eef_pos()
    #     # アームの根元方向（原点）を常に向く
    #     cam_pos  = eef + np.array([0.0, 0.0, 0.12])   # グリッパーの12cm真上
    #     lookat   = eef + np.array([0.0, 0.0, -0.08])  # グリッパー先端下向き
    #     try:
    #         cam_wrist.set_pose(pos=cam_pos, lookat=lookat)
    #     except Exception:
    #         pass

    def update_wrist_camera():
        """gripper_linkの位置と回転を取得し、YAMLのオフセットと角度を適用してカメラを更新"""
        try:
            # 1. gripper_linkの現在Pose（位置とクォータニオン）を取得
            link = robot.get_link("gripper_link")
            link_pos = link.get_pos().cpu().numpy()
            link_quat_wxyz = link.get_quat().cpu().numpy()
            
            # SciPyは(x, y, z, w)順のクォータニオンを想定するため並び替え
            link_quat_xyzw = np.array([
                link_quat_wxyz[1], link_quat_wxyz[2], 
                link_quat_wxyz[3], link_quat_wxyz[0]
            ])
            
            from scipy.spatial.transform import Rotation as R
            R_link = R.from_quat(link_quat_xyzw)
            
            # 2. カメラのワールド位置 (pos) の計算
            offset = np.array([
                cam_hand_cfg["offset_x"], 
                cam_hand_cfg["offset_y"], 
                cam_hand_cfg["offset_z"]
            ])
            cam_pos = link_pos + R_link.apply(offset)
            
            # 3. 視線(lookat)とアップベクトル(up)の計算
            # YAMLの角度からローカルの回転を定義（※MATLABでのオイラー角定義に合わせています）
            r = cam_hand_cfg["roll_deg"]
            p = cam_hand_cfg["pitch_deg"]
            y = cam_hand_cfg["yaw_deg"]
            
            # ※回転順序(XYZ, ZYX等)はURDFのカメラマウントの基準軸によって微調整が必要な場合があります
            R_cam_local = R.from_euler('xyz', [r, p, y], degrees=True)
            
            # カメラのローカル座標系における前方ベクトル(+X)と上ベクトル(+Z)
            local_forward = np.array([1.0, 0.0, 0.0])
            local_up      = np.array([0.0, 0.0, 1.0])
            
            # ワールド座標系へのベクトル変換（リンクの回転 × カメラのローカル回転）
            global_forward = R_link.apply(R_cam_local.apply(local_forward))
            global_up      = R_link.apply(R_cam_local.apply(local_up))
            
            lookat = cam_pos + global_forward
            
            # 4. カメラ姿勢の適用（upベクトルを渡すことで、画像の傾き=Rollも正確に反映させる）
            cam_wrist.set_pose(pos=cam_pos, lookat=lookat, up=global_up)
            
        except Exception as e:
            # print(f"Wrist camera update error: {e}") # デバッグ用
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
    # 変更前
    # target_qpos = robot.get_dofs_position().cpu().numpy().astype(np.float32)
    
    # 変更後：サーボ目標角度（ホームポジションで初期化）
    target_qpos = home_qpos.copy()
    
    # さらに、シミュレータ開始時のアーム位置も即座にホームポジションにテレポートさせます
    robot.set_dofs_position(target_qpos)

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
        target_qpos[:] = home_qpos.copy() #学習用の初期姿勢
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
                
        # sponge.set_pos(np.array([sp_x, sp_y, 0.0125], dtype=np.float32))
        # sponge.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # Z座標を 0.0125 から 0.02（円柱の半径）に変更し、床へのめり込みを防ぐ
        sponge.set_pos(np.array([sp_x, sp_y, 0.02], dtype=np.float32))
        
        # X軸に90度回転させるクォータニオン (w, x, y, z) = (0.7071, 0.7071, 0, 0)
        sponge.set_quat(np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32))
        
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

    low_fps_count = 0
    last_time = time.perf_counter()
    # ループの前にフラグを用意
    is_grasping = False
    relative_rot = None
    relative_pos = None  # 【追加】掴んだ瞬間の相対位置を保持する変数
    success_frame_count = 0

    try:
        while is_running:
            apply_pd_control()
            # 1. 現在の位置・速度を取得し、アーム用のPDトルクを計算
            current_pos = robot.get_dofs_position().cpu().numpy().astype(np.float32)
            current_vel = robot.get_dofs_velocity().cpu().numpy().astype(np.float32)
            full_torque = KP * (target_qpos - current_pos) - KD * current_vel

            # 2. グリッパーの力制御（減衰項付き）
            g_target = target_qpos[gripper_idx]
            g_current = current_pos[gripper_idx]
            g_vel = current_vel[gripper_idx]
            
            LOWER_LIMIT = -0.174
            UPPER_LIMIT = 1.745

            # 速度に応じたブレーキ（ダンパ）を常に計算
            damping_torque = -KD * g_vel

            if g_target - g_current > 0.01:
                if g_current >= UPPER_LIMIT:
                    current_gripper_force = damping_torque 
                else:
                    # 目標に向かう力 ＋ 速度超過を防ぐブレーキ
                    current_gripper_force = MAX_GRASP_FORCE + damping_torque
            elif g_target - g_current < -0.01:
                if g_current <= LOWER_LIMIT:
                    current_gripper_force = damping_torque
                else:
                    current_gripper_force = -MAX_GRASP_FORCE + damping_torque
            else:
                # 目標付近ではダンパのみで振動を抑え込んで静止させる
                current_gripper_force = damping_torque

            # 3. アーム（位置追従トルク）とグリッパー（定トルク）を個別に適用
            robot.control_dofs_force(full_torque[arm_idxs], arm_idxs)
            robot.control_dofs_force(
                np.array([current_gripper_force], dtype=np.float32), 
                np.array([gripper_idx], dtype=np.int32)
            )
            scene.step()

            # ──────────────────────────────────────────
            # 【プラン3 最終版】ステート管理型 仮想拘束
            # ──────────────────────────────────────────
            gripper_link = robot.get_link("gripper_link")
            g_pos = gripper_link.get_pos().cpu().numpy()
            g_quat = gripper_link.get_quat().cpu().numpy()
            s_pos = sponge.get_pos().cpu().numpy()

            # 1. 手先の「向き」を考慮して、実際のアゴの先端座標を計算
            # Genesisのquat(w,x,y,z)をscipy用(x,y,z,w)に変換
            rot = R.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])
            
            # グリッパー基準座標からアゴ先端へのローカルオフセット（仮）
            # ※ URDFや実際の見た目に合わせて調整してください（例: Z方向に-0.05など）
            jaw_offset_local = np.array([0.01, 0.0, -0.09], dtype=np.float32)
            
            # 手先の回転を適用してワールド座標でのアゴ先端位置を算出
            jaw_pos_world = g_pos + rot.apply(jaw_offset_local)

            # ──────────────────────────────────────────
            # 🎯 タスク成功判定（紙コップへの配置）
            # ──────────────────────────────────────────
            c_pos = cup.get_pos().cpu().numpy()
            c_quat = cup.get_quat().cpu().numpy()
            rot_c = R.from_quat([c_quat[1], c_quat[2], c_quat[3], c_quat[0]])
            
            diff_pos_cup = s_pos - c_pos
            s_local_to_cup = rot_c.inv().apply(diff_pos_cup)
            
            # X, Yの判定も少し余裕を持たせ、Z（高さ）は大幅に広げる
            in_x = abs(s_local_to_cup[0]) < 0.035
            in_y = abs(s_local_to_cup[1]) < 0.035
            # -1.5cm (底へのめり込み) 〜 6.0cm (コップから半分はみ出た状態) まで許容
            in_z = -0.015 < s_local_to_cup[2] < 0.060 
            
            # 空間内にあり、かつアームが掴んでいないか
            is_in_cup = in_x and in_y and in_z and not is_grasping
            
            if is_in_cup:
                success_frame_count += 1
            else:
                success_frame_count = 0  # 外に出るか掴み直したら0にリセット
            
            # デバッグ表示（Zの高さと、キープしているフレーム数を監視）
            dist_to_cup = np.linalg.norm(diff_pos_cup)
            if dist_to_cup < 0.10:
                print(f"Z:{s_local_to_cup[2]:.3f} | InZ:{in_z} | Count:{success_frame_count}/50    ", end="\r")
            
            # 50ステップ（約0.5秒）連続で条件をキープできたら成功
            if success_frame_count == 50:
                print(f"\n🎯 タスク完了: スポンジが紙コップに配置され、0.5秒間静止しました！")
                
                # 強化学習の場合はここで done = True を返します
                # ※キーボード操作用で何度も判定させたくない場合は、適当なマイナス値を入れてクールダウンさせます
                # success_frame_count = -100

            # ──────────────────────────────────────────
            # 🔴 マーカーを現在の計算位置にリアルタイム同期
            # ──────────────────────────────────────────
            marker.set_pos(jaw_pos_world)

            # 2. 条件判定用の変数を計算（距離は「アゴの先端」と「スポンジ」で測る）
            dist = np.linalg.norm(jaw_pos_world - s_pos)
            is_close = dist < 0.01
            
            g_angle = current_pos[gripper_idx]
            PENETRATION_THRESHOLD = 0.8  # めり込み判定の閾値（要調整）
            is_squeezing = g_angle < PENETRATION_THRESHOLD

            # 【数値確認用】アゴを閉じようとしている時だけ、リアルタイムの数値を表示
            if target_qpos[gripper_idx] > 0.01:
                print(f"Dist: {dist:.4f} (目標<0.01) | Angle: {g_angle:.4f} (目標<{PENETRATION_THRESHOLD})", end="\r")
            # 【追加】スポンジの現在の姿勢も取得しておく
            s_quat = sponge.get_quat().cpu().numpy()

            # 3. 状態遷移ロジック（ステートマシン）
            if not is_grasping:
                # 【掴んでいない状態】
                if target_qpos[gripper_idx] <= 0.20 and is_close and is_squeezing:
                    is_grasping = True
                    print(f"\n✅ 把持成功! (Dist: {dist:.3f}, Angle: {g_angle:.2f})")

                    # ① 姿勢の相対差分を保存（既存）
                    rot_g = R.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])
                    rot_s = R.from_quat([s_quat[1], s_quat[2], s_quat[3], s_quat[0]])
                    relative_rot = rot_g.inv() * rot_s
                    
                    # ② 【追加】位置の相対差分（ローカル座標系でのズレ）を保存
                    # 現在の手先(g_pos)とスポンジ(s_pos)のワールド座標の差分を計算
                    diff_pos = s_pos - g_pos
                    # その差分を、グリッパーの回転の逆行列を使って「グリッパーから見たローカル座標」に変換
                    relative_pos = rot_g.inv().apply(diff_pos)

            else:
                # 【掴んでいる状態】
                if target_qpos[gripper_idx] > 0.35:
                    is_grasping = False
                    relative_rot = None
                    relative_pos = None  # 【追加】離したらリセット
                    print(f"\n手を開きました (RELEASE)")

            # 4. 把持状態に基づく物理適用
            if is_grasping:
                # 現在の手先の姿勢（回転）を取得
                rot_g_current = R.from_quat([g_quat[1], g_quat[2], g_quat[3], g_quat[0]])
                
                # 【変更】位置の更新：現在の手先位置(g_pos)に、現在の姿勢で回転させた相対位置を足す
                s_pos_new = g_pos + rot_g_current.apply(relative_pos)
                sponge.set_pos(s_pos_new)
                
                # 姿勢の更新（既存）
                rot_s_new = rot_g_current * relative_rot
                q_new = rot_s_new.as_quat()
                s_quat_genesis = np.array([q_new[3], q_new[0], q_new[1], q_new[2]], dtype=np.float32)
                sponge.set_quat(s_quat_genesis)



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

            # ──────────────────────────────────────────
            # FPS低下による強制終了ロジック（whileループの末尾）
            # ──────────────────────────────────────────
            current_time = time.perf_counter()
            dt_real = current_time - last_time
            last_time = current_time

            if dt_real > 0:
                current_fps = 1.0 / dt_real
                
                if current_fps < 35.0:
                    low_fps_count += 1
                else:
                    low_fps_count = 0
                    
                if low_fps_count >= 30:
                    print("⚠️ FPSが30フレーム連続で35を下回ったため、シミュレーションを強制終了します。")
                    break  # whileループを抜ける

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

    # env = SO101ViewerEnv()

    # [変更] 古いSO101ViewerEnvは捨て、最新の本番環境をビューワーONで呼び出す
    env = SO101GraspEnv(show_viewer=True)
    
    # [変更] ダミー環境を使わずに直接モデルをロードする
    model = PPO.load(model_path, device="cpu")

    for ep in range(n_episodes):
        # [変更] Gymnasiumの仕様に合わせて obs, info で受け取る
        obs, _   = env.reset(seed=ep)
        # obs      = env.reset(seed=ep)
        done     = False
        total_r  = 0.0
        step     = 0
        MAX_STEP = 500

        print(f"Episode {ep + 1}/{n_episodes} 開始...")

        while not done and step < MAX_STEP:
            action, _ = model.predict(obs, deterministic=True)
            # obs, success = env.step(action)
            # 【修正箇所】戻り値を2つではなく、5つの変数で受け取る
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # if success:
            #     print(f"  ✓ SUCCESS! ({step}ステップ)")
            #     done = True

            # 終了判定（成功したか、最大ステップに達したか）
            done = terminated or truncated

            # info辞書から成功フラグを取り出す
            if info.get("is_success", False):
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

    SAVE_DIR = os.path.join(os.path.dirname(__file__), "genesis_so101_rl38")

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

        # if args.mode == "demo":
        #     if model_path and os.path.exists(model_path):
        #         print(f"モデルをロード: {model_path}")
        #         dummy_env = VecMonitor(DummyVecEnv([lambda: SO101GraspEnv()]))
        #         model = PPO.load(model_path, env=dummy_env, device="cpu")
        #         dummy_env.close()
        #     else:
        #         print("⚠️ モデルが見つかりません。先に --mode train を実行してください。")
        #         raise SystemExit

        if not model_path or not os.path.exists(model_path):
            print("⚠️ モデルが見つかりません。先に --mode train を実行してください。")
            raise SystemExit

        # run_viewer_demo(model, n_episodes=args.episodes)
        # [変更] ダミー環境をロードする処理を削除し、パスだけを渡す
        run_viewer_demo(model_path, n_episodes=args.episodes)

    print("\n✅ 完了!")
    if args.mode in ("train", "traindemo"):
        print(f"\n📊 TensorBoard で学習曲線を確認:")
        print(f"  tensorboard --logdir {SAVE_DIR}/logs")