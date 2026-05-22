% SO-101 模倣学習環境 3Dシミュレータ
clear; clc; close all;

%% ==========================================================
%% 1. 全体設定（JSONからのパラメータ読み込みエリア）
%% ==========================================================
% --- ロボット設定 ---
urdfPath = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf';

% --- データセットの指定 ---
% 💡 ここで使いたいデータセットのIDを指定します！
target_dataset_id = 'dataset_01_baseline'; 

% JSONファイルの読み込み
jsonText = fileread('env_config.json');
config = jsondecode(jsonText);

% ターゲットデータセット情報の取得
dataset_info = config.datasets.(target_dataset_id);
fprintf('🚀 データセットを読み込みました: %s (カメラ数: %d)\n', dataset_info.name, dataset_info.num_cameras);

% --- 環境設定の展開 ---
env = config.environment;
desk_x_min = env.desk_x_min;  desk_x_max = env.desk_x_max;
desk_y_min = env.desk_y_min;  desk_y_max = env.desk_y_max;
desk_z_offset = env.desk_z_offset;
mat_x_start = env.mat_x_start; mat_y_start = env.mat_y_start;
mat_length = env.mat_length;   mat_width = env.mat_width;
mat_z_offset = env.mat_z_offset;

% --- メインカメラ（1台目）の設定を展開 ---
% （※現在は配列の1つ目＝固定カメラをメインとして処理します）
main_cam_id = dataset_info.camera_ids{1};
cam_info = config.cameras.(main_cam_id);

cam_x = cam_info.x;
cam_y = cam_info.y;
cam_z = cam_info.z;
cam_pitch_deg = cam_info.pitch_deg;
cam_fov_h_deg = cam_info.fov_h_deg;
cam_fov_v_deg = cam_info.fov_v_deg;

% --- 表示オプション ---
show_cam_view = true;
plot_xlim = [-0.1, 0.8];
plot_ylim = [-0.8, 0.4];
plot_zlim = [-0.05, 0.6];
plot_view = [45, 30];


%% ==========================================================
%% 2. ロボットアームの描画
%% ==========================================================
figure('Name', 'SO-101 Task Environment', 'Color', 'w', 'Position', [100, 100, 900, 700]);

% URDFの読み込みと表示
robot = importrobot(urdfPath);
show(robot, homeConfiguration(robot), 'PreservePlot', false, 'FastUpdate', true);
hold on;


%% ==========================================================
%% 3. 作業机と黒マットの描画
%% ==========================================================

% --- 机の描画 ---
X_desk = [desk_x_min, desk_x_max, desk_x_max, desk_x_min];
Y_desk = [desk_y_min, desk_y_min, desk_y_max, desk_y_max];
Z_desk = repmat(desk_z_offset, 1, 4); % 全頂点にオフセットを適用

patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, ...
      'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 0.5, 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5);
text(desk_x_max - 0.05, desk_y_max - 0.05, desk_z_offset + 0.02, '作業机', ...
    'HorizontalAlignment', 'right', 'Color', [0.5 0.3 0.1], 'FontWeight', 'bold');

% --- 黒マットの描画 ---
X_mat = [mat_x_start, mat_x_start + mat_length, mat_x_start + mat_length, mat_x_start];
Y_mat = [mat_y_start, mat_y_start, mat_y_start + mat_width, mat_y_start + mat_width];
Z_mat = repmat(desk_z_offset + mat_z_offset, 1, 4); % 机の高さ + マットの浮き

patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, ...
      'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(mat_x_start + mat_length - 0.02, mat_y_start + 0.02, desk_z_offset + 0.02, '黒マット', ...
    'HorizontalAlignment', 'right', 'Color', 'w', 'FontWeight', 'bold');


%% ==========================================================
%% 4. 俯瞰カメラと視野角（フラスタム）の描画
%% ==========================================================

% カメラ本体の描画
plot3(cam_x, cam_y, cam_z, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(cam_x, cam_y, cam_z + 0.03, '俯瞰カメラ', 'Color', 'r', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% カメラの方向ベクトル計算
pitch = deg2rad(cam_pitch_deg); 
cam_dir   = [cos(pitch), 0, sin(pitch)];
cam_right = [0, -1, 0];
cam_up    = [-sin(pitch), 0, cos(pitch)];

% 視野角のベクトル計算
w = tan(deg2rad(cam_fov_h_deg) / 2);
h = tan(deg2rad(cam_fov_v_deg) / 2);

ray_tr = cam_dir + w * cam_right + h * cam_up;
ray_tl = cam_dir - w * cam_right + h * cam_up;
ray_bl = cam_dir - w * cam_right - h * cam_up;
ray_br = cam_dir + w * cam_right - h * cam_up;
rays = [ray_tr; ray_tl; ray_bl; ray_br];

% マット（または机）との交点計算
z_target = desk_z_offset + mat_z_offset; 
t = (z_target - cam_z) ./ rays(:, 3); 
intersect_pts = repmat([cam_x, cam_y, cam_z], 4, 1) + t .* rays; 

% 半透明の視野角描画
faces = [1 2 3; 1 3 4; 1 4 5; 1 5 2]; 
verts = [[cam_x, cam_y, cam_z]; intersect_pts];
patch('Faces', faces, 'Vertices', verts, 'FaceColor', [0 0.8 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

% 緑のテープ（視野境界）の描画
X_tape = [intersect_pts(:,1); intersect_pts(1,1)];
Y_tape = [intersect_pts(:,2); intersect_pts(1,2)];
Z_tape = repmat(z_target + 0.001, 5, 1); % マットからさらに1mm浮かせる

plot3(X_tape, Y_tape, Z_tape, 'g-', 'LineWidth', 4);
text(mean(X_tape), mean(Y_tape), z_target + 0.01, 'カメラ視野', 'Color', 'g', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');


%% ==========================================================
%% 5. グラフの体裁・表示設定
%% ==========================================================
axis equal;
grid on;
xlabel('X方向 [m] (正面)');
ylabel('Y方向 [m] (左右)');
zlabel('Z方向 [m] (高さ)');
title('SO-101 作業環境シミュレータ', 'FontSize', 15);

view(plot_view); 
xlim(plot_xlim);
ylim(plot_ylim);
zlim(plot_zlim);

hold off;
rotate3d on;

%% ==========================================================
%% 6. カメラ視点（現実のWebカメラの映像）のシミュレーション
%% ==========================================================
if show_cam_view
    
    % 1. まずウィンドウを立ち上げる（メニュー等は消す）
    f = figure('Name', 'Simulated Camera View', 'Color', 'w', ...
               'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'off');
    
    % 2. 💡【あなたの最強ロジック】OSに奪われる枠線のサイズを動的に計算！
    out_pos = get(f, 'OuterPosition'); 
    in_pos  = get(f, 'InnerPosition'); 
    stolen_width  = out_pos(3) - in_pos(3);
    stolen_height = out_pos(4) - in_pos(4);
    
    % 3. 中身が絶対に 640x480 になるように、外枠サイズに奪われた分を足して強制設定！
    set(f, 'OuterPosition', [1050, 100, 640 + stolen_width, 480 + stolen_height]);
    
    % ロボットと机・マットを描画
    show(robot, homeConfiguration(robot), 'PreservePlot', false, 'FastUpdate', true);
    hold on;
    patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    
    ax = gca;
    
    % 1:1:1の比率と縮尺ロック
    axis equal;    
    axis vis3d;    
    
    % 4. 確保した完璧なキャンバス上に、1ピクセルの狂いもなく 640x480 の映像エリアを配置
    set(ax, 'Units', 'pixels', 'Position', [0 0 640 480]); 
    axis off; 
    
    % 📸 カメラ設定
    camproj('perspective'); 
    campos([cam_x, cam_y, cam_z]); 
    camtarget([cam_x + cam_dir(1), cam_y + cam_dir(2), cam_z + cam_dir(3)]); 
    camup(cam_up); 
    
    % 視野角を適用
    camva(cam_fov_v_deg); 
    
    hold off;
end