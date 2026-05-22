% SO-101 模倣学習環境 3Dシミュレータ (完全統合版・エラー修正済)
clear; clc; close all;

%% ==========================================================
%% 1. 全体設定（JSONからのパラメータ読み込みエリア）
%% ==========================================================
urdfPath = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf';
target_dataset_id = 'dataset_02_multimodal'; 

jsonText = fileread('env_config.json');
config = jsondecode(jsonText);
dataset_info = config.datasets.(target_dataset_id);
fprintf('🚀 データセットを読み込みました: %s (カメラ数: %d)\n', dataset_info.name, dataset_info.num_cameras);

env = config.environment;
desk_x_min = env.desk_x_min;  desk_x_max = env.desk_x_max;
desk_y_min = env.desk_y_min;  desk_y_max = env.desk_y_max;
desk_z_offset = env.desk_z_offset;
mat_x_start = env.mat_x_start; mat_y_start = env.mat_y_start;
mat_length = env.mat_length;   mat_width = env.mat_width;
mat_z_offset = env.mat_z_offset;

main_cam_id = dataset_info.camera_ids{1};
cam_info = config.cameras.(main_cam_id);
cam_x = cam_info.x; cam_y = cam_info.y; cam_z = cam_info.z;
cam_pitch_deg = cam_info.pitch_deg;
cam_fov_h_deg = cam_info.fov_h_deg; cam_fov_v_deg = cam_info.fov_v_deg;

hand_cam_id = dataset_info.camera_ids{2};
cam_hand_info = config.cameras.(hand_cam_id);

show_cam_view = true; 
plot_xlim = [-0.1, 0.8]; plot_ylim = [-0.8, 0.4]; plot_zlim = [-0.05, 0.6];
plot_view = [45, 30];

%% ==========================================================
%% 2 & 3. ロボット・机・マットの描画
%% ==========================================================
fig_main = figure('Name', 'SO-101 Task Environment', 'Color', 'w', 'Position', [100, 100, 900, 700]);
robot = importrobot(urdfPath);
q_home = homeConfiguration(robot);
show(robot, q_home, 'PreservePlot', false, 'FastUpdate', true);
hold on;

X_desk = [desk_x_min, desk_x_max, desk_x_max, desk_x_min];
Y_desk = [desk_y_min, desk_y_min, desk_y_max, desk_y_max];
Z_desk = repmat(desk_z_offset, 1, 4);
patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 0.5, 'EdgeColor', [0.5 0.5 0.5]);

X_mat = [mat_x_start, mat_x_start + mat_length, mat_x_start + mat_length, mat_x_start];
Y_mat = [mat_y_start, mat_y_start, mat_y_start + mat_width, mat_y_start + mat_width];
z_target = desk_z_offset + mat_z_offset;
Z_mat = repmat(z_target, 1, 4);
patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k');

%% ==========================================================
%% 4. 俯瞰カメラの描画
%% ==========================================================
pitch = deg2rad(cam_pitch_deg); 
cam_dir = [cos(pitch), 0, sin(pitch)]; cam_right = [0, -1, 0]; cam_up = [-sin(pitch), 0, cos(pitch)];
w = tan(deg2rad(cam_fov_h_deg) / 2); h = tan(deg2rad(cam_fov_v_deg) / 2);
rays = [cam_dir + w*cam_right + h*cam_up; cam_dir - w*cam_right + h*cam_up; ...
        cam_dir - w*cam_right - h*cam_up; cam_dir + w*cam_right - h*cam_up];

t = (z_target - cam_z) ./ rays(:, 3); 
intersect_pts_fixed = repmat([cam_x, cam_y, cam_z], 4, 1) + t .* rays; 
poly_fixed = polyshape(intersect_pts_fixed(:,1), intersect_pts_fixed(:,2));

plot3([intersect_pts_fixed(:,1); intersect_pts_fixed(1,1)], [intersect_pts_fixed(:,2); intersect_pts_fixed(1,2)], repmat(z_target+0.001, 5, 1), 'g-', 'LineWidth', 2);

%% ==========================================================
%% 5. 手先カメラの数学的準備 ＆ 更新用ポリゴンの用意
%% ==========================================================
y_rad = deg2rad(cam_hand_info.yaw_deg); p_rad = deg2rad(cam_hand_info.pitch_deg); r_rad = deg2rad(cam_hand_info.roll_deg);
R_json = eul2rotm([y_rad, p_rad, r_rad], 'ZYX');
R_opt = [1 0 0; 0 0 -1; 0 1 0] * [0 0 -1; 0 1 0; 1 0 0]; 
R_cam_local = R_json * R_opt;
offset_local = [cam_hand_info.offset_x; cam_hand_info.offset_y; cam_hand_info.offset_z];

w_h = tan(deg2rad(cam_hand_info.fov_h_deg)/2); h_h = tan(deg2rad(cam_hand_info.fov_v_deg)/2);
local_rays = [1, -w_h, h_h; 1, w_h, h_h; 1, w_h, -h_h; 1, -w_h, -h_h]';

% リアルタイム更新用のオブジェクトを配置
poly_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [0, 1.0, 0.5], 'FaceAlpha', 0.85, 'EdgeColor', [0, 1.0, 0.5], 'LineWidth', 2);
hand_frustum_line = plot3(NaN, NaN, NaN, 'Color', [1 0.5 0], 'LineWidth', 2);

axis equal; grid on; view(plot_view); xlim(plot_xlim); ylim(plot_ylim); zlim(plot_zlim); hold off;

%% ==========================================================
%% 6. カメラ視点のシミュレーション（別ウィンドウ）
%% ==========================================================
if show_cam_view
    f_cam = figure('Name', 'Simulated Camera View', 'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'off');
    out_pos = get(f_cam, 'OuterPosition'); in_pos = get(f_cam, 'InnerPosition'); 
    set(f_cam, 'OuterPosition', [1050, 100, 640 + (out_pos(3)-in_pos(3)), 480 + (out_pos(4)-in_pos(4))]);
    
    % 💡 【超重要修正】MATLABの記憶混同を防ぐため、サブ画面用には「分身」を新しく読み込む！
    robot_cam = importrobot(urdfPath);
    show(robot_cam, q_home, 'PreservePlot', false); hold on;
    
    patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    ax = gca; axis equal; axis vis3d; set(ax, 'Units', 'pixels', 'Position', [0 0 640 480]); axis off; 
    camproj('perspective'); campos([cam_x, cam_y, cam_z]); camtarget([cam_x+cam_dir(1), cam_y+cam_dir(2), cam_z+cam_dir(3)]); camup(cam_up); camva(cam_fov_v_deg); hold off;
end

%% ==========================================================
%% 7. 実行後のインタラクティブ操作（手動スライダーUI）
%% ==========================================================
figure(fig_main); % メインウィンドウをターゲットにする
pnl = uipanel(fig_main, 'Position', [0, 0, 1, 0.15], 'Title', 'Joint Controls (順運動学)', 'BackgroundColor', 'w');
sliders = gobjects(6, 1);
for i = 1:6
    sliders(i) = uicontrol(pnl, 'Style', 'slider', 'Min', -pi, 'Max', pi, 'Value', q_home(i).JointPosition, ...
        'Units', 'normalized', 'Position', [0.02 + (i-1)*0.16, 0.2, 0.12, 0.4]);
    uicontrol(pnl, 'Style', 'text', 'String', sprintf('J%d', i), 'BackgroundColor', 'w', ...
        'Units', 'normalized', 'Position', [0.02 + (i-1)*0.16, 0.6, 0.12, 0.3]);
end

disp('🎮 インタラクティブモード開始！メインウィンドウのスライダーを動かしてください。');

q_current = q_home;
while ishandle(fig_main)
    for i = 1:6
        q_current(i).JointPosition = get(sliders(i), 'Value');
    end
    
    % メイン画面のロボットだけを高速更新（オブジェクト全消去バグを回避！）
    show(robot, q_current, 'Parent', fig_main.CurrentAxes, 'PreservePlot', false, 'FastUpdate', true);
    
    T_link = getTransform(robot, q_current, cam_hand_info.attached_link);
    R_link = T_link(1:3, 1:3); P_link = T_link(1:3, 4);
    P_global = P_link + R_link * offset_local;
    R_global = R_link * R_cam_local;
    global_rays = R_global * local_rays;
    
    hand_pts = zeros(4, 3);
    is_looking_down = true;
    for i = 1:4
        if global_rays(3, i) < -1e-4
            t_hand = (z_target - P_global(3)) / global_rays(3, i);
            hand_pts(i, :) = (P_global + t_hand * global_rays(:, i))';
        else
            is_looking_down = false;
        end
    end
    
    if is_looking_down
        hand_pts_closed = [hand_pts; hand_pts(1,:)];
        set(hand_frustum_line, 'XData', hand_pts_closed(:,1), 'YData', hand_pts_closed(:,2), 'ZData', repmat(z_target+0.002, 5, 1));
        
        poly_hand = polyshape(hand_pts(:,1), hand_pts(:,2));
        poly_intersect = intersect(poly_fixed, poly_hand);
        
        if poly_intersect.NumRegions > 0
            v = poly_intersect.Vertices;
            v(isnan(v(:,1)), :) = []; 
            set(poly_patch, 'XData', v(:,1), 'YData', v(:,2), 'ZData', repmat(z_target+0.003, size(v,1), 1));
        else
            set(poly_patch, 'XData', [], 'YData', [], 'ZData', []); 
        end
    else
        set(hand_frustum_line, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
        set(poly_patch, 'XData', [], 'YData', [], 'ZData', []);
    end
    
    drawnow;
end