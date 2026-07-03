function so101_3d_simulator()
% SO-101 模倣学習環境 3Dシミュレータ (完全統合・重力物理エンジン搭載版)ver3

% 💡 [修正] サブフォルダへの検索パス追加は、関数定義の内側で実行します
addpath(fullfile(pwd, 'config'));
addpath(fullfile(pwd, 'utils'));

clc; close all;

%% ==========================================================
%% 0. ユーザー設定パラメータ (Control Parameters)
%% ==========================================================
ctrl_param = struct();

% --- ワークスペース限界値 (Workspace Limits) [m] ---
ctrl_param.ws_x_min = 0.02;  % 奥行き(X) 下限
ctrl_param.ws_x_max = 0.45;  % 奥行き(X) 上限
ctrl_param.ws_z_min = 0.06;  % 高さ(Z) 下限
ctrl_param.ws_z_max = 0.30;  % 高さ(Z) 上限

% --- 初期目標座標 (Initial Target Position) [m] ---
ctrl_param.init_x = 0.15;
ctrl_param.init_y = 0.00;
ctrl_param.init_z = 0.06;

% --- 制御器の応答パラメータ (Controller Responses) ---
ctrl_param.key_step = 0.02;          
ctrl_param.xyz_speed_limit = 0.003;  
ctrl_param.gripper_speed = 0.04;     
ctrl_param.gripper_open_ang = 0.60;  
ctrl_param.gripper_close_ang = 0.00; 
ctrl_param.auto_z_grab = 0.06; 
ctrl_param.auto_z_lift = 0.16; 

% ==========================================================
% 🧽 スポンジ（対象物体）と物理エンジンのパラメータ
% ==========================================================
ctrl_param.sponge_size = [0.03, 0.03, 0.04];    
ctrl_param.sponge_color = [0.2, 0.8, 0.2];      
ctrl_param.sponge_init_pos = [0.25, 0.0, 0.00]; 

ctrl_param.gravity = 9.81; % 重力加速度 [m/s^2]
ctrl_param.dt = 0.02;      % サンプリング周期 [s]

%% 1. 全体設定 ＆ 設定ファイルの読み込み
urdfPath = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf';
target_dataset_id = 'dataset_02_multimodal'; 

[env, cam_info, cam_hand_info] = load_environment_config('env_config.json', target_dataset_id);

plot_xlim = [-0.1, 0.8]; plot_ylim = [-0.8, 0.4]; plot_zlim = [-0.05, 0.6];
plot_view = [180, 0];
show_cam_view = true;

%% 2 ~ 4. メイン環境の構築
[robot, q_home, fig_main, poly_fixed, cam_dir, cam_up] = create_main_environment(...
    urdfPath, env, cam_info, plot_xlim, plot_ylim, plot_zlim, plot_view);

%% 5. 手先カメラの数学的準備
[R_cam_local, offset_local, local_rays, cam_hand_fov_v] = setup_hand_camera_geometry(cam_hand_info);

handles.poly_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [0, 1.0, 0.5], 'FaceAlpha', 0.85, 'EdgeColor', [0, 1.0, 0.5], 'LineWidth', 2);
handles.hand_frustum_line = plot3(NaN, NaN, NaN, 'Color', [1 0.5 0], 'LineWidth', 2);
handles.hand_cam_marker = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
handles.hand_drop_line = plot3(NaN, NaN, NaN, '--', 'Color', [1 0.5 0], 'LineWidth', 1.5);
handles.hand_footprint_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
handles.hand_frustum_patch = patch('Faces', [1 2 3; 1 3 4; 1 4 5; 1 5 2], 'Vertices', zeros(5,3), 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.25, 'EdgeColor', [1 0.5 0], 'LineWidth', 1.5);
hold off;

%% 6. 別ウィンドウの初期化
[f_cams, ax_cam, robot_cam, ax_cam_hand, robot_cam_hand] = init_camera_views(...
    show_cam_view, urdfPath, q_home, env, cam_info, cam_dir, cam_up, cam_hand_info);

%% ==========================================================
%% 7. 実行後のインタラクティブ操作
%% ==========================================================
figure(fig_main);

pnl = uipanel(fig_main, 'Position', [0, 0, 1, 0.15], 'BackgroundColor', 'w', 'BorderType', 'line');
uicontrol(pnl, 'Style', 'text', 'String', '🎮 2リンク・手首IKモード', 'Position', [10, 65, 200, 20], 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
uicontrol(pnl, 'Style', 'text', 'String', '[W/S]: 前後(X)  |  [Q/E]: 上下(Z)  |  [A/D]: 旋回  |  [H]: 昇降  |  [G]: 掴む', 'Position', [10, 35, 550, 25], 'FontSize', 11, 'BackgroundColor', 'w', 'HorizontalAlignment', 'left');

txt_status = uicontrol(pnl, 'Style', 'text', 'String', '初期化中...', ...
    'Position', [10, 5, 600, 25], 'FontSize', 14, 'FontWeight', 'bold', 'ForegroundColor', [0 0.4 0.8], 'BackgroundColor', 'w', 'HorizontalAlignment', 'left');

disp('🎮 W/A/S/D/Q/E キーで手首の位置を直接操作します。');

% 状態保存
data = struct();
data.param = ctrl_param; 
data.target_x = ctrl_param.init_x;       
data.target_y = ctrl_param.init_y;        
data.target_z = ctrl_param.init_z;
data.current_x = ctrl_param.init_x;      
data.current_y = ctrl_param.init_y;       
data.current_z = ctrl_param.init_z;      
data.gripper_target = 0;    
data.gripper_current = ctrl_param.gripper_close_ang; 
data.auto_z_state = 0; 
data.sponge_pos = ctrl_param.sponge_init_pos; 
data.sponge_vz = 0.0; 
data.is_grasped = false;                      

fig_main.UserData = data;

set(fig_main, 'WindowKeyPressFcn', @(src, event) handle_keyboard_input(fig_main, event));
if exist('f_cams', 'var') && ishandle(f_cams)
    set(f_cams, 'WindowKeyPressFcn', @(src, event) handle_keyboard_input(fig_main, event));
end

q_current = q_home;
z_target = env.desk_z_offset + env.mat_z_offset; 

gripper_joint_indices = [];
for i = 1:numel(q_current)
    joint_name = q_current(i).JointName;
    if contains(lower(joint_name), 'finger') || contains(lower(joint_name), 'gripper')
        gripper_joint_indices = [gripper_joint_indices, i];
    end
end
if isempty(gripper_joint_indices) && numel(q_current) > 6
    gripper_joint_indices = 7:numel(q_current);
end

hold(fig_main.CurrentAxes, 'on');
h_target_marker = plot3(fig_main.CurrentAxes, NaN, NaN, NaN, 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');
h_actual_marker = plot3(fig_main.CurrentAxes, NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'k');

h_line_x = plot3(fig_main.CurrentAxes, NaN, NaN, NaN, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
h_line_z = plot3(fig_main.CurrentAxes, NaN, NaN, NaN, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

% 🧽 スポンジのパッチを全画面に生成
[v_sponge, f_sponge] = create_cuboid_vertices(data.sponge_pos, data.param.sponge_size, 0.0624);
h_sponge = patch(fig_main.CurrentAxes, 'Vertices', v_sponge, 'Faces', f_sponge, ...
    'FaceColor', data.param.sponge_color, 'EdgeColor', [0.1, 0.4, 0.1], 'FaceAlpha', 0.9, 'LineWidth', 1.5);

h_sponge_cam = []; h_sponge_hand = [];
if show_cam_view && exist('f_cams', 'var') && ishandle(f_cams)
    h_sponge_cam = patch(ax_cam, 'Vertices', v_sponge, 'Faces', f_sponge, 'FaceColor', data.param.sponge_color, 'EdgeColor', [0.1, 0.4, 0.1], 'FaceAlpha', 0.9);
    h_sponge_hand = patch(ax_cam_hand, 'Vertices', v_sponge, 'Faces', f_sponge, 'FaceColor', data.param.sponge_color, 'EdgeColor', [0.1, 0.4, 0.1], 'FaceAlpha', 0.9);
end

% GIK（一般化逆運動学）ソルバーの準備
endEffector = 'gripper_link'; 
gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'position', 'aiming'});
gik.SolverParameters.AllowRandomRestart = false;
gik.SolverParameters.MaxIterations = 150;

posCon = constraintPositionTarget(endEffector);
posCon.Weights = 100.0; 

aimCon = constraintAiming(endEffector);
aimCon.AngularTolerance = deg2rad(1);
aimCon.Weights = 0.001; 

% --- リアルタイムループ ---
while ishandle(fig_main)
    data = fig_main.UserData;
        
    xyz_speed_limit = data.param.xyz_speed_limit; 
    
    if data.current_x < data.target_x
        data.current_x = min(data.target_x, data.current_x + xyz_speed_limit);
    elseif data.current_x > data.target_x
        data.current_x = max(data.target_x, data.current_x - xyz_speed_limit);
    end
    
    if data.current_y < data.target_y
        data.current_y = min(data.target_y, data.current_y + xyz_speed_limit);
    elseif data.current_y > data.target_y
        data.current_y = max(data.target_y, data.current_y - xyz_speed_limit);
    end
    
    if data.current_z < data.target_z
        data.current_z = min(data.target_z, data.current_z + xyz_speed_limit);
    elseif data.current_z > data.target_z
        data.current_z = max(data.target_z, data.current_z - xyz_speed_limit);
    end
    
    fig_main.UserData.current_x = data.current_x;
    fig_main.UserData.current_y = data.current_y;
    fig_main.UserData.current_z = data.current_z;
    
    target_z_world = data.current_z + 0.0624;
    posCon.TargetPosition = [data.current_x, data.current_y, target_z_world];
    aimCon.TargetPoint = [data.current_x, data.current_y, 10.0]; 
    
    [q_sol, solInfo] = gik(q_current, posCon, aimCon);
    
    q_sol(5).JointPosition = 0;
    q_sol(6).JointPosition = 0;
    q_current = q_sol;

    if data.gripper_target == 1
        target_angle = data.param.gripper_open_ang; 
    else
        target_angle = data.param.gripper_close_ang; 
    end
    delta_limit = data.param.gripper_speed; 

    if data.gripper_current < target_angle
        data.gripper_current = min(target_angle, data.gripper_current + delta_limit);
    elseif data.gripper_current > target_angle
        data.gripper_current = max(target_angle, data.gripper_current - delta_limit);
    end
    fig_main.UserData.gripper_current = data.gripper_current;
    for idx = gripper_joint_indices
        q_current(idx).JointPosition = data.gripper_current;
    end

    actual_tform = getTransform(robot, q_current, endEffector);
    actual_x = actual_tform(1, 4);
    actual_y = actual_tform(2, 4);
    actual_z = actual_tform(3, 4) - 0.0624; 

    % 🧽 スポンジの物理挙動（等加速度運動）
    if data.is_grasped
        data.sponge_pos = [actual_x, actual_y, actual_z - 0.01];
        data.sponge_vz = 0.0;
    else
        floor_z = data.param.sponge_init_pos(3);
        if data.sponge_pos(3) > floor_z
            data.sponge_vz = data.sponge_vz - (data.param.gravity * data.param.dt);
            data.sponge_pos(3) = data.sponge_pos(3) + (data.sponge_vz * data.param.dt);
            
            if data.sponge_pos(3) <= floor_z
                data.sponge_pos(3) = floor_z;
                data.sponge_vz = 0.0;
            end
        end
    end
    fig_main.UserData.sponge_pos = data.sponge_pos; 
    fig_main.UserData.sponge_vz = data.sponge_vz;
    
    [v_sponge, ~] = create_cuboid_vertices(data.sponge_pos, data.param.sponge_size, 0.0624);
    set(h_sponge, 'Vertices', v_sponge);
    if show_cam_view && exist('f_cams', 'var') && ishandle(f_cams)
        set(h_sponge_cam, 'Vertices', v_sponge);
        set(h_sponge_hand, 'Vertices', v_sponge);
    end
    
    txt_status.String = sprintf('🎯 目標 [x:%5.3f y:%5.3f z:%5.3f] | 🤖 実際 [x:%5.3f y:%5.3f z:%5.3f]', ...
                                data.target_x, data.target_y, data.target_z, actual_x, actual_y, actual_z);

    set(h_target_marker, 'XData', data.target_x, 'YData', data.target_y, 'ZData', data.target_z + 0.0624); 
    set(h_actual_marker, 'XData', actual_tform(1,4), 'YData', actual_tform(2,4), 'ZData', actual_tform(3,4));

    set(h_line_x, 'XData', [-0.1, 0.8], 'YData', [data.target_y, data.target_y], 'ZData', [data.target_z + 0.0624, data.target_z + 0.0624]);
    set(h_line_z, 'XData', [data.target_x, data.target_x], 'YData', [data.target_y, data.target_y], 'ZData', [-0.05, 0.6]);

    show(robot, q_current, 'Parent', fig_main.CurrentAxes, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
    
    T_link = getTransform(robot, q_current, cam_hand_info.attached_link);
    R_link = T_link(1:3, 1:3); P_link = T_link(1:3, 4);
    
    P_global = P_link + R_link * offset_local;
    R_global = R_link * R_cam_local;
    global_rays = R_global * local_rays;

    if show_cam_view && exist('f_cams', 'var') && ishandle(f_cams)
        show(robot_cam, q_current, 'Parent', ax_cam, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
        show(robot_cam_hand, q_current, 'Parent', ax_cam_hand, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
        
        cam_dir_hand = R_global * [1; 0; 0];
        cam_up_hand  = R_global * [0; 0; 1]; 
        set(ax_cam_hand, 'CameraPosition', P_global', 'CameraTarget', (P_global + cam_dir_hand)', ...
                         'CameraUpVector', cam_up_hand', 'CameraViewAngle', cam_hand_fov_v);
    end
    
    set(handles.hand_cam_marker, 'XData', P_global(1), 'YData', P_global(2), 'ZData', P_global(3));
    set(handles.hand_drop_line, 'XData', [P_global(1), P_global(1)], 'YData', [P_global(2), P_global(2)], 'ZData', [P_global(3), z_target]);
    update_camera_projections(P_global, global_rays, z_target, env, poly_fixed, handles);
    
    drawnow;
    pause(data.param.dt); 
end

end 


%% 🛠️ 以下、独立したローカル関数群

function [env, cam_info, cam_hand_info] = load_environment_config(jsonFile, dataset_id)
    jsonText = fileread(jsonFile);
    config = jsondecode(jsonText);
    dataset_info = config.datasets.(dataset_id);
    fprintf('🚀 データセットを読み込みました: %s (カメラ数: %d)\n', dataset_info.name, dataset_info.num_cameras);
    
    env = config.environment;
    cam_info = config.cameras.(dataset_info.camera_ids{1});
    cam_hand_info = config.cameras.(dataset_info.camera_ids{2});
end

function [robot, q_home, fig_main, poly_fixed, cam_dir, cam_up] = create_main_environment(urdfPath, env, cam_info, xlims, ylims, zlims, view_angle)
    fig_main = figure('Name', 'SO-101 Task Environment', 'Color', 'w', 'Position', [50, 150, 750, 700]);
    robot = importrobot(urdfPath);

    robot.getBody('upper_arm_link').Joint.PositionLimits = [-1.2, 1.2]; 
    robot.getBody('lower_arm_link').Joint.PositionLimits = [-1.55, 1.69];

    q_home = homeConfiguration(robot);
    show(robot, q_home, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
    hold on;

    X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
    Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
    Z_desk = repmat(env.desk_z_offset, 1, 4);
    patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 0.5, 'EdgeColor', [0.5 0.5 0.5]);

    X_mat = [env.mat_x_start, env.mat_x_start + env.mat_length, env.mat_x_start + env.mat_length, env.mat_x_start];
    Y_mat = [env.mat_y_start, env.mat_y_start, env.mat_y_start + env.mat_width, env.mat_y_start + env.mat_width];
    z_target = env.desk_z_offset + env.mat_z_offset;
    Z_mat = repmat(z_target + 0.004, 1, 4);
    patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k');

    plot3([cam_info.x, cam_info.x], [cam_info.y, cam_info.y], [z_target, cam_info.z], 'r--', 'LineWidth', 1.5);
    plot3(cam_info.x, cam_info.y, cam_info.z, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(cam_info.x, cam_info.y, cam_info.z + 0.04, '📷 固定カメラ', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 11, 'HorizontalAlignment', 'center');
    
    pitch = deg2rad(cam_info.pitch_deg); 
    cam_dir = [cos(pitch), 0, sin(pitch)]; cam_right = [0, -1, 0]; cam_up = [-sin(pitch), 0, cos(pitch)];
    w = tan(deg2rad(cam_info.fov_h_deg) / 2); h = tan(deg2rad(cam_info.fov_v_deg) / 2);
    rays = [cam_dir + w*cam_right + h*cam_up; cam_dir - w*cam_right + h*cam_up; ...
            cam_dir - w*cam_right - h*cam_up; cam_dir + w*cam_right - h*cam_up];
    t = (z_target - cam_info.z) ./ rays(:, 3); 
    intersect_pts_fixed = repmat([cam_info.x, cam_info.y, cam_info.z], 4, 1) + t .* rays; 
    poly_fixed = polyshape(intersect_pts_fixed(:,1), intersect_pts_fixed(:,2));

    plot3([intersect_pts_fixed(:,1); intersect_pts_fixed(1,1)], [intersect_pts_fixed(:,2); intersect_pts_fixed(1,2)], repmat(z_target+0.008, 5, 1), 'g-', 'Color','#4646C9' ,'LineWidth', 2);
    patch('XData', intersect_pts_fixed(:,1), 'YData', intersect_pts_fixed(:,2), 'ZData', repmat(z_target+0.008, 4, 1), 'FaceColor', '#4646C9', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    patch('Faces', [1 2 3; 1 3 4; 1 4 5; 1 5 2], 'Vertices', [[cam_info.x, cam_info.y, cam_info.z]; intersect_pts_fixed], 'FaceColor', [0 0.8 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    axis equal; grid on; view(view_angle); xlim(xlims); ylim(ylims); zlim(zlims);
end

function [R_cam_local, offset_local, local_rays, fov_v] = setup_hand_camera_geometry(cam_hand_info)
    y_rad = -deg2rad(cam_hand_info.yaw_deg); p_rad = deg2rad(cam_hand_info.pitch_deg); r_rad = deg2rad(cam_hand_info.roll_deg);
    R_json = eul2rotm([y_rad, p_rad, r_rad], 'YZX');
    R_tweak = eul2rotm([pi, 0, 0], 'ZYX'); 
    R_base = [1 0 0; 0 0 -1; 0 1 0] * [0 0 -1; 0 1 0; 1 0 0];
    R_opt = R_base * R_tweak;
    R_cam_local = R_json * R_opt;
    offset_local = [cam_hand_info.offset_x; cam_hand_info.offset_y; cam_hand_info.offset_z];
    
    window_aspect = 640 / 480; 
    h_h = tan(deg2rad(cam_hand_info.fov_v_deg)/2);
    w_h = h_h * window_aspect; 
    
    local_rays = [1, -w_h, h_h; 1, w_h, h_h; 1, w_h, -h_h; 1, -w_h, -h_h]';
    fov_v = cam_hand_info.fov_v_deg;
end

function [f_cams, ax_cam, robot_cam, ax_cam_hand, robot_cam_hand] = init_camera_views(show_cam_view, urdfPath, q_home, env, cam_info, cam_dir, cam_up, cam_hand_info)
    f_cams = []; ax_cam = []; robot_cam = []; ax_cam_hand = []; robot_cam_hand = [];
    if ~show_cam_view, return; end
    X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
    Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
    Z_desk = repmat(env.desk_z_offset, 1, 4);
    X_mat = [env.mat_x_start, env.mat_x_start + env.mat_length, env.mat_x_start + env.mat_length, env.mat_x_start];
    Y_mat = [env.mat_y_start, env.mat_y_start, env.mat_y_start + env.mat_width, env.mat_y_start + env.mat_width];
    Z_mat = repmat(env.desk_z_offset + env.mat_z_offset, 1, 4);
    f_cams = figure('Name', 'Camera Views (Top: Fixed, Bottom: FPV)', 'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'off');
    set(f_cams, 'Position', [850, 100, 480, 720]);
    pnl_top = uipanel('Parent', f_cams, 'Position', [0, 0.5, 1.0, 0.5], 'BackgroundColor', 'w', 'BorderType', 'line', 'Title', ' 📷 固定カメラ ', 'FontSize', 11, 'FontWeight', 'bold');
    pnl_bot = uipanel('Parent', f_cams, 'Position', [0, 0.0, 1.0, 0.5], 'BackgroundColor', 'w', 'BorderType', 'line', 'Title', ' 🦾 手先カメラ (FPV) ', 'FontSize', 11, 'FontWeight', 'bold');
    
    % 固定カメラ
    ax_cam = axes('Parent', pnl_top, 'Position', [0, 0, 1, 1]); 
    robot_cam = importrobot(urdfPath);
    show(robot_cam, q_home, 'Parent', ax_cam, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off'); 
    hold(ax_cam, 'on');    
    patch('Parent', ax_cam, 'XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('Parent', ax_cam, 'XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    axis(ax_cam, 'equal'); axis(ax_cam, 'vis3d'); axis(ax_cam, 'off'); 
    axes(ax_cam); 
    camproj('perspective'); campos([cam_info.x, cam_info.y, cam_info.z]); 
    camtarget([cam_info.x+cam_dir(1), cam_info.y+cam_dir(2), cam_info.z+cam_dir(3)]); camup(cam_up); camva(cam_info.fov_v_deg); 
    set(ax_cam, 'CameraPositionMode', 'manual', 'CameraTargetMode', 'manual', 'CameraUpVectorMode', 'manual', 'CameraViewAngleMode', 'manual', 'XLimMode', 'manual', 'YLimMode', 'manual', 'ZLimMode', 'manual');

    % 手先カメラ
    ax_cam_hand = axes('Parent', pnl_bot, 'Position', [0, 0, 1, 1]); 
    robot_cam_hand = importrobot(urdfPath);
    show(robot_cam_hand, q_home, 'Parent', ax_cam_hand, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off'); 
    hold(ax_cam_hand, 'on');    
    patch('Parent', ax_cam_hand, 'XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('Parent', ax_cam_hand, 'XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    axis(ax_cam_hand, 'equal'); axis(ax_cam_hand, 'vis3d'); axis(ax_cam_hand, 'off'); 
    axes(ax_cam_hand); 
    camproj('perspective'); camva(cam_hand_info.fov_v_deg); 
    set(ax_cam_hand, 'CameraPositionMode', 'manual', 'CameraTargetMode', 'manual', 'CameraUpVectorMode', 'manual', 'CameraViewAngleMode', 'manual', 'XLimMode', 'manual', 'YLimMode', 'manual', 'ZLimMode', 'manual');
end

function update_camera_projections(P_global, global_rays, z_target, env, poly_fixed, handles)
    MAX_DIST = 2.0; 
    F = zeros(4, 3);
    for i = 1:4
        F(i, :) = (P_global + MAX_DIST * global_rays(:, i))';
    end
    
    set(handles.hand_frustum_patch, 'Vertices', [P_global'; F]);
    
    edges = [
        P_global', F(1,:);  P_global', F(2,:);
        P_global', F(3,:);  P_global', F(4,:);
        F(1,:), F(2,:);     F(2,:), F(3,:);
        F(3,:), F(4,:);     F(4,:), F(1,:)
    ];
    
    footprint_pts = [];
    for i = 1:8
        p1 = edges(i, 1:3); p2 = edges(i, 4:6);
        if (p1(3) - z_target) * (p2(3) - z_target) <= 1e-6 
            if abs(p2(3) - p1(3)) > 1e-6 
                t_cross = (z_target - p1(3)) / (p2(3) - p1(3));
                if t_cross >= -0.01 && t_cross <= 1.01
                    cross_pt = p1 + t_cross * (p2 - p1);
                    footprint_pts = [footprint_pts; cross_pt(1:2)];
                end
            end
        end
    end
    
    if size(footprint_pts, 1) >= 3
        k = convhull(footprint_pts(:,1), footprint_pts(:,2));
        hull_pts = footprint_pts(k, :);
        poly_hull = polyshape(hull_pts(:,1), hull_pts(:,2));
        
        X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
        Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
        poly_desk = polyshape(X_desk, Y_desk);
        poly_hand = intersect(poly_hull, poly_desk);
        
        if poly_hand.NumRegions > 0
            v_hand = poly_hand.Vertices;
            v_hand(isnan(v_hand(:,1)), :) = []; 
            
            set(handles.hand_frustum_line, 'XData', [v_hand(:,1); v_hand(1,1)], 'YData', [v_hand(:,2); v_hand(1,2)], 'ZData', repmat(z_target+0.012, size(v_hand,1)+1, 1));
            set(handles.hand_footprint_patch, 'XData', v_hand(:,1), 'YData', v_hand(:,2), 'ZData', repmat(z_target+0.012, size(v_hand,1), 1));
            
            poly_intersect = intersect(poly_fixed, poly_hand);
            if poly_intersect.NumRegions > 0
                v_int = poly_intersect.Vertices;
                v_int(isnan(v_int(:,1)), :) = [];
                set(handles.poly_patch, 'XData', v_int(:,1), 'YData', v_int(:,2), 'ZData', repmat(z_target+0.016, size(v_int,1), 1)); 
            else
                set(handles.poly_patch, 'XData', [], 'YData', [], 'ZData', []); 
            end
        else
            clear_projections(handles);
        end
    else
        clear_projections(handles);
    end
end

function clear_projections(handles)
    set(handles.hand_frustum_line, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(handles.hand_footprint_patch, 'XData', [], 'YData', [], 'ZData', []);
    set(handles.poly_patch, 'XData', [], 'YData', [], 'ZData', []); 
end

%% 📦 関数: handle_keyboard_input
function handle_keyboard_input(fig, event)
    data = fig.UserData;
    p = data.param; 
    
    switch lower(event.Key)
        case 'w' 
            data.target_x = data.target_x + p.key_step;
        case 's' 
            data.target_x = data.target_x - p.key_step;
        case 'a' 
            data.target_y = data.target_y + p.key_step;
        case 'd' 
            data.target_y = data.target_y - p.key_step;
        case 'q' 
            data.target_z = data.target_z + p.key_step;
        case 'e' 
            data.target_z = data.target_z - p.key_step;
        case 'h' 
            if data.auto_z_state == 0
                data.auto_z_state = 1;
                data.target_z = p.auto_z_lift;
                disp('⬆️ アーム: 持ち上げ高さに移動');
            else
                data.auto_z_state = 0;
                data.target_z = p.auto_z_grab;
                disp('⬇️ アーム: 掴む高さに移動');
            end
        case 'g' 
            if data.gripper_target == 0
                data.gripper_target = 1;
                disp('🔓 Gripper: 開放運動を開始');
                data.is_grasped = false; 
            else
                data.gripper_target = 0;
                disp('🔒 Gripper: 閉鎖運動を開始');
                
                dist_xy = sqrt((data.current_x - data.sponge_pos(1))^2 + (data.current_y - data.sponge_pos(2))^2);
                dist_z = abs((data.current_z - 0.06) - data.sponge_pos(3)); 
                
                if dist_xy < 0.03 && dist_z < 0.03
                    data.is_grasped = true;
                    disp('🌟 見事！スポンジを把持しました！');
                else
                    disp('💨 スポンジを空振りしました...');
                end
            end
    end
    
    data.target_x = max(p.ws_x_min, min(data.target_x, p.ws_x_max)); 
    data.target_z = max(p.ws_z_min, min(data.target_z, p.ws_z_max)); 
    
    fig.UserData = data; 
end

%% 📦 関数: solve_geometric_ik
function q_angles = solve_geometric_ik(target_x, target_y, target_z)
    q1 = atan2(target_y, target_x);
    target_local_x = sqrt(target_x^2 + target_y^2); 
    lift_offset_x = -0.0304; 
    lift_offset_z = -0.0542; 
    l1 = 0.1160; alpha1 = 0.2438; 
    l2 = 0.1350; alpha2 = 0.0385; 
    base_x = target_local_x - lift_offset_x;
    base_z = target_z - lift_offset_z;
    dist_sq = base_x^2 + base_z^2;
    dist = sqrt(dist_sq);
    if dist > (l1 + l2 - 0.001)
        ratio = (l1 + l2 - 0.001) / dist;
        base_x = base_x * ratio; base_z = base_z * ratio;
        dist_sq = base_x^2 + base_z^2;
        dist = sqrt(dist_sq);
    end
    phi = atan2(base_z, base_x);
    cos_theta2 = (dist_sq - l1^2 - l2^2) / (2 * l1 * l2);
    cos_theta2 = max(-1, min(1, cos_theta2)); 
    geo_theta2 = -acos(cos_theta2); 
    gamma = atan2(l2 * sin(geo_theta2), l1 + l2 * cos(geo_theta2));
    geo_theta1 = phi - gamma;
    
    q2_urdf = (pi/2) - geo_theta1 - alpha1; 
    q3_urdf = -geo_theta2 - alpha2;         
    q1      = max(-1.9198, min(1.9198, q1));
    q2_urdf = max(-1.7453, min(1.7453, q2_urdf));
    q3_urdf = max(-1.6900, min(1.6900, q3_urdf));
    q_angles = [q1, q2_urdf, q3_urdf, 0, 0, 0];
end

%% 📦 関数: create_cuboid_vertices
function [v, f] = create_cuboid_vertices(pos, size, z_offset)
    w = size(1)/2; d = size(2)/2; h = size(3);
    x = pos(1); y = pos(2); z = pos(3) + z_offset; 
    
    v = [
        x-w, y-d, z;   x+w, y-d, z;   x+w, y+d, z;   x-w, y+d, z;   
        x-w, y-d, z+h; x+w, y-d, z+h; x+w, y+d, z+h; x-w, y+d, z+h  
    ];
    
    f = [
        1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8  
    ];
end