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
fig_main = figure('Name', 'SO-101 Task Environment', 'Color', 'w', 'Position', [50, 150, 750, 700]);
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

% 💡【修正】机の表面と完全に重なって消えるのを防ぐため、1ミリ(+0.001)だけ浮かせる！
Z_mat = repmat(z_target + 0.004, 1, 4);

patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k');

%% ==========================================================
%% 4. 俯瞰カメラの描画
%% ==========================================================
% 💡 [NEW] カメラの位置から机の高さ(z_target)まで、垂直な「赤い点線」を下ろす！
plot3([cam_x, cam_x], [cam_y, cam_y], [z_target, cam_z], 'r--', 'LineWidth', 1.5);

% 💡 [変更] カメラ本体を「四角(rs)」から「丸(ro)」に変更！
plot3(cam_x, cam_y, cam_z, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
text(cam_x, cam_y, cam_z + 0.04, '📷 固定カメラ', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 11, 'HorizontalAlignment', 'center');
pitch = deg2rad(cam_pitch_deg); 
cam_dir = [cos(pitch), 0, sin(pitch)]; cam_right = [0, -1, 0]; cam_up = [-sin(pitch), 0, cos(pitch)];
w = tan(deg2rad(cam_fov_h_deg) / 2); h = tan(deg2rad(cam_fov_v_deg) / 2);
rays = [cam_dir + w*cam_right + h*cam_up; cam_dir - w*cam_right + h*cam_up; ...
        cam_dir - w*cam_right - h*cam_up; cam_dir + w*cam_right - h*cam_up];

t = (z_target - cam_z) ./ rays(:, 3); 
intersect_pts_fixed = repmat([cam_x, cam_y, cam_z], 4, 1) + t .* rays; 
poly_fixed = polyshape(intersect_pts_fixed(:,1), intersect_pts_fixed(:,2));

plot3([intersect_pts_fixed(:,1); intersect_pts_fixed(1,1)], [intersect_pts_fixed(:,2); intersect_pts_fixed(1,2)], repmat(z_target+0.001, 5, 1), 'g-', 'Color','#4646C9' ,'LineWidth', 2);
patch('XData', intersect_pts_fixed(:,1), 'YData', intersect_pts_fixed(:,2), 'ZData', repmat(z_target+0.008, 4, 1), 'FaceColor', '#4646C9', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

% 💡 【復活】私が消してしまった水色のピラミッドを再追加！
faces = [1 2 3; 1 3 4; 1 4 5; 1 5 2];
verts_fixed = [[cam_x, cam_y, cam_z]; intersect_pts_fixed];
patch('Faces', faces, 'Vertices', verts_fixed, 'FaceColor', [0 0.8 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

%% ==========================================================
%% 5. 手先カメラの数学的準備 ＆ 更新用ポリゴンの用意
%% ==========================================================
y_rad = -deg2rad(cam_hand_info.yaw_deg); p_rad = deg2rad(cam_hand_info.pitch_deg); r_rad = deg2rad(cam_hand_info.roll_deg);
R_json = eul2rotm([y_rad, p_rad, r_rad], 'YZX');
R_tweak = eul2rotm([pi, 0, 0], 'ZYX');      % パターンB: 左右180度反転（←今コレが一番怪しいです！）
R_base = [1 0 0; 0 0 -1; 0 1 0] * [0 0 -1; 0 1 0; 1 0 0];

R_opt = R_base * R_tweak;R_cam_local = R_json * R_opt;
offset_local = [cam_hand_info.offset_x; cam_hand_info.offset_y; cam_hand_info.offset_z];
w_h = tan(deg2rad(cam_hand_info.fov_h_deg)/2); h_h = tan(deg2rad(cam_hand_info.fov_v_deg)/2);
local_rays = [1, -w_h, h_h; 1, w_h, h_h; 1, w_h, -h_h; 1, -w_h, -h_h]';

% リアルタイム更新用のオブジェクトを配置
poly_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [0, 1.0, 0.5], 'FaceAlpha', 0.85, 'EdgeColor', [0, 1.0, 0.5], 'LineWidth', 2);
hand_frustum_line = plot3(NaN, NaN, NaN, 'Color', [1 0.5 0], 'LineWidth', 2);

% 💡 [NEW] 手先カメラ本体の目印（オレンジの丸）を配置！
hand_cam_marker = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% 手先カメラから机に垂直に落ちる点線
hand_drop_line = plot3(NaN, NaN, NaN, '--', 'Color', [1 0.5 0], 'LineWidth', 1.5);

% 手先カメラの机の上の「投影面（かぶっている部分）」を薄いオレンジで塗りつぶすパッチ
hand_footprint_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

% 💡 [変更] ピラミッドの輪郭線（EdgeColor）もオレンジにして、空中で見やすくする！
hand_frustum_patch = patch('Faces', [1 2 3; 1 3 4; 1 4 5; 1 5 2], ...
                           'Vertices', zeros(5,3), ...
                           'FaceColor', [1 0.5 0], 'FaceAlpha', 0.25, 'EdgeColor', [1 0.5 0], 'LineWidth', 1.5);

axis equal; grid on; view(plot_view); xlim(plot_xlim); ylim(plot_ylim); zlim(plot_zlim); hold off;

% 三次元回転を有効にする
rotate3d(fig_main.CurrentAxes, 'on');

%% ==========================================================
%% 6. カメラ視点のシミュレーション（別ウィンドウ）
%% ==========================================================
% 変数を初期化（エラー防止）
f_cam = []; ax_cam = []; robot_cam = [];

if show_cam_view
    f_cam = figure('Name', 'Simulated Camera View', 'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'off');
    out_pos = get(f_cam, 'OuterPosition'); in_pos = get(f_cam, 'InnerPosition'); 
    
    stolen_W = out_pos(3) - in_pos(3);
    stolen_H = out_pos(4) - in_pos(4);
    set(f_cam, 'OuterPosition', [870, 150, 640 + stolen_W, 480 + stolen_H]);
    
    robot_cam = importrobot(urdfPath);
    show(robot_cam, q_home, 'PreservePlot', false, 'FastUpdate', true); hold on;    
    
    patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    
    ax_cam = gca; 
    axis equal; axis vis3d; set(ax_cam, 'Units', 'pixels', 'Position', [0 0 640 480]); axis off; 
    
    camproj('perspective'); 
    campos([cam_x, cam_y, cam_z]); 
    camtarget([cam_x+cam_dir(1), cam_y+cam_dir(2), cam_z+cam_dir(3)]); 
    camup(cam_up); 
    camva(cam_fov_v_deg); 
    
    % ==========================================================
    % 💡 [NEW] MATLABのお節介オートフォーカスを完全に無効化（ロック）する！
    % ==========================================================
    set(ax_cam, 'CameraPositionMode', 'manual', ...
                'CameraTargetMode', 'manual', ...
                'CameraUpVectorMode', 'manual', ...
                'CameraViewAngleMode', 'manual', ...
                'XLimMode', 'manual', ...
                'YLimMode', 'manual', ...
                'ZLimMode', 'manual');
                
    hold off;
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
    
% メイン画面のロボットを更新
    show(robot, q_current, 'Parent', fig_main.CurrentAxes, 'PreservePlot', false, 'FastUpdate', true);
    
    % 💡 [追加] カメラ画面が開いていれば、そっちのアームも同期して動かす！
    if show_cam_view && ishandle(f_cam)
        show(robot_cam, q_current, 'Parent', ax_cam, 'PreservePlot', false, 'FastUpdate', true);
        
        % 🚨【最強の解決策】show関数が破壊したカメラ視点を、毎フレーム「上書き」してねじ伏せる！
        set(ax_cam, 'CameraPosition', [cam_x, cam_y, cam_z], ...
                    'CameraTarget', [cam_x+cam_dir(1), cam_y+cam_dir(2), cam_z+cam_dir(3)], ...
                    'CameraUpVector', cam_up, ...
                    'CameraViewAngle', cam_fov_v_deg);
    end
    
    % 3. 手先カメラの順運動学
    T_link = getTransform(robot, q_current, cam_hand_info.attached_link);
    R_link = T_link(1:3, 1:3); P_link = T_link(1:3, 4);
    P_global = P_link + R_link * offset_local;
    R_global = R_link * R_cam_local;
    global_rays = R_global * local_rays;
    
    % 💡 [NEW] 手先カメラの丸い目印を現在位置（P_global）に移動！
    set(hand_cam_marker, 'XData', P_global(1), 'YData', P_global(2), 'ZData', P_global(3));
    
    set(hand_drop_line, 'XData', [P_global(1), P_global(1)], 'YData', [P_global(2), P_global(2)], 'ZData', [P_global(3), z_target]);

    hand_pts = zeros(4, 3);
    
    MAX_DIST = 2.0; % 視野を「長さ2mのピラミッド立体」とする
    F = zeros(4, 3);
    for i = 1:4
        F(i, :) = (P_global + MAX_DIST * global_rays(:, i))';
    end
    
    % 💡 空中のピラミッドは常時更新
    set(hand_frustum_patch, 'Vertices', [P_global'; F]);
    
    % ピラミッドを構成する8本の辺（側面4本、底面4本）
    edges = [
        P_global', F(1,:);  P_global', F(2,:);
        P_global', F(3,:);  P_global', F(4,:);
        F(1,:), F(2,:);     F(2,:), F(3,:);
        F(3,:), F(4,:);     F(4,:), F(1,:)
    ];
    
    footprint_pts = [];
    % 8本の辺それぞれについて、机(z_target)を貫通しているかチェック
    for i = 1:8
        p1 = edges(i, 1:3); p2 = edges(i, 4:6);
        if (p1(3) - z_target) * (p2(3) - z_target) <= 1e-6 
            if abs(p2(3) - p1(3)) > 1e-6 % ゼロ除算回避
                t_cross = (z_target - p1(3)) / (p2(3) - p1(3));
                % 辺の線分上で交差している場合のみ点を追加
                if t_cross >= -0.01 && t_cross <= 1.01
                    cross_pt = p1 + t_cross * (p2 - p1);
                    footprint_pts = [footprint_pts; cross_pt(1:2)];
                end
            end
        end
    end
    
    % 机にぶつかった点が3つ以上あれば、図形を描画！
    if size(footprint_pts, 1) >= 3
        % 輪ゴムアルゴリズムで点群を包み込む
        k = convhull(footprint_pts(:,1), footprint_pts(:,2));
        hull_pts = footprint_pts(k, :);
        poly_hull = polyshape(hull_pts(:,1), hull_pts(:,2));
        
        % 💡 [NEW] 2m先まで伸びた巨大ポリゴンを「机のサイズ」で型抜きしてカット！（Zファイティング防止）
        poly_desk = polyshape(X_desk, Y_desk);
        poly_hand = intersect(poly_hull, poly_desk);
        
        if poly_hand.NumRegions > 0
            v_hand = poly_hand.Vertices;
            v_hand(isnan(v_hand(:,1)), :) = []; % 切れ目(NaN)を除去
            
            % 💡 [変更] 手先カメラのオレンジ枠と塗りは「12ミリ(0.012)」浮かせる！
            set(hand_frustum_line, 'XData', [v_hand(:,1); v_hand(1,1)], 'YData', [v_hand(:,2); v_hand(1,2)], 'ZData', repmat(z_target+0.012, size(v_hand,1)+1, 1));
            set(hand_footprint_patch, 'XData', v_hand(:,1), 'YData', v_hand(:,2), 'ZData', repmat(z_target+0.012, size(v_hand,1), 1));
            poly_intersect = intersect(poly_fixed, poly_hand);
            
            if poly_intersect.NumRegions > 0
                v_int = poly_intersect.Vertices;
                v_int(isnan(v_int(:,1)), :) = [];
                % 💡 [変更] 重なっている緑の部分は最上階の「16ミリ(0.016)」にする！
                set(poly_patch, 'XData', v_int(:,1), 'YData', v_int(:,2), 'ZData', repmat(z_target+0.016, size(v_int,1), 1)); 
            else
                set(poly_patch, 'XData', [], 'YData', [], 'ZData', []); 
            end
        else
            set(hand_frustum_line, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
            set(hand_footprint_patch, 'XData', [], 'YData', [], 'ZData', []);
            set(poly_patch, 'XData', [], 'YData', [], 'ZData', []);
        end
    else
        % 机と交差していない場合は枠線などを消す
        set(hand_frustum_line, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
        set(hand_footprint_patch, 'XData', [], 'YData', [], 'ZData', []);
        set(poly_patch, 'XData', [], 'YData', [], 'ZData', []);
    end
    
    drawnow;
end