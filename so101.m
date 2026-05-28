function so101_3d_simulator()
% SO-101 模倣学習環境 3Dシミュレータ (完全統合・関数カプセル化版)
clc; close all;

%% ==========================================================
%% 1. 全体設定 ＆ 設定ファイルの読み込み
%% ==========================================================
urdfPath = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf';
target_dataset_id = 'dataset_02_multimodal'; 

% 💡 [関数化] JSONパラメータの読み込み
[env, cam_info, cam_hand_info] = load_environment_config('env_config.json', target_dataset_id);

% 描画範囲の設定
plot_xlim = [-0.1, 0.8]; plot_ylim = [-0.8, 0.4]; plot_zlim = [-0.05, 0.6];
plot_view = [45, 30];
show_cam_view = true;

%% ==========================================================
%% 2 ~ 4. メイン環境（ロボット・机・マット・固定カメラ）の構築
%% ==========================================================
% 💡 [関数化] 3D空間のベース環境を構築
[robot, q_home, fig_main, poly_fixed, cam_dir, cam_up] = create_main_environment(...
    urdfPath, env, cam_info, plot_xlim, plot_ylim, plot_zlim, plot_view);

%% ==========================================================
%% 5. 手先カメラの数学的準備 ＆ 更新用オブジェクトの生成
%% ==========================================================
% 💡 [関数化] 手先カメラの座標系・光線ベクトルの初期設定
[R_cam_local, offset_local, local_rays, cam_hand_fov_v] = setup_hand_camera_geometry(cam_hand_info);

% リアルタイム更新用のグラフィックスオブジェクトを配置
handles.poly_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [0, 1.0, 0.5], 'FaceAlpha', 0.85, 'EdgeColor', [0, 1.0, 0.5], 'LineWidth', 2);
handles.hand_frustum_line = plot3(NaN, NaN, NaN, 'Color', [1 0.5 0], 'LineWidth', 2);
handles.hand_cam_marker = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
handles.hand_drop_line = plot3(NaN, NaN, NaN, '--', 'Color', [1 0.5 0], 'LineWidth', 1.5);
handles.hand_footprint_patch = patch('XData', [], 'YData', [], 'ZData', [], 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
handles.hand_frustum_patch = patch('Faces', [1 2 3; 1 3 4; 1 4 5; 1 5 2], 'Vertices', zeros(5,3), 'FaceColor', [1 0.5 0], 'FaceAlpha', 0.25, 'EdgeColor', [1 0.5 0], 'LineWidth', 1.5);

hold off;
rotate3d(fig_main.CurrentAxes, 'on');

%% ==========================================================
%% 6. 別ウィンドウ（固定カメラ視点・手先カメラ視点）の初期化
%% ==========================================================
% 💡 [変更] 上下分割の1つのウィンドウにまとめる！
[f_cams, ax_cam, robot_cam, ax_cam_hand, robot_cam_hand] = init_camera_views(...
    show_cam_view, urdfPath, q_home, env, cam_info, cam_dir, cam_up, cam_hand_info);

%% ==========================================================
%% 7. 実行後のインタラクティブ操作（メインループ）
%% ==========================================================
figure(fig_main);
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
z_target = env.desk_z_offset + env.mat_z_offset;

% --- リアルタイムループ ---
while ishandle(fig_main)
    for i = 1:6
        q_current(i).JointPosition = get(sliders(i), 'Value');
    end
    
    % メイン画面のロボットを更新
    show(robot, q_current, 'Parent', fig_main.CurrentAxes, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
    
    % 手先カメラの順運動学計算
    T_link = getTransform(robot, q_current, cam_hand_info.attached_link);
    R_link = T_link(1:3, 1:3); P_link = T_link(1:3, 4);
    P_global = P_link + R_link * offset_local;
    R_global = R_link * R_cam_local;
    global_rays = R_global * local_rays;

    % 💡 [修正] f_cam の古いコードを削除し、f_cams のブロック1つに完全統合！
    if show_cam_view && ishandle(f_cams)
        % 👆 上半分：固定カメラの更新
        show(robot_cam, q_current, 'Parent', ax_cam, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
        set(ax_cam, 'CameraPosition', [cam_info.x, cam_info.y, cam_info.z], ...
                    'CameraTarget', [cam_info.x+cam_dir(1), cam_info.y+cam_dir(2), cam_info.z+cam_dir(3)], ...
                    'CameraUpVector', cam_up, 'CameraViewAngle', cam_info.fov_v_deg);
        
        % 👇 下半分：手先カメラ(FPV)の更新
        show(robot_cam_hand, q_current, 'Parent', ax_cam_hand, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
        cam_dir_hand = R_global * [1; 0; 0];
        cam_up_hand  = R_global * [0; 0; 1]; 
        set(ax_cam_hand, 'CameraPosition', P_global', 'CameraTarget', (P_global + cam_dir_hand)', ...
                         'CameraUpVector', cam_up_hand', 'CameraViewAngle', cam_hand_fov_v);
    end
    
    % マーカーと垂直点線の位置更新
    set(handles.hand_cam_marker, 'XData', P_global(1), 'YData', P_global(2), 'ZData', P_global(3));
    set(handles.hand_drop_line, 'XData', [P_global(1), P_global(1)], 'YData', [P_global(2), P_global(2)], 'ZData', [P_global(3), z_target]);

    % 幾何学的な視野の重なり・凸包ポリゴンを計算して画面を更新
    update_camera_projections(P_global, global_rays, z_target, env, poly_fixed, handles);
    
    drawnow;
end

end % === メイン関数終了 ===


%% ==========================================================
%% 🛠️ 以下、独立したローカル関数群（サブルーチン）
%% ==========================================================

function [env, cam_info, cam_hand_info] = load_environment_config(jsonFile, dataset_id)
    % JSON設定ファイルを読み込んで構造体にパースする関数
    jsonText = fileread(jsonFile);
    config = jsondecode(jsonText);
    dataset_info = config.datasets.(dataset_id);
    fprintf('🚀 データセットを読み込みました: %s (カメラ数: %d)\n', dataset_info.name, dataset_info.num_cameras);
    
    env = config.environment;
    cam_info = config.cameras.(dataset_info.camera_ids{1});
    cam_hand_info = config.cameras.(dataset_info.camera_ids{2});
end


function [robot, q_home, fig_main, poly_fixed, cam_dir, cam_up] = create_main_environment(urdfPath, env, cam_info, xlims, ylims, zlims, view_angle)
    % メインの3Dシミュレーション空間（机・マット・固定カメラ等）を生成する関数
    fig_main = figure('Name', 'SO-101 Task Environment', 'Color', 'w', 'Position', [50, 150, 750, 700]);
    robot = importrobot(urdfPath);
    q_home = homeConfiguration(robot);
    show(robot, q_home, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
    hold on;

    % 机の描画
    X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
    Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
    Z_desk = repmat(env.desk_z_offset, 1, 4);
    patch('XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 0.5, 'EdgeColor', [0.5 0.5 0.5]);

    % マットの描画（Zファイティング防止で4mm浮かす）
    X_mat = [env.mat_x_start, env.mat_x_start + env.mat_length, env.mat_x_start + env.mat_length, env.mat_x_start];
    Y_mat = [env.mat_y_start, env.mat_y_start, env.mat_y_start + env.mat_width, env.mat_y_start + env.mat_width];
    z_target = env.desk_z_offset + env.mat_z_offset;
    Z_mat = repmat(z_target + 0.004, 1, 4);
    patch('XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k');

    % 固定カメラの目印と点線
    plot3([cam_info.x, cam_info.x], [cam_info.y, cam_info.y], [z_target, cam_info.z], 'r--', 'LineWidth', 1.5);
    plot3(cam_info.x, cam_info.y, cam_info.z, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(cam_info.x, cam_info.y, cam_info.z + 0.04, '📷 固定カメラ', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 11, 'HorizontalAlignment', 'center');
    
    % 固定カメラの視野ポリゴン計算
    pitch = deg2rad(cam_info.pitch_deg); 
    cam_dir = [cos(pitch), 0, sin(pitch)]; cam_right = [0, -1, 0]; cam_up = [-sin(pitch), 0, cos(pitch)];
    w = tan(deg2rad(cam_info.fov_h_deg) / 2); h = tan(deg2rad(cam_info.fov_v_deg) / 2);
    rays = [cam_dir + w*cam_right + h*cam_up; cam_dir - w*cam_right + h*cam_up; ...
            cam_dir - w*cam_right - h*cam_up; cam_dir + w*cam_right - h*cam_up];

    t = (z_target - cam_info.z) ./ rays(:, 3); 
    intersect_pts_fixed = repmat([cam_info.x, cam_info.y, cam_info.z], 4, 1) + t .* rays; 
    poly_fixed = polyshape(intersect_pts_fixed(:,1), intersect_pts_fixed(:,2));

    % 固定カメラの青い足元投影面（8mm浮かす）
    plot3([intersect_pts_fixed(:,1); intersect_pts_fixed(1,1)], [intersect_pts_fixed(:,2); intersect_pts_fixed(1,2)], repmat(z_target+0.008, 5, 1), 'g-', 'Color','#4646C9' ,'LineWidth', 2);
    patch('XData', intersect_pts_fixed(:,1), 'YData', intersect_pts_fixed(:,2), 'ZData', repmat(z_target+0.008, 4, 1), 'FaceColor', '#4646C9', 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    % 固定カメラの半透明ピラミッド
    patch('Faces', [1 2 3; 1 3 4; 1 4 5; 1 5 2], 'Vertices', [[cam_info.x, cam_info.y, cam_info.z]; intersect_pts_fixed], 'FaceColor', [0 0.8 1], 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    % 軸と視点の設定
    axis equal; grid on; view(view_angle); xlim(xlims); ylim(ylims); zlim(zlims);
end


function [R_cam_local, offset_local, local_rays, fov_v] = setup_hand_camera_geometry(cam_hand_info)
    % 手先カメラの座標変換行列とローカル視野光線を初期計算する関数
    y_rad = -deg2rad(cam_hand_info.yaw_deg); p_rad = deg2rad(cam_hand_info.pitch_deg); r_rad = deg2rad(cam_hand_info.roll_deg);
    R_json = eul2rotm([y_rad, p_rad, r_rad], 'YZX');
    R_tweak = eul2rotm([pi, 0, 0], 'ZYX'); % パターンB: 左右180度反転
    R_base = [1 0 0; 0 0 -1; 0 1 0] * [0 0 -1; 0 1 0; 1 0 0];

    R_opt = R_base * R_tweak;
    R_cam_local = R_json * R_opt;
    offset_local = [cam_hand_info.offset_x; cam_hand_info.offset_y; cam_hand_info.offset_z];
    
    % ==========================================================
    % 💡 [修正] 模倣学習の取得解像度（640x480）の比率に完全同期させる！
    % ==========================================================
    window_aspect = 640 / 480; % 4:3 の比率 (1.3333...)
    
    % 縦の広がり(h_h)はJSONの垂直FOV(49.1度)から厳密に計算
    h_h = tan(deg2rad(cam_hand_info.fov_v_deg)/2);
    
    % 横の広がり(w_h)は、JSONの値ではなく画面比率（640x480）から物理的に逆算する！
    % これにより、3D空間のピラミッドと実際のレンダリング映像が完全に一致します。
    w_h = h_h * window_aspect; 
    
    local_rays = [1, -w_h, h_h; 1, w_h, h_h; 1, w_h, -h_h; 1, -w_h, -h_h]';
    fov_v = cam_hand_info.fov_v_deg;
end

function [f_cams, ax_cam, robot_cam, ax_cam_hand, robot_cam_hand] = init_camera_views(show_cam_view, urdfPath, q_home, env, cam_info, cam_dir, cam_up, cam_hand_info)
    % 縦に並べた統合カメラウィンドウを生成する関数
    f_cams = []; ax_cam = []; robot_cam = []; ax_cam_hand = []; robot_cam_hand = [];
    if ~show_cam_view, return; end

    X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
    Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
    Z_desk = repmat(env.desk_z_offset, 1, 4);
    X_mat = [env.mat_x_start, env.mat_x_start + env.mat_length, env.mat_x_start + env.mat_length, env.mat_x_start];
    Y_mat = [env.mat_y_start, env.mat_y_start, env.mat_y_start + env.mat_width, env.mat_y_start + env.mat_width];
    Z_mat = repmat(env.desk_z_offset + env.mat_z_offset, 1, 4);

    % ==========================================================
    % 💡 [変更] ウィンドウを一回り縮小（480 × 720）※比率は4:3を完全キープ！
    % ==========================================================
    f_cams = figure('Name', 'Camera Views (Top: Fixed, Bottom: FPV)', 'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none', 'Resize', 'off');
    set(f_cams, 'Position', [850, 100, 480, 720]);

    % ==========================================================
    % 💡 [変更] パネルに「タイトル（見出し）」と「境界線(line)」を追加！
    % ==========================================================
    pnl_top = uipanel('Parent', f_cams, 'Position', [0, 0.5, 1.0, 0.5], ...
                      'BackgroundColor', 'w', 'BorderType', 'line', ...
                      'Title', ' 📷 固定カメラ ', 'FontSize', 11, 'FontWeight', 'bold');
                      
    pnl_bot = uipanel('Parent', f_cams, 'Position', [0, 0.0, 1.0, 0.5], ...
                      'BackgroundColor', 'w', 'BorderType', 'line', ...
                      'Title', ' 🦾 手先カメラ (FPV) ', 'FontSize', 11, 'FontWeight', 'bold');

    % --- 👆 上半分: 固定カメラ ---
    ax_cam = axes('Parent', pnl_top, 'Position', [0, 0, 1, 1]); 
    robot_cam = importrobot(urdfPath);
    show(robot_cam, q_home, 'Parent', ax_cam, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off'); 
    hold(ax_cam, 'on');    
    
    patch('Parent', ax_cam, 'XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('Parent', ax_cam, 'XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    
    axis(ax_cam, 'equal'); axis(ax_cam, 'vis3d'); axis(ax_cam, 'off'); 
    axes(ax_cam); % 現在の軸としてセット
    camproj('perspective'); campos([cam_info.x, cam_info.y, cam_info.z]); 
    camtarget([cam_info.x+cam_dir(1), cam_info.y+cam_dir(2), cam_info.z+cam_dir(3)]); camup(cam_up); camva(cam_info.fov_v_deg); 
    set(ax_cam, 'CameraPositionMode', 'manual', 'CameraTargetMode', 'manual', 'CameraUpVectorMode', 'manual', 'CameraViewAngleMode', 'manual', 'XLimMode', 'manual', 'YLimMode', 'manual', 'ZLimMode', 'manual');

    % --- 👇 下半分: 手先カメラ ---
    ax_cam_hand = axes('Parent', pnl_bot, 'Position', [0, 0, 1, 1]); 
    robot_cam_hand = importrobot(urdfPath);
    show(robot_cam_hand, q_home, 'Parent', ax_cam_hand, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off'); 
    hold(ax_cam_hand, 'on');    
    
    patch('Parent', ax_cam_hand, 'XData', X_desk, 'YData', Y_desk, 'ZData', Z_desk, 'FaceColor', [0.85, 0.76, 0.65], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    patch('Parent', ax_cam_hand, 'XData', X_mat, 'YData', Y_mat, 'ZData', Z_mat, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
    
    axis(ax_cam_hand, 'equal'); axis(ax_cam_hand, 'vis3d'); axis(ax_cam_hand, 'off'); 
    axes(ax_cam_hand); % 現在の軸としてセット
    camproj('perspective'); camva(cam_hand_info.fov_v_deg); 
    set(ax_cam_hand, 'CameraPositionMode', 'manual', 'CameraTargetMode', 'manual', 'CameraUpVectorMode', 'manual', 'CameraViewAngleMode', 'manual', 'XLimMode', 'manual', 'YLimMode', 'manual', 'ZLimMode', 'manual');
end

function update_camera_projections(P_global, global_rays, z_target, env, poly_fixed, handles)
    % 毎フレーム呼び出され、幾何学的な視野交差・凸包・型抜き・重なり面を描画更新する関数
    MAX_DIST = 2.0; 
    F = zeros(4, 3);
    for i = 1:4
        F(i, :) = (P_global + MAX_DIST * global_rays(:, i))';
    end
    
    % 空中のオレンジピラミッド更新
    set(handles.hand_frustum_patch, 'Vertices', [P_global'; F]);
    
    % ピラミッドを構成する8本の辺
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
    
    % 机に交差している場合
    if size(footprint_pts, 1) >= 3
        k = convhull(footprint_pts(:,1), footprint_pts(:,2));
        hull_pts = footprint_pts(k, :);
        poly_hull = polyshape(hull_pts(:,1), hull_pts(:,2));
        
        % 机のサイズで型抜き（Zファイティング防止）
        X_desk = [env.desk_x_min, env.desk_x_max, env.desk_x_max, env.desk_x_min];
        Y_desk = [env.desk_y_min, env.desk_y_min, env.desk_y_max, env.desk_y_max];
        poly_desk = polyshape(X_desk, Y_desk);
        poly_hand = intersect(poly_hull, poly_desk);
        
        if poly_hand.NumRegions > 0
            v_hand = poly_hand.Vertices;
            v_hand(isnan(v_hand(:,1)), :) = []; 
            
            % オレンジ枠と影を更新（12mm浮かす）
            set(handles.hand_frustum_line, 'XData', [v_hand(:,1); v_hand(1,1)], 'YData', [v_hand(:,2); v_hand(1,2)], 'ZData', repmat(z_target+0.012, size(v_hand,1)+1, 1));
            set(handles.hand_footprint_patch, 'XData', v_hand(:,1), 'YData', v_hand(:,2), 'ZData', repmat(z_target+0.012, size(v_hand,1), 1));
            
            % 固定カメラとの重なり（緑ポリゴン）を計算
            poly_intersect = intersect(poly_fixed, poly_hand);
            if poly_intersect.NumRegions > 0
                v_int = poly_intersect.Vertices;
                v_int(isnan(v_int(:,1)), :) = [];
                % 緑の重なり面を更新（16mm浮かす）
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
    % 視野が外れた際に机の上の描画をすべて非表示にする関数
    set(handles.hand_frustum_line, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    set(handles.hand_footprint_patch, 'XData', [], 'YData', [], 'ZData', []);
    set(handles.poly_patch, 'XData', [], 'YData', [], 'ZData', []); 
end

