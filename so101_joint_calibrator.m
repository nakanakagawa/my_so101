function so101_joint_calibrator()
    % URDFのパス
    urdfPath = '/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf';
    
    % ロボットの読み込み
    robot = importrobot(urdfPath);
    q_home = homeConfiguration(robot);
    
    % UIフィギュアの作成
    fig = figure('Name', 'SO-101 Joint Limit Calibrator', 'Position', [100, 100, 1000, 600], 'Color', 'w');
    
    % 3D描画エリア（左側）
    ax = axes(fig, 'Position', [0.05, 0.1, 0.55, 0.8]);
    rotate3d(ax, 'on');
    show(robot, q_home, 'Parent', ax, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
    view(ax, [45, 30]); axis(ax, 'equal'); grid(ax, 'on');
    title(ax, '🤖 姿勢プレビュー', 'FontSize', 14);
    
    % 🚨【大修正】URDFから「動かせる関節」を自動で探し出す！
    movable_body_names = {};
    joint_names = {};
    for i = 1:robot.NumBodies
        b = robot.Bodies{i};
        % 関節が 'fixed'（固定）じゃなければリストに追加
        if ~strcmp(b.Joint.Type, 'fixed') 
            movable_body_names{end+1} = b.Name;      % リンク名 (getBody用)
            joint_names{end+1} = b.Joint.Name;       % ジョイント名 (表示用)
        end
    end
    num_joints = length(joint_names);
    
    % スライダーとテキスト用の配列
    sliders = gobjects(num_joints, 1);
    val_texts = gobjects(num_joints, 1);
    
    % UIパネル（右側）
    pnl = uipanel(fig, 'Position', [0.62, 0.05, 0.35, 0.9], 'Title', '🎮 関節角度コントローラー', 'BackgroundColor', 'w', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 各関節のスライダーを生成
    for i = 1:num_joints
        % getBodyには正しい「リンク名」を渡して関節情報を取得する
        jnt = robot.getBody(movable_body_names{i}).Joint;
        lims = jnt.PositionLimits;
        
        if isinf(lims(1)), lims = [-pi, pi]; end 
        
        y_pos = 0.9 - (i-1)*(0.9/num_joints);
        
        % ラベル（関節名）
        uicontrol(pnl, 'Style', 'text', 'String', joint_names{i}, ...
            'Units', 'normalized', 'Position', [0.05, y_pos, 0.9, 0.05], ...
            'HorizontalAlignment', 'left', 'BackgroundColor', 'w', 'FontWeight', 'bold', 'ForegroundColor', 'b');
        
        % 値表示テキスト
        val_texts(i) = uicontrol(pnl, 'Style', 'text', 'String', '0.0° ( 0.000 rad )', ...
            'Units', 'normalized', 'Position', [0.4, y_pos, 0.55, 0.05], ...
            'HorizontalAlignment', 'right', 'BackgroundColor', 'w', 'FontWeight', 'bold');
        
        % スライダー本体
        sliders(i) = uicontrol(pnl, 'Style', 'slider', ...
            'Units', 'normalized', 'Position', [0.05, y_pos-0.05, 0.9, 0.05], ...
            'Min', lims(1), 'Max', lims(2), 'Value', 0);
        
        % スライダーを動かした時のイベント
        addlistener(sliders(i), 'ContinuousValueChange', @(src, event) updateRobot(src, i));
    end
    
    % リセットボタン
    uicontrol(pnl, 'Style', 'pushbutton', 'String', '🔄 全て0度に戻す', ...
        'Units', 'normalized', 'Position', [0.05, 0.02, 0.9, 0.08], ...
        'FontSize', 12, 'FontWeight', 'bold', 'Callback', @(~,~) resetRobot());

    % ==========================================
    % 内部関数：スライダーが動いた時の処理
    % ==========================================
    function updateRobot(src, joint_idx)
        val_rad = src.Value;
        val_deg = rad2deg(val_rad); 
        
        val_texts(joint_idx).String = sprintf('%5.1f° ( %6.3f rad )', val_deg, val_rad);
        
        q_current = q_home;
        for j = 1:num_joints
            for k = 1:length(q_current)
                if strcmp(q_current(k).JointName, joint_names{j})
                    q_current(k).JointPosition = sliders(j).Value;
                    break;
                end
            end
        end
        show(robot, q_current, 'Parent', ax, 'PreservePlot', false, 'FastUpdate', true, 'Frames', 'off');
        drawnow;
    end

    function resetRobot()
        for j = 1:num_joints
            sliders(j).Value = 0;
            updateRobot(sliders(j), j);
        end
    end
end