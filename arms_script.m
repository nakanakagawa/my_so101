% ロボットの構造解析スクリプト
robot = importrobot('/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf');
q_test = homeConfiguration(robot);
for i=1:length(q_test), q_test(i).JointPosition = 0; end

fprintf('=== 🤖 初期姿勢(全関節0度)の絶対座標 ===\n');
links = {'shoulder_lift', 'elbow_flex', 'wrist_flex', 'camera_mount_fixed_joint'};
for i = 1:length(links)
    try
        T = getTransform(robot, q_test, links{i});
        fprintf('%-25s : X=%7.4f, Y=%7.4f, Z=%7.4f\n', links{i}, T(1,4), T(2,4), T(3,4));
    catch
    end
end

fprintf('\n=== 🔄 回転軸の極性テスト ===\n');
q_test(2).JointPosition = 0.5; T2 = getTransform(robot, q_test, 'elbow_flex');
fprintf('肩(J2)を +0.5rad  -> 肘の位置: X=%7.4f, Z=%7.4f\n', T2(1,4), T2(3,4));
q_test(2).JointPosition = 0;

q_test(3).JointPosition = 0.5; T3 = getTransform(robot, q_test, 'wrist_flex');
fprintf('肘(J3)を +0.5rad  -> 手首の位置: X=%7.4f, Z=%7.4f\n', T3(1,4), T3(3,4));