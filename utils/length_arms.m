% アームの長さを表示するスクリプト
% 1. 一度念のためにパスのキャッシュをクリアする
rehash

% 2. ロボットの読み込みと骨の長さの一覧表示
robot = importrobot('/home/hogehoge/Genesis/examples/robots/so101/so101_new.urdf');

fprintf('\n=== ロボットの骨の長さ（オフセット）一覧 ===\n');
for i = 1:robot.NumBodies
    body = robot.Bodies{i};
    T = body.Joint.JointToParentTransform;
    pos = T(1:3, 4)'; 
    fprintf('%-20s : XYZ = [%8.4f, %8.4f, %8.4f] (距離: %.4f m)\n', ...
        body.Joint.Name, pos(1), pos(2), pos(3), norm(pos));
end
fprintf('===========================================\n');