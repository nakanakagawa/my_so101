% ｓimulinkにURDFインポートするファイル
% 検索パスの拡張とSimscape Multibodyへのインポート
% ==========================================================

% 1. URDFおよびアセットが配置されている絶対パスを定義
robot_root_dir = '/home/hogehoge/Genesis/examples/robots/so101';

% 2. ロボットのルートディレクトリと、メッシュが入ったassetsディレクトリを検索パスに追加
addpath(robot_root_dir);
addpath(fullfile(robot_root_dir, 'assets'));

% 3. カレントディレクトリを変更せずにURDFファイルをインポート
% (検索パスが通っているため、ファイル名のみの指定で認識されます)
smimport('so101_new.urdf');