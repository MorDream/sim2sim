import os
import sys
import trimesh

def convert_dae_to_stl(folder_path):
    if not os.path.isdir(folder_path):
        print(f"路径无效：{folder_path}")
        return

    # 获取所有 .dae 文件
    dae_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.dae')]

    if not dae_files:
        print("未找到 .dae 文件。")
        return

    for dae_file in dae_files:
        dae_path = os.path.join(folder_path, dae_file)
        stl_path = os.path.join(folder_path, os.path.splitext(dae_file)[0] + '.stl')

        try:
            mesh = trimesh.load(dae_path)
            if mesh.is_empty:
                print(f"跳过空模型：{dae_file}")
                continue

            mesh.export(stl_path)
            print(f"转换成功：{dae_file} -> {os.path.basename(stl_path)}")

        except Exception as e:
            print(f"转换失败：{dae_file}，错误信息：{e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法：python convert.py <文件夹路径>")
    else:
        convert_dae_to_stl(sys.argv[1])
