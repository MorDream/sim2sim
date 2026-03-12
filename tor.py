import torch

def inspect_pt_file(file_path):
    # 尝试加载文件
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"文件类型: {type(data)}")

    # 1. 如果是字典 (State Dict 或 Checkpoint)
    if isinstance(data, dict):
        print("内容摘要: 字典包含以下键:", data.keys())
        # 尝试查找模型权重相关的键
        for key in ['model', 'state_dict', 'model_state_dict', 'policy']:
            if key in data:
                print(f"\n--- 发现关键组件: {key} ---")
                inner_data = data[key]
                if isinstance(inner_data, dict):
                    print_dict_shapes(inner_data)
                return
        # 如果没有上述键，直接打印根字典
        print_dict_shapes(data)

    # 2. 如果是导出的完整模型 (TorchScript)
    elif isinstance(data, torch.jit.ScriptModule) or hasattr(data, 'graph'):
        print("检测到 TorchScript 模型")
        # 打印输入输出信息
        try:
            print(f"输入层结构: {list(data.graph.inputs())}")
            print(f"输出层结构: {list(data.graph.outputs())}")
        except:
            print("无法解析计算图，尝试查看模型参数:")
            for name, param in data.named_parameters():
                print(f"{name}: {param.shape}")

    # 3. 如果是普通的 nn.Module 对象
    elif isinstance(data, torch.nn.Module):
        print("检测到 PyTorch nn.Module 对象")
        print(data)

    else:
        print("未知格式，尝试直接打印数据内容:")
        print(data)

def print_dict_shapes(d):
    """ 辅助函数：打印字典中 Tensor 的形状 """
    items = list(d.items())
    if not items:
        print("字典为空")
        return
    
    print(f"{'层名':<40} | {'形状':<20}")
    print("-" * 65)
    for name, value in items:
        if isinstance(value, torch.Tensor):
            print(f"{name:<40} | {str(list(value.shape)):<20}")

# 替换成你的文件名
inspect_pt_file(r'/root/mym/parkour-main/legged_gym/logs/field_go2/Feb04_02-53-08_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromFeb03_07-44-57/model_30000.pt')