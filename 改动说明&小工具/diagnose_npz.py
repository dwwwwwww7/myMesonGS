#!/usr/bin/env python3
"""
诊断NPZ文件问题
检查文件是否存在以及内容是否正确
"""

import os
import sys
import numpy as np

def check_npz_structure(model_path, iteration):
    """检查NPZ文件结构"""
    
    # 尝试两种路径格式
    possible_paths = [
        os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "pc_npz", "bins"),
        os.path.join(model_path, "point_cloud", f"iter_{iteration}", "pc_npz", "bins"),
    ]
    
    bins_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            bins_dir = path
            break
    
    if bins_dir is None:
        print(f"✗ 错误: 未找到bins目录")
        print(f"\n尝试的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    print(f"✓ 找到bins目录: {bins_dir}\n")
    
    # 检查必要的文件
    required_files = {
        "oct.npz": ["points", "params"],
        "orgb.npz": ["f", "i"],
        "t.npz": ["t"]
    }
    
    all_ok = True
    
    for filename, expected_keys in required_files.items():
        filepath = os.path.join(bins_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ 缺少文件: {filename}")
            all_ok = False
            continue
        
        print(f"✓ {filename} 存在")
        
        try:
            data = np.load(filepath)
            print(f"  包含的键: {list(data.keys())}")
            
            # 检查预期的键
            missing_keys = [k for k in expected_keys if k not in data.keys()]
            if missing_keys:
                print(f"  ✗ 缺少键: {missing_keys}")
                all_ok = False
            else:
                print(f"  ✓ 所有预期的键都存在")
            
            # 显示数据形状
            for key in data.keys():
                arr = data[key]
                if hasattr(arr, 'shape'):
                    print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    print(f"    {key}: {arr}")
            
            data.close()
            
        except Exception as e:
            print(f"  ✗ 读取文件时出错: {e}")
            all_ok = False
        
        print()
    
    # 检查是否有旧格式的文件（应该不存在）
    old_files = ["ntk.npz", "um.npz", "ct.npz"]
    old_files_found = []
    
    for filename in old_files:
        filepath = os.path.join(bins_dir, filename)
        if os.path.exists(filepath):
            old_files_found.append(filename)
    
    if old_files_found:
        print(f"⚠ 警告: 发现旧格式文件（应该已被删除）:")
        for f in old_files_found:
            print(f"  - {f}")
        print()
    
    # 验证维度一致性
    if all_ok:
        print("验证维度一致性...")
        try:
            t_data = np.load(os.path.join(bins_dir, "t.npz"))["t"]
            orgb_data = np.load(os.path.join(bins_dir, "orgb.npz"))
            
            depth = int(t_data[0])
            n_block = int(t_data[1])
            
            orgb_f = orgb_data["f"]
            orgb_i = orgb_data["i"]
            
            print(f"  depth: {depth}")
            print(f"  n_block: {n_block}")
            print(f"  DC系数 (f) 形状: {orgb_f.shape}")
            print(f"  AC系数 (i) 形状: {orgb_i.shape}")
            
            # 检查是否是55维
            if orgb_f.shape[0] == 55:
                print(f"  ✓ DC系数是55维（正确）")
            else:
                print(f"  ✗ DC系数不是55维（当前: {orgb_f.shape[0]}）")
                all_ok = False
            
            # 计算预期的AC系数大小
            expected_params = 2 + 55 * n_block * 2
            actual_params = len(t_data)
            
            print(f"  预期量化参数数量: {expected_params}")
            print(f"  实际量化参数数量: {actual_params}")
            
            if expected_params == actual_params:
                print(f"  ✓ 量化参数数量正确")
            else:
                print(f"  ✗ 量化参数数量不匹配")
                all_ok = False
            
        except Exception as e:
            print(f"  ✗ 验证时出错: {e}")
            all_ok = False
    
    return all_ok

def main():
    if len(sys.argv) < 3:
        print("用法: python diagnose_npz.py <MODEL_PATH> <ITERATION>")
        print("示例: python diagnose_npz.py D:/3DGS_seq/result_mesongs/output/truck_config3 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    iteration = sys.argv[2]
    
    print("="*70)
    print("诊断NPZ文件")
    print("="*70)
    print(f"模型路径: {model_path}")
    print(f"迭代: {iteration}")
    print()
    
    if not os.path.exists(model_path):
        print(f"✗ 错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    success = check_npz_structure(model_path, iteration)
    
    print("="*70)
    if success:
        print("✓ 所有检查通过！NPZ文件结构正确。")
        print("\n可以使用以下命令进行渲染:")
        print(f"python render.py -s <SCENE_PATH> -m {model_path} --iteration {iteration} --dec_npz --skip_train")
    else:
        print("✗ 发现问题，请检查上述错误信息。")
        print("\n可能的解决方案:")
        print("  1. 重新运行训练")
        print("  2. 检查保存代码是否正确")
        print("  3. 确认使用的是最新版本的代码")
    print("="*70)

if __name__ == "__main__":
    main()
