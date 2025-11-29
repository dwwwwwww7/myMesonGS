#!/usr/bin/env python3
"""
诊断和修复渲染路径问题
自动查找正确的NPZ文件路径
"""

import os
import sys
import glob

def find_npz_dirs(model_path):
    """查找所有可能的NPZ目录"""
    patterns = [
        os.path.join(model_path, "point_cloud", "iteration_*", "pc_npz"),
        os.path.join(model_path, "point_cloud", "iter_*", "pc_npz"),
    ]
    
    found_dirs = []
    for pattern in patterns:
        dirs = glob.glob(pattern)
        found_dirs.extend(dirs)
    
    return found_dirs

def check_npz_files(npz_dir):
    """检查NPZ目录中的必要文件"""
    bins_dir = os.path.join(npz_dir, "bins")
    if not os.path.exists(bins_dir):
        return False, "bins目录不存在"
    
    required_files = ["oct.npz", "orgb.npz", "t.npz"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(bins_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        return False, f"缺少文件: {', '.join(missing_files)}"
    
    return True, "所有必要文件都存在"

def extract_iteration_number(path):
    """从路径中提取迭代号"""
    # 尝试匹配 iteration_XX 或 iter_XX
    import re
    match = re.search(r'(?:iteration|iter)_(\w+)', path)
    if match:
        return match.group(1)
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: python fix_render_path.py <MODEL_PATH> [ITERATION]")
        print("示例: python fix_render_path.py D:/3DGS_seq/result_mesongs/output/truck_config3 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    target_iteration = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*70)
    print("诊断渲染路径问题")
    print("="*70)
    print(f"模型路径: {model_path}")
    if target_iteration:
        print(f"目标迭代: {target_iteration}")
    print()
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"✗ 错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    print("✓ 模型路径存在")
    
    # 查找所有NPZ目录
    print("\n查找NPZ目录...")
    npz_dirs = find_npz_dirs(model_path)
    
    if not npz_dirs:
        print("✗ 未找到任何NPZ目录")
        print("\n可能的原因:")
        print("  1. 训练尚未完成")
        print("  2. 保存路径不正确")
        print("  3. 文件被移动或删除")
        sys.exit(1)
    
    print(f"✓ 找到 {len(npz_dirs)} 个NPZ目录:\n")
    
    # 检查每个目录
    valid_dirs = []
    for npz_dir in npz_dirs:
        iteration = extract_iteration_number(npz_dir)
        is_valid, msg = check_npz_files(npz_dir)
        
        status = "✓" if is_valid else "✗"
        print(f"{status} 迭代 {iteration:8} | {npz_dir}")
        print(f"  {msg}")
        
        if is_valid:
            valid_dirs.append((iteration, npz_dir))
    
    if not valid_dirs:
        print("\n✗ 没有找到有效的NPZ目录")
        sys.exit(1)
    
    # 如果指定了目标迭代，查找匹配的
    if target_iteration:
        print(f"\n查找迭代 {target_iteration}...")
        matching = [d for it, d in valid_dirs if it == target_iteration]
        
        if matching:
            print(f"✓ 找到匹配的迭代: {matching[0]}")
            print("\n推荐的渲染命令:")
            print(f"python render.py \\")
            print(f"    -s <SCENE_PATH> \\")
            print(f"    -m {model_path} \\")
            print(f"    --iteration {target_iteration} \\")
            print(f"    --dec_npz \\")
            print(f"    --skip_train")
        else:
            print(f"✗ 未找到迭代 {target_iteration}")
            print(f"\n可用的迭代: {', '.join([it for it, _ in valid_dirs])}")
    else:
        print("\n可用的迭代:")
        for iteration, npz_dir in valid_dirs:
            print(f"  - {iteration}")
        
        print("\n推荐使用最新的迭代:")
        latest_iter, latest_dir = valid_dirs[-1]
        print(f"python render.py \\")
        print(f"    -s <SCENE_PATH> \\")
        print(f"    -m {model_path} \\")
        print(f"    --iteration {latest_iter} \\")
        print(f"    --dec_npz \\")
        print(f"    --skip_train")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
