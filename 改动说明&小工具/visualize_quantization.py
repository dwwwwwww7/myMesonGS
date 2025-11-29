#!/usr/bin/env python3
"""
可视化量化过程
展示量化器的分布和参数
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimSun'  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_quantization_params(model_path, iteration):
    """可视化量化参数"""
    
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
        print(f"错误: 未找到bins目录")
        return
    
    # 加载量化参数
    t_path = os.path.join(bins_dir, "t.npz")
    if not os.path.exists(t_path):
        print(f"错误: 未找到 t.npz")
        return
    
    trans_array = np.load(t_path)["t"]
    
    depth = int(trans_array[0])
    n_block = int(trans_array[1])
    
    print("="*70)
    print("量化参数分析")
    print("="*70)
    print(f"八叉树深度: {depth}")
    print(f"块数量: {n_block}")
    print(f"总量化器数量: {55 * n_block}")
    print(f"总参数数量: {len(trans_array)}")
    print()
    
    # 提取所有的 scale 和 zero_point
    params = trans_array[2:]  # 跳过 depth 和 n_block
    
    scales = []
    zero_points = []
    
    for i in range(0, len(params), 2):
        scales.append(params[i])
        zero_points.append(params[i+1])
    
    scales = np.array(scales)
    zero_points = np.array(zero_points)
    
    # 重塑为 [55, n_block]
    scales = scales.reshape(55, n_block)
    zero_points = zero_points.reshape(55, n_block)
    
    print("Scale 统计:")
    print(f"  最小值: {scales.min():.6f}")
    print(f"  最大值: {scales.max():.6f}")
    print(f"  平均值: {scales.mean():.6f}")
    print(f"  标准差: {scales.std():.6f}")
    print()
    
    print("Zero Point 统计:")
    print(f"  最小值: {zero_points.min():.2f}")
    print(f"  最大值: {zero_points.max():.2f}")
    print(f"  平均值: {zero_points.mean():.2f}")
    print(f"  标准差: {zero_points.std():.2f}")
    print()
    
    # 特征名称
    feature_names = (
        ["opacity"] + 
        [f"euler_{i}" for i in range(3)] + 
        [f"f_dc_{i}" for i in range(3)] + 
        [f"f_rest_{i}" for i in range(45)] + 
        [f"scale_{i}" for i in range(3)]
    )
    
    # 分析每个维度
    print("各维度 Scale 统计:")
    print(f"{'维度':<15} {'最小值':>12} {'最大值':>12} {'平均值':>12} {'标准差':>12}")
    print("-"*70)
    
    key_dims = [0, 1, 2, 3, 4, 5, 6, 7, 51, 52, 53, 54]  # 关键维度
    for i in key_dims:
        name = feature_names[i]
        print(f"{name:<15} {scales[i].min():>12.6f} {scales[i].max():>12.6f} "
              f"{scales[i].mean():>12.6f} {scales[i].std():>12.6f}")
    
    print()
    
    # 创建可视化
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Scale 热图
        im1 = axes[0, 0].imshow(scales, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Scale 热图 (55维 × 块数)')
        axes[0, 0].set_xlabel('块索引')
        axes[0, 0].set_ylabel('特征维度')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 添加关键维度标签
        key_positions = [0, 3, 6, 51, 54]
        key_labels = ['opacity', 'euler', 'f_dc', 'f_rest_end', 'scale']
        axes[0, 0].set_yticks(key_positions)
        axes[0, 0].set_yticklabels(key_labels)
        
        # 2. Scale 分布直方图
        axes[0, 1].hist(scales.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Scale 分布')
        axes[0, 1].set_xlabel('Scale 值')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Zero Point 热图
        im2 = axes[1, 0].imshow(zero_points, aspect='auto', cmap='coolwarm')
        axes[1, 0].set_title('Zero Point 热图 (55维 × 块数)')
        axes[1, 0].set_xlabel('块索引')
        axes[1, 0].set_ylabel('特征维度')
        plt.colorbar(im2, ax=axes[1, 0])
        axes[1, 0].set_yticks(key_positions)
        axes[1, 0].set_yticklabels(key_labels)
        
        # 4. 关键维度的 Scale 对比
        key_features = {
            'opacity': 0,
            'euler_0': 1,
            'f_dc_0': 4,
            'f_rest_0': 7,
            'scale_0': 52
        }
        
        for name, idx in key_features.items():
            axes[1, 1].plot(scales[idx], label=name, marker='o', markersize=3)
        
        axes[1, 1].set_title('关键维度的 Scale 变化（跨块）')
        axes[1, 1].set_xlabel('块索引')
        axes[1, 1].set_ylabel('Scale 值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(model_path, f"quantization_analysis_iter{iteration}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化图像已保存: {output_path}")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"可视化失败: {e}")
        print("(可能是因为没有安装 matplotlib)")
    
    print("="*70)

def main():
    if len(sys.argv) < 3:
        print("用法: python visualize_quantization.py <MODEL_PATH> <ITERATION>")
        print("示例: python visualize_quantization.py D:/3DGS_seq/result_mesongs/output/truck_config3 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    iteration = sys.argv[2]
    
    visualize_quantization_params(model_path, iteration)

if __name__ == "__main__":
    main()
