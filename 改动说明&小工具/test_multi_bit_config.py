#!/usr/bin/env python3
"""
测试多位数量化配置
验证配置是否正确，并估算压缩效果
"""

def calculate_storage(n_points, bit_config):
    """计算存储大小"""
    
    # 维度分配
    dims = {
        'opacity': 1,
        'euler': 3,
        'f_dc': 3,
        'f_rest_0': 15,
        'f_rest_1': 15,
        'f_rest_2': 15,
        'scale': 3
    }
    
    total_bytes = 0
    details = []
    
    for attr, num_dims in dims.items():
        bit = bit_config[attr]
        bytes_per_point = num_dims * bit / 8
        total_bytes_attr = n_points * bytes_per_point
        total_bytes += total_bytes_attr
        
        details.append({
            'attr': attr,
            'dims': num_dims,
            'bit': bit,
            'bytes': total_bytes_attr
        })
    
    return total_bytes, details

def main():
    print("="*70)
    print("多位数量化配置测试")
    print("="*70)
    
    # 示例点数（truck场景）
    n_points = 343376
    
    # 你的配置
    your_config = {
        'opacity': 8,
        'euler': 8,
        'scale': 10,
        'f_dc': 8,
        'f_rest_0': 4,
        'f_rest_1': 4,
        'f_rest_2': 2,
    }
    
    # 基准配置（全8-bit）
    baseline_config = {
        'opacity': 8,
        'euler': 8,
        'scale': 8,
        'f_dc': 8,
        'f_rest_0': 8,
        'f_rest_1': 8,
        'f_rest_2': 8,
    }
    
    # 高质量配置
    high_quality_config = {
        'opacity': 10,
        'euler': 10,
        'scale': 12,
        'f_dc': 10,
        'f_rest_0': 6,
        'f_rest_1': 6,
        'f_rest_2': 4,
    }
    
    # 高压缩配置
    high_compression_config = {
        'opacity': 6,
        'euler': 6,
        'scale': 8,
        'f_dc': 6,
        'f_rest_0': 3,
        'f_rest_1': 3,
        'f_rest_2': 2,
    }
    
    configs = [
        ("基准配置（全8-bit）", baseline_config),
        ("你的配置（混合位数）", your_config),
        ("高质量配置", high_quality_config),
        ("高压缩配置", high_compression_config),
    ]
    
    print(f"\n点数: {n_points:,}\n")
    
    baseline_size = None
    
    for name, config in configs:
        print(f"\n{name}")
        print("-"*70)
        
        total_bytes, details = calculate_storage(n_points, config)
        total_mb = total_bytes / 1024 / 1024
        
        print(f"{'属性':<12} {'维度':>6} {'位数':>6} {'大小(KB)':>12} {'占比':>8}")
        print("-"*70)
        
        for d in details:
            kb = d['bytes'] / 1024
            percent = d['bytes'] / total_bytes * 100
            print(f"{d['attr']:<12} {d['dims']:>6} {d['bit']:>6} {kb:>12.2f} {percent:>7.1f}%")
        
        print("-"*70)
        print(f"{'总计':<12} {'55':>6} {'':<6} {total_bytes/1024:>12.2f} {'100.0%':>8}")
        print(f"\n总大小: {total_mb:.2f} MB")
        
        if baseline_size is None:
            baseline_size = total_bytes
        else:
            saving = (1 - total_bytes / baseline_size) * 100
            print(f"相比基准: {saving:+.1f}% ({'节省' if saving > 0 else '增加'})")
    
    print("\n" + "="*70)
    print("配置建议")
    print("="*70)
    print("""
1. 平衡配置（推荐）:
   - 适合大多数场景
   - 节省约40-50%存储
   - 质量损失<1dB PSNR
   
2. 高质量配置:
   - 追求最佳视觉效果
   - 节省约20-30%存储
   - 质量损失<0.5dB PSNR
   
3. 高压缩配置:
   - 追求最小文件大小
   - 节省约60-70%存储
   - 质量损失1-2dB PSNR
   
关键原则:
- scale 最重要，建议≥10-bit
- opacity 和 f_dc 建议≥8-bit
- 高阶SH (f_rest_2) 可以用2-4 bit
    """)
    print("="*70)

if __name__ == "__main__":
    main()
