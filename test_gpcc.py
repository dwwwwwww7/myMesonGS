#!/usr/bin/env python3
"""
测试 GPCC 压缩/解压缩功能
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.gpcc_utils import compress_gpcc, decompress_gpcc


def test_gpcc_basic():
    """测试基本的 GPCC 压缩/解压缩功能"""
    print("="*70)
    print("测试 GPCC 压缩/解压缩功能")
    print("="*70)
    
    # 1. 创建测试数据
    print("\n1. 创建测试点云数据...")
    n_points = 1000
    # 创建随机 3D 点云（整数坐标）
    test_points = np.random.randint(0, 1000, size=(n_points, 3), dtype=np.int32)
    print(f"   测试点数: {n_points}")
    print(f"   数据形状: {test_points.shape}")
    print(f"   数据类型: {test_points.dtype}")
    print(f"   坐标范围: X[{test_points[:,0].min()}, {test_points[:,0].max()}], "
          f"Y[{test_points[:,1].min()}, {test_points[:,1].max()}], "
          f"Z[{test_points[:,2].min()}, {test_points[:,2].max()}]")
    
    # 2. 测试压缩
    print("\n2. 测试 GPCC 压缩...")
    try:
        compressed_data = compress_gpcc(test_points)
        print(f"   ✓ 压缩成功")
        print(f"   原始大小: {test_points.nbytes:,} bytes")
        print(f"   压缩后大小: {len(compressed_data):,} bytes")
        print(f"   压缩率: {len(compressed_data) / test_points.nbytes * 100:.2f}%")
    except Exception as e:
        print(f"   ✗ 压缩失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试解压缩
    print("\n3. 测试 GPCC 解压缩...")
    try:
        decompressed_points = decompress_gpcc(compressed_data)
        print(f"   ✓ 解压缩成功")
        print(f"   解压缩后点数: {decompressed_points.shape[0]}")
        print(f"   解压缩后形状: {decompressed_points.shape}")
        print(f"   解压缩后类型: {decompressed_points.dtype}")
    except Exception as e:
        print(f"   ✗ 解压缩失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 验证数据一致性
    print("\n4. 验证数据一致性...")
    
    # 检查点数
    if decompressed_points.shape[0] != test_points.shape[0]:
        print(f"   ✗ 点数不匹配: {decompressed_points.shape[0]} vs {test_points.shape[0]}")
        return False
    
    # 检查数据是否完全一致
    if np.array_equal(test_points, decompressed_points):
        print(f"   ✓ 数据完全一致（无损压缩）")
    else:
        # 计算差异
        diff = np.abs(test_points - decompressed_points)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"   ⚠ 数据有差异（有损压缩）")
        print(f"   最大差异: {max_diff}")
        print(f"   平均差异: {mean_diff:.4f}")
        
        # 检查差异是否在可接受范围内
        if max_diff > 10:
            print(f"   ✗ 差异过大，可能有问题")
            return False
        else:
            print(f"   ✓ 差异在可接受范围内")
    
    print("\n" + "="*70)
    print("✓ GPCC 功能测试通过")
    print("="*70)
    return True


def test_gpcc_with_real_data():
    """使用真实的八叉树数据测试 GPCC"""
    print("\n" + "="*70)
    print("测试 GPCC 与真实八叉树数据")
    print("="*70)
    
    # 1. 创建模拟八叉树数据
    print("\n1. 创建模拟八叉树数据...")
    depth = 10
    max_coord = 2 ** depth
    n_points = 5000
    
    # 生成八叉树坐标（Morton 编码的解码结果）
    voxel_coords = np.random.randint(0, max_coord, size=(n_points, 3), dtype=np.int32)
    print(f"   八叉树深度: {depth}")
    print(f"   点数: {n_points}")
    print(f"   坐标范围: [0, {max_coord})")
    
    # 2. 压缩
    print("\n2. 压缩八叉树数据...")
    try:
        compressed = compress_gpcc(voxel_coords)
        print(f"   ✓ 压缩成功")
        print(f"   原始大小: {voxel_coords.nbytes:,} bytes")
        print(f"   压缩后大小: {len(compressed):,} bytes")
        print(f"   压缩率: {len(compressed) / voxel_coords.nbytes * 100:.2f}%")
    except Exception as e:
        print(f"   ✗ 压缩失败: {e}")
        return False
    
    # 3. 解压缩
    print("\n3. 解压缩八叉树数据...")
    try:
        decompressed = decompress_gpcc(compressed)
        print(f"   ✓ 解压缩成功")
        print(f"   解压缩后点数: {decompressed.shape[0]}")
    except Exception as e:
        print(f"   ✗ 解压缩失败: {e}")
        return False
    
    # 4. 验证
    print("\n4. 验证数据...")
    if decompressed.shape[0] == voxel_coords.shape[0]:
        print(f"   ✓ 点数匹配")
    else:
        print(f"   ✗ 点数不匹配: {decompressed.shape[0]} vs {voxel_coords.shape[0]}")
        return False
    
    print("\n" + "="*70)
    print("✓ 真实数据测试通过")
    print("="*70)
    return True


def test_gpcc_codec_path():
    """测试 GPCC 编解码器路径是否正确"""
    print("\n" + "="*70)
    print("检查 GPCC 编解码器配置")
    print("="*70)
    
    # 读取 gpcc_utils.py 中的默认路径
    from utils.gpcc_utils import compress_gpcc, decompress_gpcc
    import inspect
    
    # 获取函数源码
    compress_source = inspect.getsource(compress_gpcc)
    
    # 提取默认路径
    import re
    match = re.search(r'gpcc_codec_path\s*=\s*r?"([^"]+)"', compress_source)
    if match:
        default_path = match.group(1)
        print(f"\n默认 GPCC 路径: {default_path}")
        
        # 检查文件是否存在
        if os.path.exists(default_path):
            print(f"✓ GPCC 编解码器文件存在")
            
            # 检查是否可执行
            if os.access(default_path, os.X_OK) or default_path.endswith('.exe'):
                print(f"✓ GPCC 编解码器可执行")
            else:
                print(f"⚠ GPCC 编解码器可能没有执行权限")
        else:
            print(f"✗ GPCC 编解码器文件不存在")
            print(f"\n请检查以下内容:")
            print(f"1. 是否已安装 GPCC 编解码器")
            print(f"2. 路径是否正确")
            print(f"3. 修改 utils/gpcc_utils.py 中的 gpcc_codec_path 参数")
            return False
    else:
        print(f"⚠ 无法从源码中提取默认路径")
    
    print("\n" + "="*70)
    return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("GPCC 功能完整测试")
    print("="*70)
    
    # 测试 1: 检查编解码器路径
    print("\n【测试 1】检查 GPCC 编解码器配置")
    if not test_gpcc_codec_path():
        print("\n⚠ 编解码器配置有问题，但继续测试...")
    
    # 测试 2: 基本功能测试
    print("\n【测试 2】基本压缩/解压缩功能")
    test1_passed = test_gpcc_basic()
    
    # 测试 3: 真实数据测试
    if test1_passed:
        print("\n【测试 3】真实八叉树数据测试")
        test2_passed = test_gpcc_with_real_data()
    else:
        test2_passed = False
        print("\n【测试 3】跳过（基本测试未通过）")
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"基本功能测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"真实数据测试: {'✓ 通过' if test2_passed else '✗ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n✓ 所有测试通过！GPCC 功能正常")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查 GPCC 配置")
        print("\n常见问题:")
        print("1. GPCC 编解码器路径不正确")
        print("   → 修改 utils/gpcc_utils.py 中的 gpcc_codec_path")
        print("2. GPCC 编解码器未安装")
        print("   → 从 MPEG 官网下载并编译 TMC3")
        print("3. 临时目录权限问题")
        print("   → 检查系统临时目录的读写权限")
        return 1


if __name__ == "__main__":
    sys.exit(main())
