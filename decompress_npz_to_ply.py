#!/usr/bin/env python3
"""
从 NPZ 解压缩生成 PLY
使用 GaussianModel 的方法确保一致性
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel


def decompress_to_ply(bins_dir):
    """
    从 bins 目录的 NPZ 文件解压缩生成 PLY
    
    Args:
        bins_dir: bins 目录路径
    """
    print("="*70)
    print("从 NPZ 解压缩生成 PLY")
    print("="*70)
    print(f"Bins 目录: {bins_dir}\n")
    
    # 输出路径
    pc_npz_dir = os.path.dirname(bins_dir)
    output_path = os.path.join(pc_npz_dir, "point_cloud.ply")
    
    # 创建高斯模型
    print("创建高斯模型...")
    gaussians = GaussianModel(sh_degree=3, depth=20)
    
    # 加载 NPZ
    print("加载并解压缩 NPZ 文件...\n")
    try:
        # 调用 GaussianModel 的 load_npz 方法
        # 这个方法应该在 gaussian_model.py 中定义
        gaussians.load_npz(bins_dir)
        
    except AttributeError as e:
        print(f"❌ 错误: GaussianModel 没有 load_npz 方法")
        print(f"   {e}")
        print("\n请检查 scene/gaussian_model.py 中是否有 load_npz 方法")
        return False
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存 PLY
    print("\n保存 PLY 文件...")
    try:
        gaussians.save_decompressed_ply(output_path)
        
    except AttributeError as e:
        print(f"❌ 错误: GaussianModel 没有 save_decompressed_ply 方法")
        print(f"   {e}")
        print("\n尝试使用 save_ply 方法...")
        
        try:
            gaussians.save_ply(output_path)
        except Exception as e2:
            print(f"❌ save_ply 也失败: {e2}")
            return False
            
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ 成功!")
    print(f"PLY 文件: {output_path}")
    print("="*70)
    
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python decompress_npz_to_ply.py <bins目录路径>")
        print()
        print("示例:")
        print("  python decompress_npz_to_ply.py \"E:\\3dgs data\\my_RAHT_results\\output_gpcc_zsavenpz_bitpacknew5\\train_config4\\point_cloud\\iteration_0\\pc_npz\\bins\"")
        print()
        print("输出:")
        print("  <pc_npz目录>/point_cloud.ply")
        sys.exit(1)
    
    bins_dir = sys.argv[1]
    
    if not os.path.exists(bins_dir):
        print(f"❌ 错误: 目录不存在: {bins_dir}")
        sys.exit(1)
    
    success = decompress_to_ply(bins_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
