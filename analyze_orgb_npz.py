#!/usr/bin/env python3
"""
快速查看 orgb.npz 文件内容和数据结构的脚本，压缩保存的方式
重点显示文件大小组成和数据结构，并可通过--save-txt指令保存终端输出为TXT格式
使用案例：
python analyze_orgb_npz.py bins/orgb.npz --save-txt

"""

import sys
import os
import json
import numpy as np
import zipfile
from collections import defaultdict
from datetime import datetime
from io import StringIO

class OutputCapture:
    """捕获终端输出的类"""
    def __init__(self):
        self.output = StringIO()
        self.original_stdout = sys.stdout
    
    def start_capture(self):
        sys.stdout = self
    
    def stop_capture(self):
        sys.stdout = self.original_stdout
        return self.output.getvalue()
    
    def write(self, text):
        self.original_stdout.write(text)  # 同时输出到终端
        self.output.write(text)  # 捕获到内存
    
    def flush(self):
        self.original_stdout.flush()

def detect_npz_compression(npz_path):
    """检测NPZ文件的压缩格式"""
    try:
        # NPZ文件实际上是ZIP文件
        with zipfile.ZipFile(npz_path, 'r') as zf:
            # 检查ZIP文件中的压缩方法
            compression_methods = set()
            for info in zf.infolist():
                compression_methods.add(info.compress_type)
            
            # 判断压缩格式
            if zipfile.ZIP_DEFLATED in compression_methods:
                return "numpy.savez_compressed (ZIP_DEFLATED)"
            elif zipfile.ZIP_STORED in compression_methods:
                return "numpy.savez (ZIP_STORED, 无压缩)"
            else:
                return f"未知压缩方法: {compression_methods}"
    except:
        return "无法检测压缩格式"

def analyze_npz_structure(npz_path, save_txt=False):
    """分析NPZ文件的数据结构和大小组成"""
    
    # 创建输出捕获器
    output_capture = OutputCapture()
    if save_txt:
        output_capture.start_capture()
    
    print("="*80)
    print("orgb.npz 文件结构分析")
    print("="*80)
    print(f"文件路径: {npz_path}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(npz_path):
        print(f"[ERROR] 文件不存在")
        if save_txt:
            output_capture.stop_capture()
        return None
    
    total_file_size = os.path.getsize(npz_path)
    print(f"文件总大小: {total_file_size:,} bytes ({total_file_size/1024:.2f} KB, {total_file_size/1024/1024:.2f} MB)")
    
    # 检测压缩格式
    compression_format = detect_npz_compression(npz_path)
    print(f"NPZ压缩格式: {compression_format}")
    
    try:
        data = np.load(npz_path)
        
        print(f"\n" + "="*60)
        print("【数据结构概览】")
        print("="*60)
        print(f"包含数组数量: {len(data.keys())}")
        print(f"数组键名: {sorted(list(data.keys()))}")
        
        # 分析存储格式
        is_packed = 'packed' in data and len(data['packed']) > 0 and data['packed'][0] == 1
        is_grouped = any(key.startswith('i_') and key.endswith('bit') for key in data.keys())
        is_unified = 'i' in data and not is_grouped and not is_packed
        
        storage_format = "unknown"
        if is_packed:
            storage_format = "bit_packed"
            print(f"  📦 位打包格式 (Bit-packed format)")
        elif is_grouped:
            storage_format = "grouped"
            print(f"  📊 分组格式 (Grouped format)")
        elif is_unified:
            storage_format = "unified"
            print(f"  📄 统一格式 (Unified format)")
        else:
            print(f"  ❓ 未知格式")
        

        
        print(f"\n" + "="*60)
        print("【文件大小组成分析】")
        print("="*60)
        
        total_uncompressed_size = 0
        arrays_info = []
        
        for key in sorted(data.keys()):
            arr = data[key]
            uncompressed_size = arr.nbytes
            total_uncompressed_size += uncompressed_size
            
            arrays_info.append({
                'key': key,
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'elements': arr.size,
                'uncompressed_bytes': uncompressed_size,
                'array': arr
            })
        
        # 显示每个数组的详细信息
        print(f"{'数组名':<15} {'形状':<20} {'类型':<10} {'元素数':<12} {'未压缩大小':<15} {'占比':<8}")
        print("-" * 90)
        
        for info in arrays_info:
            percentage = (info['uncompressed_bytes'] / total_uncompressed_size) * 100
            size_str = f"{info['uncompressed_bytes']:,} B"
            if info['uncompressed_bytes'] >= 1024:
                size_str += f" ({info['uncompressed_bytes']/1024:.1f} KB)"
            
            print(f"{info['key']:<15} {str(info['shape']):<20} {info['dtype']:<10} {info['elements']:<12,} {size_str:<15} {percentage:>6.1f}%")
        
        print("-" * 90)
        print(f"{'总计':<15} {'':<20} {'':<10} {sum(info['elements'] for info in arrays_info):<12,} {total_uncompressed_size:,} B ({total_uncompressed_size/1024:.1f} KB) {'100.0%':>6}")
        
        # 压缩效率
        compression_ratio = total_file_size / total_uncompressed_size
        compression_percentage = (1 - compression_ratio) * 100
        
        print(f"\n压缩效率:")
        print(f"  未压缩大小: {total_uncompressed_size:,} bytes ({total_uncompressed_size/1024:.2f} KB)")
        print(f"  压缩后大小: {total_file_size:,} bytes ({total_file_size/1024:.2f} KB)")
        print(f"  压缩比: {compression_ratio:.3f} ({compression_percentage:.1f}% 减少)")
        
        print(f"\n" + "="*60)
        print("【RAHT数据结构分析】")
        print("="*60)
        
        # DC系数分析
        if 'f' in data:
            dc_coeff = data['f']
            print(f"DC系数 (f):")
            print(f"  形状: {dc_coeff.shape}")
            print(f"  含义: RAHT变换的DC分量 (55维特征的频域表示)")
            if len(dc_coeff.shape) == 1 and dc_coeff.shape[0] == 55:
                print(f"  结构: opacity(1) + euler(3) + f_dc(3) + f_rest(45) + scale(3)")
                print(f"  数值范围: [{dc_coeff.min():.4f}, {dc_coeff.max():.4f}]")
                
                # 显示55维特征的分组
                feature_groups = [
                    ("opacity", 0, 1, "透明度"),
                    ("euler", 1, 4, "欧拉角(旋转)"),
                    ("f_dc", 4, 7, "SH系数0阶(基础颜色)"),
                    ("f_rest", 7, 52, "SH系数1-3阶(颜色细节)"),
                    ("scale", 52, 55, "缩放参数")
                ]
                
                print(f"  详细结构:")
                for name, start, end, desc in feature_groups:
                    values = dc_coeff[start:end]
                    print(f"    {name:8s} [{start:2d}:{end:2d}]: {desc:15s} 范围[{values.min():8.4f}, {values.max():8.4f}]")
        
        # AC系数分析
        print(f"\nAC系数 (频域细节):")
        
        if is_grouped:
            print(f"  存储方式: 按量化位数分组存储")
            
            # 统计各个位宽组
            bit_groups = defaultdict(dict)
            total_ac_size = 0
            
            for key in data.keys():
                if key.startswith('i_') and key.endswith('bit'):
                    bit = int(key.split('_')[1].replace('bit', ''))
                    group_data = data[key]
                    dims_key = f'dims_{bit}bit'
                    
                    if dims_key in data:
                        dims = data[dims_key]
                        bit_groups[bit] = {
                            'data': group_data,
                            'dims': dims,
                            'shape': group_data.shape,
                            'size_bytes': group_data.nbytes,
                            'nonzero_ratio': np.count_nonzero(group_data) / group_data.size if group_data.size > 0 else 0
                        }
                        total_ac_size += group_data.nbytes
            
            print(f"  分组详情:")
            print(f"    {'位宽':<6} {'维度数':<8} {'数据形状':<20} {'大小':<15} {'稀疏度':<10} {'维度索引示例'}")
            print(f"    {'-'*6} {'-'*8} {'-'*20} {'-'*15} {'-'*10} {'-'*20}")
            
            for bit in sorted(bit_groups.keys()):
                info = bit_groups[bit]
                sparsity = (1 - info['nonzero_ratio']) * 100
                size_str = f"{info['size_bytes']:,} B"
                if info['size_bytes'] >= 1024:
                    size_str += f" ({info['size_bytes']/1024:.1f}KB)"
                
                dims_preview = str(info['dims'][:5].tolist()) if len(info['dims']) > 5 else str(info['dims'].tolist())
                if len(info['dims']) > 5:
                    dims_preview = dims_preview[:-1] + ",...]"
                
                print(f"    {bit:>4d}   {len(info['dims']):>6d}   {str(info['shape']):<20} {size_str:<15} {sparsity:>6.1f}%   {dims_preview}")
            
            print(f"  AC系数总大小: {total_ac_size:,} bytes ({total_ac_size/1024:.2f} KB)")
            
            # 分析维度分配
            print(f"\n  维度分配策略:")
            all_dims_used = set()
            for bit in sorted(bit_groups.keys()):
                dims = bit_groups[bit]['dims']
                all_dims_used.update(dims)
                
                # 推断维度含义
                dim_meanings = []
                for dim in dims[:3]:  # 只显示前3个维度的含义
                    if dim == 0:
                        dim_meanings.append("opacity")
                    elif 1 <= dim <= 3:
                        dim_meanings.append(f"euler_{dim-1}")
                    elif 4 <= dim <= 6:
                        dim_meanings.append(f"f_dc_{dim-4}")
                    elif 7 <= dim <= 51:
                        dim_meanings.append(f"f_rest_{dim-7}")
                    elif 52 <= dim <= 54:
                        dim_meanings.append(f"scale_{dim-52}")
                    else:
                        dim_meanings.append(f"unknown_{dim}")
                
                meanings_str = ", ".join(dim_meanings)
                if len(dims) > 3:
                    meanings_str += ", ..."
                
                print(f"    {bit:2d}-bit: {meanings_str}")
            
            print(f"  总维度覆盖: {len(all_dims_used)}/55 ({'完整' if len(all_dims_used) == 55 else '不完整'})")
        
        elif is_unified:
            ac_data = data['i']
            print(f"  存储方式: 统一存储")
            print(f"  数据形状: {ac_data.shape}")
            print(f"  数据大小: {ac_data.nbytes:,} bytes ({ac_data.nbytes/1024:.2f} KB)")
            
            if len(ac_data.shape) == 1:
                total_elements = ac_data.shape[0]
                if total_elements % 55 == 0:
                    n_points = total_elements // 55
                    print(f"  推断结构: {n_points:,} 个AC点 × 55 维特征")
                else:
                    print(f"  注意: 总元素数 {total_elements:,} 不能被55整除")
            
            nonzero_ratio = np.count_nonzero(ac_data) / ac_data.size if ac_data.size > 0 else 0
            sparsity = (1 - nonzero_ratio) * 100
            print(f"  稀疏度: {sparsity:.2f}% (非零元素: {nonzero_ratio*100:.2f}%)")
        
        elif is_packed:
            print(f"  存储方式: 位级打包存储")
            if 'i' in data:
                bitstream = data['i']
                print(f"  位流大小: {len(bitstream):,} bytes")
                print(f"  总位数: {len(bitstream) * 8:,} bits")
        
        # 其他辅助数据
        print(f"\n辅助数据:")
        for key in sorted(data.keys()):
            if key not in ['f', 'i'] and not (key.startswith('i_') and key.endswith('bit')) and not key.startswith('dims_'):
                arr = data[key]
                print(f"  {key}: {arr.shape} {arr.dtype} ({arr.nbytes} bytes)")
                if arr.size <= 10:
                    print(f"    数据: {arr}")
        
        data.close()
        
        # 保存TXT文件
        if save_txt:
            captured_output = output_capture.stop_capture()
            txt_path = npz_path.replace('.npz', '_analysis.txt')
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(captured_output)
                print(f"\n✓ 分析结果已保存到: {txt_path}")
            except Exception as e:
                print(f"\n✗ 保存TXT文件失败: {e}")
        
        print(f"\n" + "="*80)
        print("分析完成")
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        
        if save_txt:
            output_capture.stop_capture()

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_orgb_npz.py <orgb.npz路径> [--save-txt]")
        print()
        print("参数:")
        print("  orgb.npz路径    要分析的NPZ文件路径")
        print("  --save-txt      保存终端输出为TXT格式")
        print()
        print("示例:")
        print("  python analyze_orgb_npz.py bins/orgb.npz")
        print("  python analyze_orgb_npz.py bins/orgb.npz --save-txt")
        print("  python analyze_orgb_npz.py \"E:/path/to/orgb.npz\" --save-txt")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    save_txt = '--save-txt' in sys.argv
    
    analyze_npz_structure(npz_path, save_txt=save_txt)

if __name__ == "__main__":
    main()



def analyze_first_n_gaussians(npz_path, n=100):
    """
    分析前N个高斯点的详细属性值
    
    Args:
        npz_path: NPZ文件路径
        n: 要分析的高斯点数量（默认100）
    """
    print("\n" + "="*70)
    print(f"【前{n}个高斯点详细分析】")
    print("="*70)
    
    data = np.load(npz_path)
    
    # 获取DC系数
    dc_coeff = data['f']
    print(f"\nDC系数 (1个点):")
    print(f"  形状: {dc_coeff.shape}")
    print(f"  值: {dc_coeff}")
    
    # 检测存储格式
    is_packed = 'packed' in data and data['packed'][0] == 1
    is_grouped = 'grouped' in data and data['grouped'][0] == 1
    
    # 计算高斯点数量
    if 'i' in data:
        if is_packed and 'bit_config' in data:
            bit_config = data['bit_config'].tolist()
            total_bits_per_point = sum(bit_config)
            bitstream = bytes(data["i"])
            total_bits = len(bitstream) * 8
            n_points = total_bits // total_bits_per_point
            print(f"\n高斯点数量: {n_points:,}")
            print(f"  (从位流大小推算: {len(bitstream):,} bytes × 8 / {total_bits_per_point} bits/point)")
        else:
            ac_data = data['i']
            if len(ac_data.shape) == 2:
                n_points = ac_data.shape[0]
                print(f"\n高斯点数量: {n_points:,}")
            elif len(ac_data.shape) == 1 and ac_data.shape[0] % 55 == 0:
                n_points = ac_data.shape[0] // 55
                print(f"\n高斯点数量: {n_points:,}")
                print(f"  (从数组大小推算: {ac_data.shape[0]:,} / 55)")
    
    if is_packed:
        print(f"\n检测到位打包格式")
        bitstream = bytes(data["i"])
        bit_config = data['bit_config'].tolist()
        
        print(f"  位宽配置: {bit_config}")
        print(f"  总位数/点: {sum(bit_config)} bits")
        
        # 解包前N个高斯点
        print(f"\n正在解包前{min(n, n_points)}个高斯点...")
        actual_n = min(n, n_points)
        first_n_data = np.zeros((actual_n, 55), dtype=np.float32)
        
        bit_pos = 0
        for i in range(actual_n):
            for c in range(55):
                bits = bit_config[c]
                value = 0
                
                # 读取位流
                for b in range(bits):
                    byte_idx = bit_pos // 8
                    bit_idx = bit_pos % 8
                    if byte_idx < len(bitstream) and bitstream[byte_idx] & (1 << bit_idx):
                        value |= (1 << b)
                    bit_pos += 1
                
                first_n_data[i, c] = value
        
    elif is_grouped:
        print(f"\n检测到分组存储格式")
        print(f"  (暂不支持详细解析分组格式)")
        return
    else:
        print(f"\n未知格式或统一存储格式")
        if 'i' in data:
            ac_data = data['i']
            if len(ac_data.shape) == 2:
                actual_n = min(n, ac_data.shape[0])
                first_n_data = ac_data[:actual_n]
                print(f"  提取前{actual_n}个点")
            else:
                print(f"  无法解析数据格式")
                return
        else:
            return
    
    # 显示前N个高斯点的统计
    print(f"\n前{actual_n}个高斯点的属性统计:")
    print("-"*70)
    
    feature_groups = [
        ("opacity", 0, 1, "透明度"),
        ("euler", 1, 4, "欧拉角(旋转)"),
        ("f_dc", 4, 7, "SH系数0阶(基础颜色)"),
        ("f_rest_0", 7, 22, "SH系数1阶(15维)"),
        ("f_rest_1", 22, 37, "SH系数2阶(15维)"),
        ("f_rest_2", 37, 52, "SH系数3阶(15维)"),
        ("scale", 52, 55, "缩放参数"),
    ]
    
    for name, start, end, desc in feature_groups:
        values = first_n_data[:, start:end]
        print(f"\n{name:12s} [{start:2d}:{end:2d}]: {desc}")
        print(f"  形状: {values.shape}")
        print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  均值: {values.mean():.4f}")
        print(f"  标准差: {values.std():.4f}")
        
        # 显示前5个点的值
        if end - start <= 3:
            print(f"  前5个点:")
            for i in range(min(5, actual_n)):
                vals = values[i]
                vals_str = " ".join([f"{v:8.2f}" for v in vals])
                print(f"    点{i}: {vals_str}")
    
    # 保存详细数据到文件
    output_dir = os.path.dirname(npz_path)
    if not output_dir:
        output_dir = '.'
    output_file = os.path.join(output_dir, f'first_{actual_n}_gaussians.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"前{actual_n}个高斯点的详细属性值\n")
        f.write(f"文件: {npz_path}\n")
        f.write("="*70 + "\n\n")
        
        f.write("DC系数:\n")
        f.write(f"{dc_coeff}\n\n")
        
        f.write(f"AC系数 (前{actual_n}个高斯点):\n")
        f.write("-"*70 + "\n")
        
        # 写入表头
        f.write(f"{'点ID':>6s} | ")
        for name, start, end, _ in feature_groups:
            for i in range(start, end):
                f.write(f"{name}_{i-start:02d} ")
        f.write("\n")
        f.write("-"*200 + "\n")
        
        # 写入数据
        for i in range(actual_n):
            f.write(f"{i:6d} | ")
            for j in range(55):
                f.write(f"{first_n_data[i, j]:8.2f} ")
            f.write("\n")
    
    print(f"\n✓ 详细数据已保存到: {output_file}")
    data.close()


def compare_two_npz(npz_path1, npz_path2, n=100):
    """
    对比两个NPZ文件的差异
    
    Args:
        npz_path1: 第一个NPZ文件路径
        npz_path2: 第二个NPZ文件路径
        n: 对比前N个高斯点
    """
    print("\n" + "="*70)
    print(f"【对比两个NPZ文件】")
    print("="*70)
    print(f"文件1: {npz_path1}")
    print(f"文件2: {npz_path2}")
    
    data1 = np.load(npz_path1)
    data2 = np.load(npz_path2)
    
    # 对比DC系数
    dc1 = data1['f']
    dc2 = data2['f']
    
    print(f"\nDC系数对比:")
    print(f"  文件1: min={dc1.min():.6f}, max={dc1.max():.6f}, mean={dc1.mean():.6f}")
    print(f"  文件2: min={dc2.min():.6f}, max={dc2.max():.6f}, mean={dc2.mean():.6f}")
    
    dc_diff = dc1 - dc2
    print(f"  差异: min={dc_diff.min():.6f}, max={dc_diff.max():.6f}")
    print(f"  最大绝对差异: {np.abs(dc_diff).max():.6f}")
    print(f"  平均绝对差异: {np.abs(dc_diff).mean():.6f}")
    
    # 显示差异最大的维度
    max_diff_idx = np.abs(dc_diff).argmax()
    print(f"  差异最大的维度: {max_diff_idx} (差异={dc_diff[max_diff_idx]:.6f})")
    
    # 对比AC系数
    if 'i' in data1 and 'i' in data2:
        size1 = len(data1['i'])
        size2 = len(data2['i'])
        print(f"\nAC系数大小对比:")
        print(f"  文件1: {size1:,} bytes")
        print(f"  文件2: {size2:,} bytes")
        if size1 != size2:
            print(f"  差异: {size2 - size1:,} bytes ({(size2-size1)/size1*100:.2f}%)")
        else:
            print(f"  大小相同")
            
            # 如果大小相同，对比内容
            ac1 = data1['i']
            ac2 = data2['i']
            if np.array_equal(ac1, ac2):
                print(f"  内容完全相同")
            else:
                diff_count = np.sum(ac1 != ac2)
                print(f"  内容不同: {diff_count:,} / {len(ac1):,} 字节不同 ({diff_count/len(ac1)*100:.2f}%)")
    
    # 对比文件大小
    import os
    file1_size = os.path.getsize(npz_path1)
    file2_size = os.path.getsize(npz_path2)
    print(f"\n文件大小对比:")
    print(f"  文件1: {file1_size:,} bytes ({file1_size/1024:.2f} KB)")
    print(f"  文件2: {file2_size:,} bytes ({file2_size/1024:.2f} KB)")
    if file1_size != file2_size:
        print(f"  差异: {file2_size - file1_size:,} bytes ({(file2_size-file1_size)/file1_size*100:.2f}%)")
    else:
        print(f"  大小相同")
    
    data1.close()
    data2.close()


def main_extended():
    """扩展的主函数，支持更多功能"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 基础分析:     python analyze_orgb_npz.py <npz文件路径> [--save-txt]")
        print("  2. 详细分析:     python analyze_orgb_npz.py <npz文件路径> --detail [高斯点数量]")
        print("  3. 对比两个文件: python analyze_orgb_npz.py <文件1> <文件2> --compare [高斯点数量]")
        print("\n示例:")
        print("  python analyze_orgb_npz.py output/truck/bins/orgb.npz")
        print("  python analyze_orgb_npz.py output/truck/bins/orgb.npz --detail 200")
        print("  python analyze_orgb_npz.py file1.npz file2.npz --compare 100")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    # 检查是否是对比模式
    if '--compare' in sys.argv and len(sys.argv) >= 3:
        npz_path2 = sys.argv[2]
        n = 100
        if len(sys.argv) >= 5:
            try:
                n = int(sys.argv[4])
            except:
                pass
        
        print("="*70)
        print("文件1分析:")
        print("="*70)
        analyze_npz_structure(npz_path, save_txt=False)
        
        print("\n" + "="*70)
        print("文件2分析:")
        print("="*70)
        analyze_npz_structure(npz_path2, save_txt=False)
        
        compare_two_npz(npz_path, npz_path2, n)
        
    elif '--detail' in sys.argv:
        # 详细分析模式
        n = 100
        if len(sys.argv) >= 4:
            try:
                n = int(sys.argv[3])
            except:
                pass
        
        analyze_npz_structure(npz_path, save_txt=False)
        analyze_first_n_gaussians(npz_path, n)
    else:
        # 基础分析模式
        save_txt = '--save-txt' in sys.argv
        analyze_npz_structure(npz_path, save_txt=save_txt)


# 如果直接运行此脚本，使用扩展的主函数
if __name__ == "__main__" and len(sys.argv) > 1 and ('--detail' in sys.argv or '--compare' in sys.argv):
    main_extended()
