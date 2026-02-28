import torch
import numpy as np
from scene.gaussian_model import pack_bits, unpack_bits

print("=== 位打包功能测试 ===")

# 测试 1: 基本功能
print("\n测试 1: 基本功能")
arr = torch.tensor([[1, 300, 25, 65535]])  # 转换为 (1, 4) 形状
bit_widths = [3, 9, 6, 16] 

print(f"原始数据: {arr}")
print(f"位宽配置: {bit_widths}")

packed = pack_bits(arr, bit_widths)
print(f"打包结果: {len(packed)} bytes")
print(f"二进制形式: {' '.join(f'{b:08b}' for b in packed)}")
print(f"十进制形式: {list(packed)}")

unpacked = unpack_bits(packed, bit_widths, arr.shape[0])
print(f"解包结果: {unpacked}")
print(f"数据正确性: {'✓' if torch.allclose(arr.float(), torch.from_numpy(unpacked).float()) else '✗'}")

# 测试 2: 多维数据
print("\n测试 2: 多维数据")
data = torch.tensor([[1, 300, 25], [15, 200, 63], [7, 150, 31]])
bit_depths = [4, 9, 6]

print(f"原始数据形状: {data.shape}")
print(f"位宽配置: {bit_depths}")

# 展平数据进行打包
flat_data = data.flatten()
flat_bits = bit_depths * data.shape[0]

packed_multi = pack_bits(flat_data.reshape(-1, len(bit_depths)), bit_depths)
unpacked_multi = unpack_bits(packed_multi, bit_depths, data.shape[0])
unpacked_reshaped = unpacked_multi

print(f"打包大小: {len(packed_multi)} bytes")
print(f"二进制形式: {' '.join(f'{b:08b}' for b in packed_multi)}")
print(f"十进制形式: {list(packed_multi)}")
print(f"数据正确性: {'✓' if torch.allclose(data.float(), torch.from_numpy(unpacked_reshaped).float()) else '✗'}")


print("\n=== 测试完成 ===")