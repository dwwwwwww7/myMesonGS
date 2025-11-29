# 解压缩PLY文件生成指南

## 功能说明

现在渲染时会自动从NPZ文件重构并保存PLY文件，方便：
- 验证解压缩质量
- 可视化压缩后的模型
- 与原始模型对比
- 使用其他工具查看

## 自动生成

### 使用render.py

当使用 `--dec_npz` 参数时，会自动生成解压缩的PLY文件：

```bash
python render.py \
    -s D:/3DGS_seq/truck \
    -m D:/3DGS_seq/result_mesongs/output/truck_config3 \
    --iteration 20 \
    --dec_npz \
    --skip_test
```

**输出**:
```
【加载压缩模型】
  目录: .../bins
  ...
【加载完成】所有属性已从 RAHT 特征中提取

【保存解压缩的PLY文件】
  保存路径: .../point_cloud/iteration_20/decompressed.ply
  点数: 343376
  特征维度: DC=3, Rest=45
  ✓ PLY文件已保存
  文件大小: 156.78 MB
```

### 文件位置

```
<MODEL_PATH>/
└── point_cloud/
    └── iteration_20/
        ├── pc_npz/              # 压缩文件
        │   └── bins/
        │       ├── oct.npz
        │       ├── orgb.npz
        │       └── t.npz
        └── decompressed.ply     # 解压缩的PLY ← 新生成
```

## 文件内容

### PLY文件包含的属性

```
vertex (N个点)
├── x, y, z              # 位置（从八叉树解码）
├── nx, ny, nz           # 法向量（全0）
├── f_dc_0, f_dc_1, f_dc_2                    # DC球谐系数
├── f_rest_0 ... f_rest_44                    # 高阶球谐系数
├── opacity                                    # 不透明度
├── scale_0, scale_1, scale_2                 # 缩放
└── rot_0, rot_1, rot_2, rot_3                # 旋转（四元数）
```

### 数据来源

| 属性 | 来源 | 说明 |
|------|------|------|
| xyz | oct.npz | 八叉树解码 |
| f_dc | orgb.npz | RAHT逆变换 |
| f_rest | orgb.npz | RAHT逆变换 |
| opacity | orgb.npz | RAHT逆变换 |
| scale | orgb.npz | RAHT逆变换 |
| rotation | 从euler转换 | RAHT逆变换 → euler → quaternion |

## 使用场景

### 1. 质量验证

对比原始PLY和解压缩PLY：

```bash
# 原始模型
original.ply  # 训练前的模型

# 压缩后
decompressed.ply  # 从NPZ重构的模型

# 对比
# - 点数是否一致
# - 属性范围是否合理
# - 视觉质量是否可接受
```

### 2. 可视化

使用PLY查看器查看：

```bash
# CloudCompare
cloudcompare decompressed.ply

# MeshLab
meshlab decompressed.ply

# Python
import open3d as o3d
pcd = o3d.io.read_point_cloud("decompressed.ply")
o3d.visualization.draw_geometries([pcd])
```

### 3. 进一步处理

```python
from plyfile import PlyData

# 读取PLY
plydata = PlyData.read("decompressed.ply")

# 提取属性
xyz = np.stack([
    plydata.elements[0]["x"],
    plydata.elements[0]["y"],
    plydata.elements[0]["z"]
], axis=1)

opacity = plydata.elements[0]["opacity"]
# ... 其他属性
```

### 4. 与原始模型对比

```python
import numpy as np
from plyfile import PlyData

# 加载两个模型
original = PlyData.read("original.ply")
decompressed = PlyData.read("decompressed.ply")

# 对比点数
print(f"原始点数: {len(original.elements[0])}")
print(f"解压缩点数: {len(decompressed.elements[0])}")

# 对比位置
xyz_orig = np.stack([original.elements[0]["x"], ...], axis=1)
xyz_decomp = np.stack([decompressed.elements[0]["x"], ...], axis=1)

# 计算位置误差
pos_error = np.abs(xyz_orig - xyz_decomp).mean()
print(f"平均位置误差: {pos_error:.6f}")

# 对比其他属性
# ...
```

## 手动生成

如果只想生成PLY而不渲染：

```python
# create_ply_from_npz.py
import torch
from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams

# 配置
model_path = "D:/3DGS_seq/result_mesongs/output/truck_config3"
iteration = 20

# 加载模型
gaussians = GaussianModel(sh_degree=3)
npz_path = f"{model_path}/point_cloud/iteration_{iteration}/pc_npz"
gaussians.load_npz(npz_path)

# 保存PLY
ply_path = f"{model_path}/point_cloud/iteration_{iteration}/decompressed.ply"
gaussians.save_decompressed_ply(ply_path)

print(f"PLY文件已保存: {ply_path}")
```

运行：
```bash
python create_ply_from_npz.py
```

## 文件大小对比

### 示例（343,376个点）

| 文件 | 大小 | 说明 |
|------|------|------|
| original.ply | ~180 MB | 原始训练模型 |
| bins.zip | ~10 MB | 压缩后（NPZ） |
| decompressed.ply | ~157 MB | 解压缩后 |

**观察**:
- 压缩比: ~18:1
- 解压缩后略小于原始（因为点数可能减少）
- 质量损失主要来自量化和八叉树

## 注意事项

### 1. Euler角转换

解压缩时需要将euler角转回四元数：

```python
# euler → quaternion
roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

cy = np.cos(yaw * 0.5)
sy = np.sin(yaw * 0.5)
cp = np.cos(pitch * 0.5)
sp = np.sin(pitch * 0.5)
cr = np.cos(roll * 0.5)
sr = np.sin(roll * 0.5)

w = cr * cp * cy + sr * sp * sy
x = sr * cp * cy - cr * sp * sy
y = cr * sp * cy + sr * cp * sy
z = cr * cp * sy - sr * sp * cy

rotation = np.stack([w, x, y, z], axis=-1)
```

### 2. 点数变化

由于八叉树编码可能合并点，解压缩后的点数可能少于原始：

```
原始: 520,000 个点
剪枝后: 343,376 个点
八叉树后: 343,376 个点
解压缩: 343,376 个点
```

### 3. 精度损失

解压缩后的值会有量化误差：

```python
# 原始值
opacity_original = 0.8765432

# 8-bit量化后
opacity_decompressed = 0.8750000  # 略有差异

# 误差
error = abs(opacity_original - opacity_decompressed)
# ≈ 0.0015 (可接受)
```

## 批量生成

生成多个迭代的PLY：

```bash
#!/bin/bash
# batch_generate_ply.sh

MODEL_PATH="D:/3DGS_seq/result_mesongs/output/truck_config3"
SCENE_PATH="D:/3DGS_seq/truck"

for ITER in 0 10 20; do
    echo "生成迭代 $ITER 的PLY..."
    python render.py \
        -s "$SCENE_PATH" \
        -m "$MODEL_PATH" \
        --iteration $ITER \
        --dec_npz \
        --skip_train \
        --skip_test \
        --quick
    echo "完成！"
done

echo "所有PLY文件已生成："
ls -lh "$MODEL_PATH"/point_cloud/iteration_*/decompressed.ply
```

## 验证脚本

创建一个验证脚本：

```python
# verify_decompression.py
import numpy as np
from plyfile import PlyData

def verify_ply(ply_path):
    """验证PLY文件的完整性"""
    print(f"验证: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    
    n_points = len(vertex)
    print(f"  点数: {n_points}")
    
    # 检查必要的属性
    required_attrs = ['x', 'y', 'z', 'opacity', 'scale_0', 'rot_0', 'f_dc_0']
    for attr in required_attrs:
        if attr not in vertex.data.dtype.names:
            print(f"  ✗ 缺少属性: {attr}")
            return False
        else:
            data = vertex[attr]
            print(f"  ✓ {attr}: [{data.min():.4f}, {data.max():.4f}]")
    
    # 检查数值范围
    opacity = vertex['opacity']
    if opacity.min() < -10 or opacity.max() > 10:
        print(f"  ⚠ opacity范围异常: [{opacity.min()}, {opacity.max()}]")
    
    print(f"  ✓ 验证通过")
    return True

# 使用
verify_ply("decompressed.ply")
```

## 总结

现在渲染时会自动生成解压缩的PLY文件：

- ✅ **自动生成**: 使用 `--dec_npz` 时自动创建
- ✅ **完整属性**: 包含所有高斯属性
- ✅ **标准格式**: 可用任何PLY查看器打开
- ✅ **方便对比**: 与原始模型对比质量
- ✅ **易于分析**: 可用Python/C++等工具处理

文件保存在: `<MODEL_PATH>/point_cloud/iteration_<N>/decompressed.ply`

开始使用吧！🎨
