# Scale (缩放) 3维处理流程详解

## 概述

Scale 是 3D 高斯点的缩放参数，控制每个高斯椭球在 x, y, z 三个方向上的大小。

**维度**: 3 (scale_x, scale_y, scale_z)

---

## 完整处理流程

### **阶段1：原始数据**

```python
# 在 GaussianModel 中
self._scaling  # 原始 scale 参数 [N, 3]
```

**特点**：
- 存储为对数空间：`log(scale)`
- 实际使用时需要 `exp()` 转换

---

### **阶段2：剪枝后**

```
原始点数: 572,303
  ↓ 剪枝 40%
剩余点数: 343,377
  ↓
Scale 也相应减少到 343,377 × 3
```

**Scale 不参与剪枝决策**，只是随着点被删除而减少。

---

### **阶段3：八叉树编码**

```python
# Scale 不参与八叉树编码
# 八叉树只编码位置信息
```

**Scale 保持原样**，不进行 Morton 排序或 RAHT 变换。

---

### **阶段4：渲染时的处理**

#### 4.1 获取原始 Scale

```python
scales = pc.get_ori_scaling  # [N, 3]
# 这是对数空间的值：log(scale)
```

#### 4.2 分块量化

```python
# 使用量化器 [52*n_block : 55*n_block]
# 对于 truck (n_block=57):
#   量化器索引: [2964 : 3135]

scalesq = torch.zeros_like(scales).cuda()
split_scale = split_length(scales.shape[0], pc.n_block)

for i in range(3):  # x, y, z 三个维度
    # 每个维度使用 n_block 个量化器
    scalesq[:, i] = seg_quant_ave(
        scales[:, i], 
        split_scale, 
        pc.qas[qa_cnt : qa_cnt + pc.n_block]
    )
    qa_cnt += pc.n_block
```

**量化过程**：
1. 将点云分成 57 块
2. 每块的 scale_x 使用独立的量化器
3. 每块的 scale_y 使用独立的量化器
4. 每块的 scale_z 使用独立的量化器
5. 总共使用 3 × 57 = 171 个量化器

#### 4.3 转换到线性空间

```python
scaling = torch.exp(scalesq)  # [N, 3]
# 从对数空间转换到线性空间
```

#### 4.4 计算协方差矩阵

```python
eulers = raht_features[:, 1:4]  # 从 RAHT 提取欧拉角
cov3D_precomp = pc.covariance_activation_for_euler(
    scaling,  # 线性空间的 scale
    1.0,      # scaling_modifier
    eulers    # 旋转（欧拉角）
)
```

**协方差矩阵计算**：
```
Cov = R × S × S^T × R^T

其中：
- R: 旋转矩阵（从欧拉角转换）
- S: 缩放矩阵（对角矩阵，对角线为 scaling）
```

#### 4.5 用于渲染

```python
rendered_image, radii = rasterizer(
    means3D = means3D,
    means2D = means2D,
    opacities = opacity,
    colors_precomp = colors_precomp,
    scales = None,      # 不直接传 scale
    rotations = None,   # 不直接传 rotation
    cov3D_precomp = cov3D_precomp  # 传预计算的协方差矩阵
)
```

---

### **阶段5：保存时的处理**

#### 5.1 获取原始 Scale

```python
scaling = self.get_ori_scaling.detach()  # [N, 3]
# 对数空间的值
```

#### 5.2 分块量化

```python
lc1 = scaling.shape[0]  # 343,377
split_scale = split_length(lc1, self.n_block)  # 分成 57 块

scaling_q = []
for i in range(3):  # x, y, z
    t1, trans1 = torch_vanilla_quant_ave(
        scaling[:, i], 
        split_scale, 
        self.qas[qa_cnt : qa_cnt + self.n_block]
    )
    scaling_q.append(t1)
    trans_array.extend(trans1)  # 保存量化参数
    qa_cnt += self.n_block

scaling_q = np.concatenate(scaling_q, axis=-1)  # [N, 3]
```

**量化参数**：
- 每个块每个维度有 2 个参数：scale, zero_point
- 总共：3 维 × 57 块 × 2 参数 = 342 个参数

#### 5.3 保存到文件

```python
np.savez_compressed(
    os.path.join(bin_dir, 'ct.npz'), 
    i=scaling_q.astype(np.uint8)
)
```

**文件内容**：
- `ct.npz`: 量化后的 scale 值（8位整数）
- 形状：[343377, 3]
- 大小：约 1 MB

---

### **阶段6：加载时的处理**

#### 6.1 从文件加载

```python
q_scale_i = torch.tensor(
    np.load(os.path.join(bin_dir, 'ct.npz'))["i"].astype(np.float32), 
    dtype=torch.float, 
    device="cuda"
).reshape(3, -1).contiguous().transpose(0, 1)  # [N, 3]
```

#### 6.2 反量化

```python
de_scale = []
scale_len = q_scale_i.shape[0]
scale_split = split_length(scale_len, self.n_block)

for i in range(3):  # x, y, z
    scale_i = torch_vanilla_dequant_ave(
        q_scale_i[:, i], 
        scale_split, 
        trans_array[qa_cnt : qa_cnt + 2*self.n_block]
    )
    de_scale.append(scale_i.reshape(-1, 1))
    qa_cnt += 2*self.n_block

de_scale = torch.concat(de_scale, axis=-1).to("cuda")  # [N, 3]
```

#### 6.3 存储为模型参数

```python
self._scaling = nn.Parameter(de_scale.requires_grad_(False))
```

---

## 关键特点

### 1. **不参与 RAHT 变换**

```
RAHT 变换的特征（52维）:
  ✓ opacity (1)
  ✓ euler (3)
  ✓ f_dc (3)
  ✓ f_rest (45)

不参与 RAHT:
  ✗ scale (3)  ← 独立处理
  ✗ position (3)  ← 八叉树编码
```

**原因**：
- Scale 是局部属性，不需要空间相关性变换
- 直接量化更简单高效

### 2. **独立的分块量化**

```
量化器分配:
  [0 : 52*n_block]      → RAHT 特征 (52维)
  [52*n_block : 55*n_block]  → Scale (3维)
  
对于 truck (n_block=57):
  [0 : 2964]     → RAHT 特征
  [2964 : 3135]  → Scale
```

### 3. **对数空间存储**

```python
# 存储: log(scale)
_scaling = log([0.1, 0.2, 0.3])  # [-2.3, -1.6, -1.2]

# 使用: exp(log(scale)) = scale
scaling = exp(_scaling)  # [0.1, 0.2, 0.3]
```

**优点**：
- 数值稳定性更好
- 量化误差影响更小
- 避免负值问题

### 4. **与旋转结合使用**

```
Scale 单独无法确定高斯形状
需要与旋转（Euler角）结合：

Gaussian = exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))

其中协方差矩阵：
Σ = R * diag(scale_x², scale_y², scale_z²) * R^T
```

---

## 数据流图

```
原始 Scale (对数空间)
  [343377, 3]
  ↓
分块量化
  使用 171 个量化器 (3 × 57)
  ↓
量化后的 Scale (8位整数)
  [343377, 3]
  ↓
保存到 ct.npz
  约 1 MB
  ↓
加载时反量化
  ↓
恢复 Scale (对数空间)
  [343377, 3]
  ↓
渲染时 exp() 转换
  ↓
线性空间的 Scale
  [343377, 3]
  ↓
与 Euler 角结合
  ↓
计算协方差矩阵
  [343377, 3, 3]
  ↓
用于渲染
```

---

## 量化细节

### 分块策略

```python
# 将 343,377 个点分成 57 块
split_scale = split_length(343377, 57)

# 结果类似：
# [6025, 6025, 6025, ..., 6027]
# 每块约 6025 个点
```

### 量化公式

```python
# 量化（保存时）
q_value = round((value - min_val) / (max_val - min_val) * 255)

# 反量化（加载时）
value = q_value / 255 * (max_val - min_val) + min_val
```

### 量化参数

每个块每个维度：
- `scale`: (max_val - min_val) / 255
- `zero_point`: min_val 对应的量化值

---

## 与其他属性的对比

| 属性 | 维度 | 处理方式 | 量化器数量 |
|------|------|---------|-----------|
| Position | 3 | 八叉树编码 | 0 |
| Opacity | 1 | RAHT变换 | 1×n_block |
| Rotation | 3 (Euler) | RAHT变换 | 3×n_block |
| f_dc | 3 | RAHT变换 | 3×n_block |
| f_rest | 45 | RAHT变换 | 45×n_block |
| **Scale** | **3** | **直接量化** | **3×n_block** |

---

## 为什么 Scale 不用 RAHT？

### 1. **空间相关性弱**
- Scale 主要由局部几何决定
- 相邻点的 scale 可能差异很大
- RAHT 变换收益不明显

### 2. **数值范围稳定**
- 对数空间的 scale 范围较小
- 直接量化效果已经很好
- 不需要额外的变换

### 3. **计算效率**
- 直接量化更快
- 不需要 Morton 排序
- 不需要逆变换

### 4. **独立性**
- Scale 与颜色、不透明度无关
- 可以独立处理和优化

---

## 总结

**Scale 的处理特点**：
1. ✅ **独立于 RAHT**：不参与 RAHT 变换
2. ✅ **分块量化**：使用 3×n_block 个独立量化器
3. ✅ **对数空间**：存储和量化都在对数空间
4. ✅ **与旋转结合**：通过协方差矩阵影响渲染
5. ✅ **高效简洁**：处理流程简单，压缩效果好

**压缩效果**：
- 原始：343,377 × 3 × 4字节 = 4.1 MB
- 压缩后：343,377 × 3 × 1字节 = 1.0 MB
- **压缩率：4倍**
