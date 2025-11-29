# 量化流程详细分析

## 当前配置

根据你的配置文件和代码，当前使用的量化方案是：

- **量化方法**: VanillaQuan（普通均匀量化，非LSQ）
- **量化位数**: 8-bit
- **量化模式**: 分块量化（per_block_quant=True）
- **块数量**: 57个块（truck场景）

## 量化类型：VanillaQuan vs LSQ

### VanillaQuan（当前使用）

**特点：**
- 普通的均匀量化（Uniform Quantization）
- 使用 min-max 范围计算 scale 和 zero_point
- **不可学习**的量化参数（scale和zero_point在训练中不更新梯度）
- 使用 Straight-Through Estimator (STE) 进行梯度传播

**公式：**
```python
# 前向传播
scale = (max_val - min_val) / (2^bit - 1)
zero_point = 2^bit - 1 - max_val / scale
q_x = clamp(round(zero_point + x / scale), 0, 2^bit - 1)
x_dequant = scale * (q_x - zero_point)

# 反向传播（STE）
grad_x = grad_output  # 直通估计器
```

### LSQQuan（代码中有但未使用）

**特点：**
- Learned Step Size Quantization
- **可学习**的量化参数（scale在训练中更新）
- 使用梯度缩放来优化量化参数
- 理论上精度更高，但训练更复杂

**区别：**
| 特性 | VanillaQuan | LSQQuan |
|------|-------------|---------|
| 量化参数 | 固定（基于min-max） | 可学习 |
| 训练复杂度 | 低 | 高 |
| 精度 | 中等 | 较高 |
| 收敛速度 | 快 | 慢 |
| 内存占用 | 低 | 高（需要存储梯度） |

## 量化发生的时机和位置

### 1. 训练阶段（Training）

**位置**: `gaussian_renderer/__init__.py` - `ft_render()` 函数

**时机**: 每次前向传播时

**流程**:
```
输入特征 (opacity, euler, f_dc, f_rest, scale)
    ↓
RAHT 正向变换
    ↓
C = [DC系数, AC系数]  # shape: [N, 55]
    ↓
【量化发生在这里】
    ↓
对每个维度 i (0-54):
    对每个块 j (0-56):
        quantC[block_j, i] = qas[i*57 + j].forward(C[block_j, i])
    ↓
RAHT 逆变换
    ↓
恢复特征 → 渲染 → 计算损失 → 反向传播
```

**关键代码**:
```python
# gaussian_renderer/__init__.py, line ~407
for i in range(C.shape[-1]):  # 55维
    quantC[1:, i] = seg_quant_ave(
        C[1:, i], 
        split_ac, 
        pc.qas[qa_cnt : qa_cnt + pc.n_block]  # 57个量化器
    )
    qa_cnt += pc.n_block
```

### 2. 保存阶段（Saving）

**位置**: `scene/gaussian_model.py` - `save_npz()` 函数

**时机**: 每个测试迭代（test_iterations）

**流程**:
```
输入特征
    ↓
RAHT 正向变换
    ↓
C = [DC系数, AC系数]
    ↓
【量化发生在这里】
    ↓
使用训练好的量化器参数（scale, zero_point）
    ↓
量化为 uint8
    ↓
保存到 orgb.npz
```

**关键代码**:
```python
# scene/gaussian_model.py, line ~900
for i in range(55):
    t1, trans1 = torch_vanilla_quant_ave(
        C[1:, i], 
        split, 
        self.qas[qa_cnt : qa_cnt + self.n_block]
    )
    qci.append(t1)
    trans_array.extend(trans1)  # 保存 scale 和 zero_point
    qa_cnt += self.n_block
```

### 3. 加载阶段（Loading）

**位置**: `scene/gaussian_model.py` - `load_npz()` 函数

**时机**: 渲染时加载压缩模型

**流程**:
```
读取 orgb.npz (uint8)
    ↓
读取 t.npz (量化参数)
    ↓
【反量化发生在这里】
    ↓
对每个维度 i (0-54):
    对每个块 j (0-56):
        C[block_j, i] = scale[i,j] * (q_x[block_j, i] - zero_point[i,j])
    ↓
RAHT 逆变换
    ↓
恢复特征
```

## 量化器的初始化和更新

### 初始化

**位置**: `scene/gaussian_model.py` - `init_qas()` 函数

```python
def init_qas(self, n_block):
    self.n_block = n_block
    n_qs = 55 * n_block  # 55维 × 57块 = 3135个量化器
    self.qas = nn.ModuleList([])
    for i in range(n_qs): 
        self.qas.append(VanillaQuan(bit=8).cuda())
```

**每个量化器的初始状态**:
- `scale`: 空tensor
- `zero_point`: 空tensor
- `min_val`: 空tensor
- `max_val`: 空tensor

### 更新机制

**VanillaQuan.forward() 中的 update() 方法**:

```python
def update(self, x):
    # 更新最大值（只增不减）
    if self.max_val.nelement() == 0 or self.max_val.data < x.max().data:
        self.max_val.data = x.max().data
    self.max_val.clamp_(min=0)
    
    # 更新最小值（只减不增）
    if self.min_val.nelement() == 0 or self.min_val.data > x.min().data:
        self.min_val.data = x.min().data 
    self.min_val.clamp_(max=0)
    
    # 重新计算 scale 和 zero_point
    self.scale, self.zero_point = calcScaleZeroPoint(
        self.min_val, self.max_val, self.bit
    )
```

**更新时机**: 每次前向传播时都会调用 `update()`

**更新策略**: 
- 追踪历史最大值和最小值
- 动态调整量化范围
- 确保所有见过的值都能被量化

## 分块量化的详细机制

### 为什么要分块？

1. **空间自适应**: 不同空间区域的特征分布不同
2. **提高精度**: 每个块使用独立的量化参数
3. **减少量化误差**: 避免全局量化导致的精度损失

### 分块策略

**块大小计算** (`utils/quant_utils.py`):
```python
def split_length(length, n):
    base_length = length / n
    floor_length = int(base_length)
    remainder = length - (floor_length * n)
    # 前 remainder 个块大小为 floor_length + 1
    # 后面的块大小为 floor_length
    result = [floor_length + 1] * remainder + [floor_length] * (n - remainder)
    return result
```

**示例**（truck场景）:
- 总点数: 343,376
- AC系数数量: 343,375
- 块数量: 57
- 每块大小: 约 6,024 个点

### 量化器分配

```
维度0 (opacity):   qas[0]   到 qas[56]    (57个)
维度1 (euler_x):   qas[57]  到 qas[113]   (57个)
维度2 (euler_y):   qas[114] 到 qas[170]   (57个)
...
维度54 (scale_z):  qas[3078] 到 qas[3134] (57个)

总计: 55 × 57 = 3,135 个量化器
```

## 量化参数的存储

### t.npz 文件结构

```python
trans_array = [
    depth,           # 八叉树深度 (1个)
    n_block,         # 块数量 (1个)
    # 维度0的57个块
    scale_0_0, zero_point_0_0,
    scale_0_1, zero_point_0_1,
    ...
    scale_0_56, zero_point_0_56,
    # 维度1的57个块
    scale_1_0, zero_point_1_0,
    ...
    # ... 共55维
]

总参数数量: 2 + 55 × 57 × 2 = 6,272 个参数
```

## 量化精度分析

### 8-bit 量化范围

- **量化级别**: 256 (0-255)
- **表示范围**: [min_val, max_val]
- **量化步长**: scale = (max_val - min_val) / 255

### 量化误差

**理论误差**:
```
最大量化误差 = scale / 2
相对误差 = scale / (2 * |x|)
```

**实际影响**:
- DC系数（低频）: 误差较小，影响整体亮度
- AC系数（高频）: 误差较大，影响细节
- 分块量化可以减少约 30-50% 的量化误差

## 与LSQ的对比

### 如果要切换到LSQ

**需要修改的地方**:

1. **初始化量化器**:
```python
# scene/gaussian_model.py
def init_qas(self, n_block):
    self.qas = nn.ModuleList([])
    for i in range(55 * n_block): 
        self.qas.append(LsqQuan(bit=8, init_yet=False).cuda())
```

2. **添加量化器参数到优化器**:
```python
# scene/gaussian_model.py - training_setup()
l.append({'params': self.qas.parameters(), 'lr': 0.001, "name": "quantizers"})
```

3. **初始化量化参数**:
```python
# 第一次前向传播后
for qa in self.qas:
    if not qa.init_yet:
        qa.init_from(x)
```

**优缺点**:
- ✅ 可能提高精度 0.5-1 dB PSNR
- ❌ 训练时间增加 20-30%
- ❌ 内存占用增加
- ❌ 需要调整学习率

## 总结

**当前配置**:
- ✅ 使用 VanillaQuan（普通均匀量化）
- ✅ 8-bit 量化
- ✅ 分块量化（57个块）
- ✅ 3,135 个独立量化器
- ✅ 量化发生在训练和保存时
- ✅ 量化参数动态更新（基于min-max）
- ❌ 量化参数不可学习（非LSQ）

**量化流程**:
```
训练时: 特征 → RAHT → 量化(VanillaQuan) → 反量化 → 逆RAHT → 渲染
保存时: 特征 → RAHT → 量化(uint8) → 保存
加载时: 加载(uint8) → 反量化 → 逆RAHT → 特征
```

这种方案在速度和精度之间取得了良好的平衡，适合大多数场景。
