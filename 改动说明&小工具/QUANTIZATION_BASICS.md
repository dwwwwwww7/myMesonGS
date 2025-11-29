# 量化基础：Scale 和 Zero Point 详解

## 什么是量化？

量化就是将**连续的浮点数**映射到**离散的整数**，以减少存储空间。

### 简单类比

想象你要用温度计测量温度：
- **浮点数**: 可以是任意精度，如 23.456789°C
- **量化后**: 只能是刻度上的值，如 23°C, 24°C, 25°C

## 核心概念

### 1. 量化公式

**前向量化**（浮点 → 整数）:
```python
q_x = round(x / scale + zero_point)
```

**反量化**（整数 → 浮点）:
```python
x = scale * (q_x - zero_point)
```

### 2. Scale（缩放因子）

**定义**: 每个量化级别代表的实际数值范围

**公式**:
```python
scale = (max_val - min_val) / (2^bit - 1)
```

**含义**:
- scale 越大 → 量化步长越大 → 精度越低
- scale 越小 → 量化步长越小 → 精度越高

**例子**:
```python
# 8-bit 量化，范围 [0, 255]
min_val = -10.0
max_val = 10.0

scale = (10.0 - (-10.0)) / (255 - 0)
      = 20.0 / 255
      = 0.0784

# 这意味着每个量化级别代表 0.0784 的数值变化
```

### 3. Zero Point（零点）

**定义**: 实际值为 0 时对应的量化整数

**公式**:
```python
zero_point = round(qmax - max_val / scale)
```

**含义**:
- zero_point 让我们可以用**无符号整数**表示**有符号浮点数**
- 它是量化范围的"偏移量"

**例子**:
```python
# 继续上面的例子
zero_point = 255 - 10.0 / 0.0784
           = 255 - 127.5
           = 127.5
           ≈ 128 (四舍五入)

# 这意味着：
# 实际值 0.0 → 量化值 128
# 实际值 -10.0 → 量化值 0
# 实际值 +10.0 → 量化值 255
```

## 详细示例

### 示例1: 对称范围 [-10, 10]

```python
min_val = -10.0
max_val = 10.0
bit = 8

# 计算 scale
scale = (10.0 - (-10.0)) / (255 - 0)
      = 20.0 / 255
      = 0.0784

# 计算 zero_point
zero_point = 255 - 10.0 / 0.0784
           = 127.5 ≈ 128

# 量化映射：
实际值    量化值
-10.0  →  0
 -5.0  →  64
  0.0  →  128  ← zero_point
 +5.0  →  192
+10.0  →  255
```

### 示例2: 非对称范围 [0, 20]

```python
min_val = 0.0
max_val = 20.0
bit = 8

# 计算 scale
scale = (20.0 - 0.0) / (255 - 0)
      = 20.0 / 255
      = 0.0784

# 计算 zero_point
zero_point = 255 - 20.0 / 0.0784
           = 0

# 量化映射：
实际值    量化值
  0.0  →  0    ← zero_point
  5.0  →  64
 10.0  →  128
 15.0  →  192
 20.0  →  255
```

### 示例3: 负数范围 [-20, 0]

```python
min_val = -20.0
max_val = 0.0
bit = 8

# 计算 scale
scale = (0.0 - (-20.0)) / (255 - 0)
      = 20.0 / 255
      = 0.0784

# 计算 zero_point
zero_point = 255 - 0.0 / 0.0784
           = 255

# 量化映射：
实际值    量化值
-20.0  →  0
-15.0  →  64
-10.0  →  128
 -5.0  →  192
  0.0  →  255  ← zero_point
```

## 为什么需要 Zero Point？

### 问题：如果没有 zero_point

假设我们只用 scale，强制 0 映射到量化值 0：

```python
# 范围 [-10, 10]，8-bit
# 如果强制 0 → 0
scale = 10.0 / 127  # 只能用一半的量化范围！

# 映射：
-10.0 → -127  ❌ 超出 [0, 255] 范围！
  0.0 →    0
+10.0 → +127
```

**问题**: 
- 负数无法表示（如果用无符号整数）
- 或者浪费一半的量化范围（如果用有符号整数）

### 解决：使用 zero_point

```python
# 范围 [-10, 10]，8-bit
scale = 20.0 / 255
zero_point = 128

# 映射：
-10.0 → 0    ✓ 在 [0, 255] 范围内
  0.0 → 128  ✓ 在 [0, 255] 范围内
+10.0 → 255  ✓ 在 [0, 255] 范围内
```

**优点**:
- ✅ 可以用无符号整数表示有符号浮点数
- ✅ 充分利用整个量化范围
- ✅ 灵活处理非对称分布

## 实际代码中的应用

### VanillaQuan 的工作流程

```python
class VanillaQuan:
    def __init__(self, bit, all_positive=True):
        self.bit = bit
        self.thd_neg = 0           # 量化下界
        self.thd_pos = 2**bit - 1  # 量化上界
        
    def update(self, x):
        # 追踪数据的实际范围
        self.min_val = x.min()
        self.max_val = x.max()
        
        # 计算量化参数
        self.scale, self.zero_point = calcScaleZeroPoint(
            self.min_val, self.max_val, self.bit
        )
    
    def forward(self, x):
        # 1. 更新 scale 和 zero_point
        self.update(x)
        
        # 2. 量化
        q_x = x / self.scale + self.zero_point
        q_x = clamp(round(q_x), 0, 2**self.bit - 1)
        
        # 3. 反量化（模拟量化误差）
        x_dequant = self.scale * (q_x - self.zero_point)
        
        return x_dequant
```

### 为什么看到的都是正数？

**答案**: 因为量化后的整数值 `q_x` 确实都是正数（0-255），但它们可以表示负的浮点数！

```python
# 例子：euler 角度可能是负数
euler_x = -1.5  # 实际值（负数）

# 量化
q_x = -1.5 / 0.01 + 150  # 假设 scale=0.01, zero_point=150
    = -150 + 150
    = 0  # 量化值（正数）

# 反量化
euler_x_dequant = 0.01 * (0 - 150)
                = 0.01 * (-150)
                = -1.5  # 恢复为负数
```

## 可视化理解

### 量化过程图解

```
实际浮点数范围:  [-10.0 ────────── 0.0 ────────── +10.0]
                    ↓               ↓               ↓
量化整数范围:      [0 ──────────── 128 ──────────── 255]
                    ↑               ↑               ↑
                 min_val      zero_point        max_val

scale = 0.0784
zero_point = 128

量化公式: q_x = x / 0.0784 + 128
反量化公式: x = 0.0784 * (q_x - 128)
```

### 不同 zero_point 的含义

```
zero_point = 0:
  实际范围全是正数 [0, max]
  [0 ────────────────────────── 255]
   ↑                             ↑
   0                           max

zero_point = 128:
  实际范围对称 [-max, +max]
  [0 ──────────── 128 ──────────── 255]
   ↑              ↑                ↑
  -max            0              +max

zero_point = 255:
  实际范围全是负数 [min, 0]
  [0 ────────────────────────── 255]
   ↑                             ↑
  min                            0
```

## 你的数据中的 Scale 和 Zero Point

### 从可视化图像中可以看到：

1. **Scale 热图**:
   - 颜色深浅 = scale 大小
   - 深色 = 小 scale = 数值范围小 = 精度高
   - 浅色 = 大 scale = 数值范围大 = 精度低

2. **Zero Point 热图**:
   - 红色 = 高 zero_point ≈ 255 = 数据主要是负数
   - 蓝色 = 低 zero_point ≈ 0 = 数据主要是正数
   - 白色 = 中等 zero_point ≈ 128 = 数据有正有负

### 实际例子（你的数据）:

```python
# opacity (总是正数)
min_val = 0.0
max_val = 1.0
scale = 1.0 / 255 = 0.0039
zero_point ≈ 0  # 接近0，因为都是正数

# euler (有正有负)
min_val = -3.14
max_val = 3.14
scale = 6.28 / 255 = 0.0246
zero_point ≈ 128  # 接近中间，因为对称

# scale (总是正数，但范围大)
min_val = -5.0  # log空间可能是负数
max_val = 2.0
scale = 7.0 / 1023 = 0.0068  # 10-bit
zero_point ≈ 730  # 偏向高值
```

## 总结

### Scale
- **作用**: 控制量化精度
- **公式**: `(max - min) / (2^bit - 1)`
- **含义**: 每个量化级别的实际数值步长

### Zero Point
- **作用**: 让无符号整数能表示有符号浮点数
- **公式**: `qmax - max / scale`
- **含义**: 实际值为0时对应的量化整数

### 为什么量化值都是正数？
- ✅ 使用无符号整数存储（uint8, uint16）
- ✅ 通过 zero_point 偏移来表示负数
- ✅ 反量化时会恢复为正确的正负值

### 关键理解
```
量化值（存储）: 总是正数 [0, 2^bit-1]
实际值（使用）: 可以是负数 [min, max]
桥梁: scale 和 zero_point
```

这就是为什么你看到的量化值都是正数，但实际上它们可以表示负数！🎯
