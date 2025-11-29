# 修改总结：移除VQ，f_rest使用RAHT变换

## 修改目标
将高阶球谐函数系数 (f_rest, 45维) 的处理方式从**矢量量化(VQ)**改为与DC分量 (f_dc, 3维) 相同的**RAHT变换 + 分块量化**方式。

## 主要修改

### 1. `scene/gaussian_model.py`

#### 1.1 `init_qas()` - 增加量化器数量
**修改前：** 10个属性 × n_block = 570个量化器（truck场景）
- opacity(1) + euler(3) + f_dc(3) + scale(3) = 10

**修改后：** 55个属性 × n_block = 3135个量化器（truck场景）
- opacity(1) + euler(3) + f_dc(3) + **f_rest(45)** + scale(3) = 55

```python
# 原来: 10 * n_block
# 现在: 55 * n_block
n_qs = 55 * n_block
```

#### 1.2 `vq_fe()` - 完全跳过VQ训练
**修改前：** 训练码本，生成索引
**修改后：** 直接返回，不做任何处理

```python
def vq_fe(self, imp, codebook_size, batch_size, steps):
    print(f"Skipping VQ - f_rest will use RAHT transform like f_dc")
    return
```

#### 1.3 `save_npz()` - 将f_rest加入RAHT变换
**修改前：** 
- 保存VQ码本到 `um.npz`
- 保存VQ索引到 `ntk.npz`
- RAHT变换：opacity(1) + euler(3) + f_dc(3) = 7维

**修改后：**
- 不再保存 `um.npz` 和 `ntk.npz`
- RAHT变换：opacity(1) + euler(3) + f_dc(3) + **f_rest(45)** = 52维

```python
rf = torch.concat([
    self.get_origin_opacity.detach(), 
    eulers.detach(), 
    self.get_features_dc.detach().contiguous().squeeze(),
    self.get_features_rest.detach().contiguous().flatten(-2)  # 新增
], axis=-1)
```

#### 1.4 `load_npz()` - 从RAHT加载f_rest
**修改前：**
- 从 `ntk.npz` 和 `um.npz` 加载VQ数据
- 通过索引查找码本重建f_rest
- 反量化7维数据

**修改后：**
- 不再加载VQ文件
- 反量化52维数据
- 从RAHT特征中提取f_rest

```python
# 反量化52维（原来是7维）
for i in range(52):  # 原来是 range(7)
    rf_i = torch_vanilla_dequant_ave(...)
    rf_orgb.append(rf_i.reshape(-1, 1))

# 从RAHT特征中提取
self._opacity = nn.Parameter(raht_features[:, :1].requires_grad_(False))
self._euler = nn.Parameter(raht_features[:, 1:4].nan_to_num_(0).requires_grad_(False))
self._features_dc = nn.Parameter(raht_features[:, 4:7].unsqueeze(1).requires_grad_(False))
self._features_rest = nn.Parameter(raht_features[:, 7:52].reshape(-1, self.n_sh - 1, 3).requires_grad_(False))
```

#### 1.5 `get_indexed_feature_extra()` - 简化为直接返回
**修改前：** 通过索引查找码本
**修改后：** 直接返回 `_features_rest`

```python
@property
def get_indexed_feature_extra(self):
    return self._features_rest
```

### 2. `arguments/__init__.py`

#### 2.1 禁用use_indexed
**修改前：** `self.use_indexed = True`
**修改后：** `self.use_indexed = False`

因为不再使用VQ索引，渲染时直接使用完整的f_rest数据。

---

## 数据维度变化

### 保存时 (save_npz)
| 文件 | 修改前 | 修改后 |
|------|--------|--------|
| `orgb.npz` | 7维 × n_points | 52维 × n_points |
| `ntk.npz` | 索引数组 | **删除** |
| `um.npz` | 码本(4096×45) | **删除** |
| `ct.npz` | 3维 × n_points | 3维 × n_points (不变) |
| `t.npz` | 2 + 7×2×n_block + 3×2×n_block | 2 + 52×2×n_block + 3×2×n_block |

### 量化参数数量
| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| truck (n_block=57) | 570个量化器 | 3135个量化器 |

---

## 压缩效果变化

### 修改前（使用VQ）
- **f_rest存储**：码本(0.74MB) + 索引(0.69MB) = 1.43 MB
- **压缩率**：~43倍

### 修改后（使用RAHT）
- **f_rest存储**：RAHT系数 + 量化参数
- **预期大小**：约 3-5 MB（取决于量化精度）
- **压缩率**：~12-20倍

### 权衡
- ✅ **优点**：
  - 代码更简洁，无需VQ训练（节省~7分钟）
  - 无需存储码本和索引
  - 质量可能更好（无VQ量化损失）
  
- ❌ **缺点**：
  - 文件大小增加约2-4 MB
  - 总压缩率降低

---

## 测试建议

1. **运行压缩测试**
```bash
bash scripts/mesongs_block.sh
```

2. **检查文件大小**
```bash
# 检查压缩后的文件
ls -lh output/*/point_cloud/iteration_0/pc_npz/bins.zip
```

3. **验证质量**
- 检查PSNR/SSIM/LPIPS指标
- 与原始VQ版本对比

4. **预期结果**
- 训练时间减少（无VQ训练）
- 文件大小增加2-4 MB
- 质量可能略有提升

---

## 回滚方法

如果需要恢复VQ功能，使用git恢复以下文件：
```bash
git checkout scene/gaussian_model.py
git checkout arguments/__init__.py
```

---

## 注意事项

1. **不兼容旧模型**：修改后的代码无法加载之前用VQ压缩的模型（因为文件格式变化）
2. **内存使用**：RAHT变换52维数据比7维需要更多内存
3. **量化参数**：从570个增加到3135个，trans_array大小显著增加

---

## 修改文件清单

- ✅ `scene/gaussian_model.py` - 主要修改
- ✅ `arguments/__init__.py` - 禁用use_indexed
- ⚠️ 不需要修改：`mesongs.py`, `vq.py`, `raht_torch.py`
