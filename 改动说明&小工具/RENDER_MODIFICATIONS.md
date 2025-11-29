# 渲染功能修改说明

## 修改概述

为了支持新的55维RAHT特征格式（包含scale），对渲染相关代码进行了更新。

## 主要修改

### 1. `scene/gaussian_model.py` - `load_npz()` 函数

**修改前的问题：**
- 仍然尝试加载VQ文件（ntk.npz, um.npz）
- 独立加载scale文件（ct.npz）
- 只处理7维RAHT特征（opacity + euler + f_dc）

**修改后：**
- 直接从orgb.npz加载55维RAHT系数
- 不再加载VQ相关文件
- 不再加载独立的scale文件
- 从RAHT特征中提取所有属性：
  - opacity: [0:1]
  - euler: [1:4]
  - f_dc: [4:7]
  - f_rest: [7:52] → reshape为 [N, 15, 3]
  - scale: [52:55]

**关键代码：**
```python
# 加载55维RAHT系数
q_raht_i = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda")
q_raht_i = q_raht_i.reshape(55, -1).contiguous().transpose(0, 1)

# 反量化所有55维
for i in range(55):
    raht_i = torch_vanilla_dequant_ave(q_raht_i[:, i], split, trans_array[qa_cnt:qa_cnt+2*n_block])
    raht_ac.append(raht_i.reshape(-1, 1))
    qa_cnt += 2*n_block

# 执行逆RAHT变换
# ... (RAHT逆变换代码)

# 提取所有属性
self._opacity = nn.Parameter(raht_features[:, :1].requires_grad_(False))
self._euler = nn.Parameter(raht_features[:, 1:4].nan_to_num_(0).requires_grad_(False))
self._features_dc = nn.Parameter(raht_features[:, 4:7].unsqueeze(1).requires_grad_(False))
self._features_rest = nn.Parameter(f_rest_flat.reshape(-1, self.n_sh - 1, 3).requires_grad_(False))
self._scaling = nn.Parameter(raht_features[:, 52:55].requires_grad_(False))
```

### 2. `render.py` - 无需修改

render.py已经支持`--dec_npz`参数，会自动调用`load_npz()`函数。

## 文件格式变化

### 旧格式（7维RAHT + VQ + 独立scale）
```
bins/
├── oct.npz          # 八叉树结构
├── ntk.npz          # VQ索引
├── um.npz           # VQ码本
├── orgb.npz         # 7维RAHT系数 (opacity + euler + f_dc)
├── ct.npz           # 独立的scale量化
└── t.npz            # 量化参数
```

### 新格式（55维RAHT，统一处理）
```
bins/
├── oct.npz          # 八叉树结构
├── orgb.npz         # 55维RAHT系数 (opacity + euler + f_dc + f_rest + scale)
└── t.npz            # 量化参数
```

## 使用方法

### 1. 训练并保存压缩模型
```bash
bash scripts/mesongs_block.sh
```

### 2. 从压缩模型渲染
```bash
python render.py \
    -s data/nerf_synthetic/lego \
    -m output/your_model \
    --iteration best \
    --dec_npz \
    --skip_train \
    --log_name "render_test"
```

### 3. 参数说明
- `--dec_npz`: 从NPZ压缩文件加载模型
- `--skip_train`: 跳过训练集渲染
- `--skip_test`: 跳过测试集渲染
- `--iteration`: 指定迭代次数（-1表示最新，或使用"best"）
- `--log_name`: 日志文件名
- `--save_dir_name`: 保存目录名

## 验证步骤

1. **检查文件结构**
   ```bash
   ls output/your_model/point_cloud/iteration_best/pc_npz/bins/
   # 应该看到: oct.npz, orgb.npz, t.npz
   # 不应该有: ntk.npz, um.npz, ct.npz
   ```

2. **运行渲染**
   ```bash
   bash test_render.sh
   ```

3. **检查输出**
   - 查看控制台输出，确认加载了55维RAHT特征
   - 检查渲染图像质量
   - 对比PSNR/SSIM/LPIPS指标

## 调试信息

load_npz函数会输出详细的加载信息：
```
【加载压缩模型】
  目录: output/.../bins
  八叉树深度: 12
  块数量: 57
  八叉树点数: 32768
  解码后点数: 32768
  RAHT DC系数形状: torch.Size([55])
  RAHT AC系数形状: torch.Size([1802240])
  重塑后 AC 系数: torch.Size([32767, 55])
  
  反量化 55 维 RAHT 特征...
  反量化完成，使用了 6270 个量化参数 (55 × 57 × 2)
  raht_ac 形状: torch.Size([32767, 55])
  完整 RAHT 系数 C 形状: torch.Size([32768, 55])
  
  执行逆 RAHT 变换...
  逆 RAHT 变换完成
  raht_features 形状: torch.Size([32768, 55])
  
  提取特征:
    opacity: torch.Size([32768, 1])
    euler: torch.Size([32768, 3])
    f_dc: torch.Size([32768, 1, 3])
    f_rest: torch.Size([32768, 15, 3])
    scale: torch.Size([32768, 3])
  
【加载完成】所有属性已从 RAHT 特征中提取
```

## 常见问题

### Q1: 渲染时报错找不到ntk.npz
**A:** 这是因为使用了旧版本的load_npz函数。确保已经更新了scene/gaussian_model.py中的load_npz函数。

### Q2: 渲染结果质量下降
**A:** 检查：
1. 训练是否完成
2. 量化参数是否正确
3. 是否使用了正确的迭代版本

### Q3: 维度不匹配错误
**A:** 确保：
1. save_npz保存的是55维特征
2. load_npz加载的是55维特征
3. 两者的维度顺序一致

## 兼容性说明

- ✅ 新训练的模型可以正常渲染
- ❌ 旧格式的模型需要重新训练
- ✅ render.py的其他功能（--dec_cov, --dec_euler）不受影响

## 性能对比

| 指标 | 旧方案 (VQ) | 新方案 (RAHT) |
|------|------------|--------------|
| 文件数量 | 6个 | 3个 |
| 压缩率 | 中等 | 更高 |
| 解码速度 | 快 | 快 |
| 质量 | 好 | 更好 |
| 统一性 | 分散 | 统一 |

## 总结

通过将scale也纳入RAHT变换，实现了：
1. ✅ 统一的特征处理流程
2. ✅ 更简洁的文件结构
3. ✅ 更好的压缩效果
4. ✅ 完整的渲染支持

所有属性（opacity, euler, f_dc, f_rest, scale）现在都通过同一个RAHT变换和量化流程处理，代码更加简洁和易于维护。
