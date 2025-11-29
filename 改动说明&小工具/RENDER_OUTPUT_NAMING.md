# 渲染输出文件夹命名说明

## 更新内容

渲染输出的文件夹名称现在会**自动包含迭代次数**，方便区分不同版本的渲染结果。

## 命名格式

### 之前
```
<MODEL_PATH>/test/npz_render/
<MODEL_PATH>/train/npz_render/
```

### 现在
```
<MODEL_PATH>/test/npz_render_iter20/
<MODEL_PATH>/train/npz_render_iter20/
```

## 使用示例

### 1. 基本用法

```bash
python render.py \
    -s D:/3DGS_seq/truck \
    -m D:/3DGS_seq/result_mesongs/output/truck_config3 \
    --iteration 20 \
    --dec_npz \
    --skip_test \
    --save_dir_name "npz_render"
```

**输出目录**:
```
D:/3DGS_seq/result_mesongs/output/truck_config3/
└── train/
    └── npz_render_iter20/
        ├── renders/
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        └── gt/
            ├── 00000.png
            ├── 00001.png
            └── ...
```

### 2. 渲染多个迭代版本

```bash
# 渲染迭代0
python render.py -s <SCENE> -m <MODEL> --iteration 0 --dec_npz --skip_test --save_dir_name "npz_render"

# 渲染迭代10
python render.py -s <SCENE> -m <MODEL> --iteration 10 --dec_npz --skip_test --save_dir_name "npz_render"

# 渲染迭代20
python render.py -s <SCENE> -m <MODEL> --iteration 20 --dec_npz --skip_test --save_dir_name "npz_render"
```

**输出目录结构**:
```
<MODEL_PATH>/train/
├── npz_render_iter0/
│   ├── renders/
│   └── gt/
├── npz_render_iter10/
│   ├── renders/
│   └── gt/
└── npz_render_iter20/
    ├── renders/
    └── gt/
```

### 3. 使用不同的基础名称

```bash
# 高质量渲染
python render.py ... --save_dir_name "high_quality"
# 输出: high_quality_iter20/

# 快速测试
python render.py ... --save_dir_name "quick_test"
# 输出: quick_test_iter20/

# 对比实验
python render.py ... --save_dir_name "experiment_A"
# 输出: experiment_A_iter20/
```

## 文件夹内容

每个渲染输出文件夹包含：

```
npz_render_iter20/
├── renders/          # 渲染的图像
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── gt/              # 真实图像（Ground Truth）
    ├── 00000.png
    ├── 00001.png
    ├── 00002.png
    └── ...
```

## 自动命名逻辑

代码会自动处理：

```python
# render.py 中的逻辑
if render_args.save_dir_name and scene.loaded_iter is not None:
    original_save_dir = render_args.save_dir_name
    render_args.save_dir_name = f"{original_save_dir}_iter{scene.loaded_iter}"
    print(f"渲染输出目录: {render_args.save_dir_name}")
```

**规则**:
1. 如果指定了 `--save_dir_name`，会自动添加 `_iter{迭代次数}`
2. 如果没有指定，使用默认名称
3. 迭代次数从实际加载的模型中获取

## 优点

### ✅ 方便对比
```bash
# 可以同时保留多个版本的渲染结果
ls <MODEL_PATH>/train/
npz_render_iter0/
npz_render_iter10/
npz_render_iter20/
npz_render_best/
```

### ✅ 避免覆盖
```bash
# 不会意外覆盖之前的渲染结果
# 每次渲染都会创建新的文件夹
```

### ✅ 清晰标识
```bash
# 一眼就能看出是哪个迭代的结果
npz_render_iter20/  # 第20次迭代
npz_render_iter100/ # 第100次迭代
```

## 批量渲染脚本

创建一个脚本来渲染多个迭代：

```bash
#!/bin/bash
# batch_render.sh

SCENE_PATH="D:/3DGS_seq/truck"
MODEL_PATH="D:/3DGS_seq/result_mesongs/output/truck_config3"

# 渲染多个迭代
for ITER in 0 5 10 15 20; do
    echo "渲染迭代 $ITER..."
    python render.py \
        -s "$SCENE_PATH" \
        -m "$MODEL_PATH" \
        --iteration $ITER \
        --dec_npz \
        --skip_test \
        --save_dir_name "npz_render" \
        --quick  # 快速模式，不保存图像
    echo "完成！"
    echo ""
done

echo "所有迭代渲染完成！"
echo "结果保存在:"
ls -d "$MODEL_PATH"/train/npz_render_iter*/
```

## 查看结果

### 查看所有渲染结果

```bash
# Windows (PowerShell)
ls D:/3DGS_seq/result_mesongs/output/truck_config3/train/npz_render_iter*/

# Linux/Mac
ls -d <MODEL_PATH>/train/npz_render_iter*/
```

### 对比不同迭代

```bash
# 查看PSNR变化
grep "psnr" <MODEL_PATH>/train/npz_render_iter*/metrics.txt

# 或者查看CSV日志
cat exp_data/csv/test_render.csv
```

## 注意事项

### 1. 磁盘空间

每个渲染结果会占用一定空间：
```
renders/ + gt/ ≈ 图像数量 × 2 × 图像大小
例如: 200张图 × 2 × 500KB ≈ 200MB
```

### 2. 清理旧结果

定期清理不需要的渲染结果：
```bash
# 删除特定迭代
rm -rf <MODEL_PATH>/train/npz_render_iter0/

# 只保留最新的几个
# (手动选择保留哪些)
```

### 3. 命名冲突

如果手动指定了包含 "iter" 的名称：
```bash
--save_dir_name "my_iter_test"
# 输出: my_iter_test_iter20/
```

建议使用不包含 "iter" 的基础名称。

## 总结

现在渲染输出会自动包含迭代次数：

- ✅ **自动命名**: `{base_name}_iter{iteration}`
- ✅ **避免覆盖**: 每次渲染创建新文件夹
- ✅ **方便对比**: 可以同时保留多个版本
- ✅ **清晰标识**: 一眼看出是哪个迭代

开始使用新的命名方式吧！🎨
