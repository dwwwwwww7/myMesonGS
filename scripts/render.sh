#!/bin/bash

# MesonGS 渲染脚本 - 支持多场景批量渲染

# 路径配置
MAINDIR="E:/swisstransfer_13badc40-f390-47aa-9715-08c791d9b3bf/sparseRAHT_backup"  #"E:/3dgs data/MesonGS"
DATADIR="E:/3dgs data/image&sparse"

# 要处理的场景列表（取消注释需要的场景）
declare -a SCENES=(
    #"playroom"
    # "bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
     "train"
     "truck"
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
)

# 默认配置
DEFAULT_CONFIG="config4"
DEFAULT_ITERATION="2000"    # 可选: "best", "-1"(自动), "0", "10", 或具体数字

# 渲染选项
SKIP_TRAIN=true             # 是否跳过训练集渲染
SKIP_TEST=false             # 是否跳过测试集渲染
SAVE_DIR_NAME="ours"        # 输出目录前缀 (metrics.py 需要以 "ours" 开头)

# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# 解析命令行参数
# ============================================================================

# 如果提供了场景名，只渲染该场景
if [ -n "$1" ]; then
    SCENES=("$1")
fi

# 配置文件
CONFIG="${2:-$DEFAULT_CONFIG}"

# 迭代次数
ITERATION="${3:-$DEFAULT_ITERATION}"

print_separator() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# ============================================================================
# 主循环
# ============================================================================

print_separator "MesonGS 批量渲染脚本"
echo "场景列表: ${SCENES[@]}"
echo "配置文件: $CONFIG"
echo "迭代: $ITERATION"
print_separator "开始渲染"

SUCCESS_COUNT=0
FAIL_COUNT=0

for SCENE in "${SCENES[@]}"; do
    print_separator "渲染场景: $SCENE"
    
    CKPT="${SCENE}_${CONFIG}"
    
    # 设置迭代参数
    if [ "$ITERATION" = "best" ] || [ "$ITERATION" = "-1" ]; then
        ITER_ARG="--iteration -1"
    else
        ITER_ARG="--iteration $ITERATION"
    fi
    
    # 构建渲染命令
    RENDER_CMD="python render.py"
    RENDER_CMD="$RENDER_CMD -s \"$DATADIR/$SCENE\""
    RENDER_CMD="$RENDER_CMD -m \"$MAINDIR/output/$CKPT\""
    RENDER_CMD="$RENDER_CMD $ITER_ARG"
    RENDER_CMD="$RENDER_CMD --dec_npz --eval -w" #-w设置渲染时的背景颜色
    RENDER_CMD="$RENDER_CMD --scene_name $SCENE"
    RENDER_CMD="$RENDER_CMD --log_name ${SCENE}_${CONFIG}"
    RENDER_CMD="$RENDER_CMD --save_dir_name $SAVE_DIR_NAME"
    
    if [ "$SKIP_TRAIN" = true ]; then
        RENDER_CMD="$RENDER_CMD --skip_train"
    fi
    
    if [ "$SKIP_TEST" = true ]; then
        RENDER_CMD="$RENDER_CMD --skip_test"
    fi
    
    echo "执行命令:"
    echo "$RENDER_CMD"
    echo ""
    
    # 执行渲染
    eval $RENDER_CMD
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 场景 $SCENE 渲染完成"
        ((SUCCESS_COUNT++))
    else
        echo ""
        echo "✗ 场景 $SCENE 渲染失败"
        ((FAIL_COUNT++))
    fi
done

# ============================================================================
# 总结
# ============================================================================

print_separator "批量渲染完成"

echo "处理摘要:"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo ""

echo "输出目录结构 (metrics.py 兼容):"
echo "  $MAINDIR/output/[场景]_$CONFIG/test/ours_XXX/renders/"
echo "  $MAINDIR/output/[场景]_$CONFIG/test/ours_XXX/gt/"
echo ""

echo "运行 metrics.py 计算指标:"
for SCENE in "${SCENES[@]}"; do
    echo "  python metrics.py -m \"$MAINDIR/output/${SCENE}_${CONFIG}\""
    python metrics.py -m "$MAINDIR/output/${SCENE}_${CONFIG}"
done
echo ""
