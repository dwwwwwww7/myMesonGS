#!/bin/bash


#!/bin/bash

ITERS=0
TEST_ITERS="$ITERS" 
CONFIG=config4
LSEG=0
CB=0
DEPTH=0
MODEL_BASE="F:/3dgs_data/models"
OUTPUT_BASE="F:/3dgs_data/my_RAHT_results2026/output_gpcc_zsavenpz_bitpacknew"
CSV_BASE="F:/3dgs_data/my_RAHT_results2026/exp_data_gpcc_zsavenpz_bitpacknew/csv"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$CSV_BASE"


process_scene () {
    local SCENE=$1
    local DATAPATH=$2

    echo "=== Processing scene: $SCENE ==="
    
    MODEL_PATH="$MODEL_BASE/$SCENE"
    INITIALPATH="$MODEL_PATH/point_cloud/iteration_30000/point_cloud.ply"
    CSVPATH="$CSV_BASE/${SCENE}_${CONFIG}.csv"
    SAVEPATH="$OUTPUT_BASE/${SCENE}_${CONFIG}"
    
    # 调试：打印路径
    echo "CSVPATH: $CSVPATH"
    echo "SAVEPATH: $SAVEPATH"

    python mesongs.py -s "$DATAPATH" \
        --given_ply_path "$INITIALPATH" \
        --num_bits 8 \
        --save_imp \
        --eval \
        --iterations $ITERS \
        --finetune_lr_scale 1 \
        --convert_SHs_python \
        --percent 0 \
        --steps 1000 \
        --scene_imp $SCENE \
        --depth $DEPTH \
        --raht \
        --clamp_color \
        --per_block_quant \
        --lseg $LSEG \
        --debug \
        --hyper_config $CONFIG \
        --csv_path "$CSVPATH" \
        --model_path "$SAVEPATH" \
        --test_iterations $TEST_ITERS 

    echo "=== Finished scene: $SCENE ==="
    echo
}

# mic scene
#SCENES=("mic" "lego" "drums" "ficus" "hotdog" "materials" "ship" "chair")
#for SCENE in "${SCENES[@]}"; do
 #   process_scene "$SCENE" "/data/zdw/datasets/nerf_synthetic/$SCENE"
#done


# TUM scenes
# SCENES=("train" "truck")
SCENES=("train")
for SCENE in "${SCENES[@]}"; do
    process_scene "$SCENE" "F:/3dgs_data/image&sparse/$SCENE"
done

# db
#SCENES=("drjohnson" "playroom")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "E:/3dgs data/image&sparse/$SCENE"
#done


# 360_v2 scenes
#SCENES=("counter" "room" "bicycle" "bonsai" "kitchen" "garden" "stump")
#SCENES=("room")
#for SCENE in "${SCENES[@]}"; do
#    process_scene "$SCENE" "/data/zdw/datasets/360_v2/$SCENE"
#done


