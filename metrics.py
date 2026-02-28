#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
#使用inria 3DGS的metrics计算方法

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, eval_train=False):
    """
    评估渲染质量
    
    目录结构：
    model_path/
    ├── train/
    │   ├── iter_0/
    │   │   ├── 00000.png
    │   │   └── ...
    │   └── iter_1/
    │       └── ...
    ├── test/
    │   ├── iter_0/
    │   │   ├── 00000.png
    │   │   └── ...
    │   └── iter_1/
    │       └── ...
    └── gt/
        └── iter_0/
            ├── 00000.png
            └── ...
    """
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            # GT 目录（所有数据集共享）
            gt_base_dir = Path(scene_dir) / "gt"
            if not gt_base_dir.exists():
                print(f"  Warning: gt directory not found: {gt_base_dir}")
                continue

            # 评估 test 和/或 train 集
            eval_sets = ["test"]
            if eval_train:
                eval_sets.append("train")
            
            for eval_set in eval_sets:
                print(f"\n  Evaluating {eval_set} set...")
                eval_dir = Path(scene_dir) / eval_set
                
                # 检查目录是否存在
                if not eval_dir.exists():
                    print(f"    Warning: {eval_set} directory not found: {eval_dir}")
                    continue

                # 遍历所有 iter_X 目录
                for iter_dir_name in sorted(os.listdir(eval_dir)):
                    # 只处理 iter_X 格式的目录
                    if not iter_dir_name.startswith('iter_'):
                        print(f"    Skipping: {iter_dir_name} (not an iteration directory)")
                        continue
                    
                    # 创建唯一的方法名（包含数据集类型和迭代号）
                    method_key = f"{eval_set}_{iter_dir_name}"
                    print(f"    Method: {method_key}")

                    full_dict[scene_dir][method_key] = {}
                    per_view_dict[scene_dir][method_key] = {}

                    # 渲染图像目录
                    renders_dir = eval_dir / iter_dir_name
                    
                    # GT 图像目录（所有迭代都使用 iter_0）
                    gt_dir = gt_base_dir / "iter_0"
                    
                    # 检查目录是否存在
                    if not renders_dir.exists():
                        print(f"      Warning: renders directory not found: {renders_dir}")
                        continue
                    if not gt_dir.exists():
                        print(f"      Warning: gt directory not found: {gt_dir}")
                        continue
                    
                    # 检查是否有图像文件
                    render_files = list(renders_dir.glob("*.png")) + list(renders_dir.glob("*.jpg"))
                    gt_files = list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg"))
                    
                    if len(render_files) == 0:
                        print(f"      Warning: no images found in {renders_dir}")
                        continue
                    if len(gt_files) == 0:
                        print(f"      Warning: no images found in {gt_dir}")
                        continue
                    
                    print(f"      Found {len(render_files)} rendered images and {len(gt_files)} GT images")
                    
                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc=f"      Evaluating {method_key}"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    print("      SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("      PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("      LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method_key].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS": torch.tensor(lpipss).mean().item()})
                    per_view_dict[scene_dir][method_key].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            # 保存结果
            results_file = os.path.join(scene_dir, "results.json")
            per_view_file = os.path.join(scene_dir, "per_view.json")
            
            with open(results_file, 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=2)
            with open(per_view_file, 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=2)
            
            print(f"\n  Results saved to:")
            print(f"    {results_file}")
            print(f"    {per_view_file}")
            
            # 打印摘要
            if full_dict[scene_dir]:  # 只有在有结果时才打印摘要
                print(f"\n  Summary for {scene_dir}:")
                print("  " + "-"*66)
                print(f"  {'Method':<30s} | {'PSNR':>10s} | {'SSIM':>10s} | {'LPIPS':>10s}")
                print("  " + "-"*66)
                for method_key, metrics in full_dict[scene_dir].items():
                    # 检查 metrics 是否包含所有必需的键
                    if 'PSNR' in metrics and 'SSIM' in metrics and 'LPIPS' in metrics:
                        print(f"  {method_key:<30s} | {metrics['PSNR']:>10.4f} | {metrics['SSIM']:>10.4f} | {metrics['LPIPS']:>10.4f}")
                    else:
                        print(f"  {method_key:<30s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s}")
                print("  " + "-"*66)
            else:
                print(f"\n  No valid results for {scene_dir}")
            
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics evaluation script")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[], 
                        help="Path(s) to model directory/directories")
    parser.add_argument('--eval_train', action='store_true', 
                        help="Also evaluate train set (default: only test set)")
    args = parser.parse_args()
    
    print("="*70)
    print("Metrics Evaluation")
    print("="*70)
    print(f"Model paths: {args.model_paths}")
    print(f"Evaluate train set: {args.eval_train}")
    print("="*70)
    
    evaluate(args.model_paths, eval_train=args.eval_train)
