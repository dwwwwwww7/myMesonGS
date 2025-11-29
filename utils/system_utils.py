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

import os
from errno import EEXIST
from os import makedirs, path


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder, prefer_best=True):
    """
    搜索最大迭代次数或 best 迭代
    
    Args:
        folder: point_cloud 目录路径
        prefer_best: 如果为 True，优先返回 'best' 迭代（如果存在）
    
    Returns:
        迭代标识（整数或字符串 'best'）
    """
    saved_iters = []
    has_best = False
    
    for fname in os.listdir(folder):
        if fname.startswith("iteration_"):
            iter_part = fname.split("_")[-1]
            if iter_part == "best":
                has_best = True
            else:
                try:
                    saved_iters.append(int(iter_part))
                except ValueError:
                    pass  # 忽略无法解析的目录名
    
    # 优先返回 best
    if prefer_best and has_best:
        return "best"
    
    # 否则返回最大数字迭代
    if saved_iters:
        return max(saved_iters)
    
    # 如果只有 best，返回 best
    if has_best:
        return "best"
    
    raise ValueError(f"No valid iteration found in {folder}")

MAIN_DIR='C:/Users/LENOVO/Desktop/RAHTGS/RAHT-from-MesonGS'
# MAIN_DIR='/home/szxie/mesongs_os'