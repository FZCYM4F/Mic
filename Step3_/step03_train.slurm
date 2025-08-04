#!/bin/bash
#SBATCH --job-name=Step03_train
#SBATCH --output=logs/Step03_train_%j.out
#SBATCH --error=logs/Step03_train_%j.err
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:2                   # 使用 2 块 GPU
#SBATCH --ntasks=2                     # 启动两个任务（每 GPU 一个进程）
#SBATCH --cpus-per-task=4              # 每个进程分配 4 个 CPU
#SBATCH --mem=32G                      # 分配 32G CPU 内存
#SBATCH --partition=gpu
#SBATCH --nodes=1

# 设置编译器环境变量（如果用到了 Triton JIT 编译）
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CFLAGS="-std=gnu99"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 初始化环境
source ~/.bashrc
conda activate mamba

# 输出基本信息
echo "Job started on $(date)"
echo "Running on $HOSTNAME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# 启动多进程分布式训练
torchrun --nproc_per_node=2 train.py

echo "Job finished on $(date)"
