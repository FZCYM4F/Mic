#!/bin/bash
#SBATCH --job-name=Step01_eval               # Job 名称
#SBATCH --output=logs/Step01_eval_%j.out     # 标准输出日志
#SBATCH --error=logs/Step01_eval_%j.err      # 标准错误日志
#SBATCH --time=10:00:00                      # 最长运行时间
#SBATCH --gres=gpu:1                         # 分配 1 块 GPU
#SBATCH --mem=32G                            # 分配内存
#SBATCH --cpus-per-task=4                    # 分配 CPU 数量
#SBATCH --partition=gpu                      # 使用 GPU 队列
#SBATCH --nodes=1

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CFLAGS="-std=gnu99"

# 初始化环境
source ~/.bashrc
conda activate mamba                          # 启动你已有的 conda 环境

# 输出基本信息
echo "Job started on $(date)"
echo "Running on $HOSTNAME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# 执行任务
python eval.py

echo "Job finished on $(date)"
