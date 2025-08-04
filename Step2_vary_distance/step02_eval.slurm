#!/bin/bash
#SBATCH --job-name=Step02_eval               # Job ����
#SBATCH --output=logs/Step02_eval_%j.out     # ��׼�����־
#SBATCH --error=logs/Step02_eval_%j.err      # ��׼������־
#SBATCH --time=50:00:00                      # �����ʱ��
#SBATCH --gres=gpu:1                         # ���� 1 �� GPU
#SBATCH --mem=32G                            # �����ڴ�
#SBATCH --cpus-per-task=4                    # ���� CPU ����
#SBATCH --partition=gpu                      # ʹ�� GPU ����
#SBATCH --nodes=1

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CFLAGS="-std=gnu99"

# ��ʼ������
source ~/.bashrc
conda activate mamba                          # ���������е� conda ����

# ���������Ϣ
echo "Job started on $(date)"
echo "Running on $HOSTNAME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# ִ������
python eval.py

echo "Job finished on $(date)"
