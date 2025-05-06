#!/bin/bash

# =========================
# SLURM Job Submission Script for Pretrained Model Evaluations
# =========================

# 目录配置（请根据你的实际路径替换）
BASE_DIR="123"  # 修改为你的项目根目录
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/models"

# 确保日志目录存在
mkdir -p "$LOGS_DIR"

# SLURM配置
SBATCH_PARTITION="gpu_partition"  # 修改为你集群的GPU分区名称
SBATCH_GPUS=1
SBATCH_CPUS=16
SBATCH_MEM="20gb"
SBATCH_TIME="2:00:00"  # 可根据需要调整时间限制

# 是否是视频评估（true/false）
video=false

# 实验类型："main" 或 "template"
experiment_type="main"

# 根据是否是视频评估选择脚本
if [ "$video" = true ]; then
    sbatch_script="evaluate_videos.sh"
    echo "Submitting video evaluations..."
else
    sbatch_script="evaluate_images.sh"
    echo "Submitting image evaluations..."
fi

# 模型与预训练方案
models=("ViT-B-32" "ViT-L-14")
pretrained_options=("openai" "datacomp_xl_s13b_b90k" "laion400m_e31")

# 提交任务
for model in "${models[@]}"; do
    for pretrained in "${pretrained_options[@]}"; do
        echo "Submitting job: Model=$model, Pretrained=$pretrained, Experiment=$experiment_type"

        sbatch --job-name="eval_${model}_${pretrained}_${experiment_type}" \
               --partition="$SBATCH_PARTITION" \
               --gres=gpu:$SBATCH_GPUS \
               --cpus-per-task=$SBATCH_CPUS \
               --mem=$SBATCH_MEM \
               --time=$SBATCH_TIME \
               --output="$LOGS_DIR/${model}_${pretrained}_${experiment_type}_%j.out" \
               --error="$LOGS_DIR/${model}_${pretrained}_${experiment_type}_%j.err" \
               --export=ALL,model="$model",pretrained="$pretrained",experiment_type="$experiment_type" \
               "$sbatch_script"
    done
done
