#!/bin/bash

# Go to the directory containing this script
cd "$(dirname "$0")" || exit

# Set PYTHONPATH to benchmarks directory
export PYTHONPATH=$(cd .. && pwd)
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=$PYTHONPATH:/root/NP-CLIP/negbench/benchmarks/src


echo "PYTHONPATH set to $PYTHONPATH"

# Set the base directory for data and logs
BASE_DIR="/root/NegBench"  # 修改为你的工作目录
DATA_DIR="/root/NegBench/data"  # 修改为你数据的根目录
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="/root/NP-CLIP/XTrainer/~/.cache/clip/"  # 修改为模型目录checkpoint.pt

# Model and pretrained options
MODEL="ViT-B-32"
MODEL_NAME="NegCLIP"
# PRETRAINED_MODEL="$MODELS_DIR/ViT-B-32.pt" 
# PRETRAINED_MODEL="$MODELS_DIR/checkpoint.pt"  # 55.97 tp5:83.27%
PRETRAINED_MODEL="$MODELS_DIR/conclip_b32_openclip_version.pt" # ImageNet:51.38% caltech101:90.01% CIFAR-10:87.78% CIFAR-100:62.10%

# Dataset paths for images
COCO_MCQ="$DATA_DIR/images/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$DATA_DIR/images/VOC2007_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$DATA_DIR/images/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$DATA_DIR/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"

# # Dataset paths for videos
# MSRVTT_RETRIEVAL="$DATA_DIR/videos/MSRVTT/msr_vtt_retrieval.csv"
# MSRVTT_NEGATED_RETRIEVAL="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_retrieval_rephrased_llama.csv"
# MSRVTT_MCQ="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_mcq_rephrased_llama.csv"

# # Activate the appropriate environment
# source activate clip_negation || conda activate clip_negation

# Set system limits
ulimit -S -n 100000

# Logs directory
RUN_LOGS_DIR="$LOGS_DIR/evaluation"
mkdir -p "$RUN_LOGS_DIR"

cd "$BASE_DIR" || exit

# Image Evaluation
echo "Starting Image Evaluation..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "image_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="/root/autodl-tmp/imagenet/val" \
    # --val-data="/root/ConCLIP/Negbench/negbench/cifar10.csv" \
    # --coco-mcq=$COCO_MCQ \
    # --voc2007-mcq=$VOC_MCQ \
    # --coco-retrieval=$COCO_RETRIEVAL \
    # --coco-negated-retrieval=$COCO_NEGATED_RETRIEVAL \
    # --batch-size=64 \
    # --workers=8


# CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
#     --model $MODEL \
#     --pretrained $PRETRAINED_MODEL \
#     --name "imagenet_$MODEL_NAME" \
#     --logs=$RUN_LOGS_DIR \
#     --dataset-type csv \
#     --csv-separator=, \
#     --csv-img-key filepath \
#     --csv-caption-key caption \
#     --zeroshot-frequency 1 \
#     --imagenet-val="/path/to/imagenet/val" \
#     --batch-size=64 \
#     --workers=8
#     --eval


# # Video Evaluation
# echo "Starting Video Evaluation..."
# CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
#     --model $MODEL \
#     --pretrained $PRETRAINED_MODEL \
#     --name "video_$MODEL_NAME" \
#     --logs=$RUN_LOGS_DIR \
#     --dataset-type csv \
#     --csv-separator=, \
#     --csv-img-key filepath \
#     --csv-caption-key caption \
#     --zeroshot-frequency 1 \
#     --imagenet-val="$DATA_DIR/images/imagenet" \
#     --msrvtt-retrieval=$MSRVTT_RETRIEVAL \
#     --msrvtt-negated-retrieval=$MSRVTT_NEGATED_RETRIEVAL \
#     --msrvtt-mcq=$MSRVTT_MCQ \
#     --video \
#     --batch-size=64 \
#     --workers=8

echo "Evaluation complete. Logs saved to $RUN_LOGS_DIR."
