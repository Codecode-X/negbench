#!/bin/bash

# Set the base directory for data and logs. Users should update this to their directory structure.
BASE_DIR="/path/to/your/research/project"  # Change this to your base directory
DATA_DIR="$BASE_DIR/data"
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/models"

# Model and pretrained options
MODEL="ViT-B-32"
MODEL_NAME="mcq_third_batch_ViT-B-32_conclip_cc12m_full_lr1e-8_clw0.99_mlw0.01"
PRETRAINED_MODEL="$MODELS_DIR/$MODEL_NAME/checkpoints/epoch_1.pt"

# Dataset paths for images
COCO_MCQ="$DATA_DIR/images/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$DATA_DIR/images/VOC2007_mcq_llama3.1_rephrased.csv"
SYNTHETIC_MCQ="$DATA_DIR/images/synthetic_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$DATA_DIR/images/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$DATA_DIR/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
SYNTHETIC_RETRIEVAL="$DATA_DIR/images/synthetic_retrieval_v1.csv"
SYNTHETIC_NEGATED_RETRIEVAL="$DATA_DIR/images/synthetic_retrieval_v2.csv"

# Dataset paths for videos
MSRVTT_RETRIEVAL="$DATA_DIR/videos/MSRVTT/msr_vtt_retrieval.csv"
MSRVTT_NEGATED_RETRIEVAL="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_retrieval_rephrased_llama.csv"
MSRVTT_MCQ="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_mcq_rephrased_llama.csv"

# Activate the appropriate environment
source activate clip_negation || conda activate clip_negation

# Set system limits
ulimit -S -n 100000

# Logs directory for this evaluation run
RUN_LOGS_DIR="$LOGS_DIR/evaluation"
mkdir -p "$RUN_LOGS_DIR"

# Image Evaluation
echo "Starting Image Evaluation..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "image_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --report-to wandb \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="$DATA_DIR/images/imagenet" \
    --coco-mcq=$COCO_MCQ \
    --voc2007-mcq=$VOC_MCQ \
    --synthetic-mcq=$SYNTHETIC_MCQ \
    --coco-retrieval=$COCO_RETRIEVAL \
    --coco-negated-retrieval=$COCO_NEGATED_RETRIEVAL \
    --synthetic-retrieval=$SYNTHETIC_RETRIEVAL \
    --synthetic-negated-retrieval=$SYNTHETIC_NEGATED_RETRIEVAL \
    --batch-size=64 \
    --workers=8

# Video Evaluation
echo "Starting Video Evaluation..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model $MODEL \
    --pretrained $PRETRAINED_MODEL \
    --name "video_$MODEL_NAME" \
    --logs=$RUN_LOGS_DIR \
    --report-to wandb \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --zeroshot-frequency 1 \
    --imagenet-val="$DATA_DIR/images/imagenet" \
    --msrvtt-retrieval=$MSRVTT_RETRIEVAL \
    --msrvtt-negated-retrieval=$MSRVTT_NEGATED_RETRIEVAL \
    --msrvtt-mcq=$MSRVTT_MCQ \
    --video \
    --batch-size=64 \
    --workers=8

echo "Evaluation complete. Logs saved to $RUN_LOGS_DIR."
