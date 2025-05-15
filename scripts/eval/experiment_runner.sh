#!/bin/bash

# ======= ./experiment_runner.sh ========
# For reproducibility and organizational purposes, this script is to be used for listing the experiments we want to run.
# For example, it can be used to facilitate grid search.

MODEL="/data/SceneUnderstanding/checkpoints/LLaVA-3D-7B/"
SCENES="/data/SceneUnderstanding/ScanNet/scans/"
EXP_DIR="/root/SceneUnderstanding/LLaVA-3D/experiments"
ANNO_DIR="/data/SceneUnderstanding/7792397/ScanQA_format"

# ~/SceneUnderstanding/LLaVA-3D/scripts/eval/sqa3d.sh \
#     --gpu 0 \
#     --model-path ${MODEL} \
#     --questions ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d.json \
#     --pred-answers ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers-rerun.json \
#     --gt-answers ${ANNO_DIR}/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
#     --video-folder ${SCENES} \
#     --chunk-idx 0 \
#     --num-chunks 1 \
#     --outfile ${EXP_DIR}/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_output-rerun.txt

~/SceneUnderstanding/LLaVA-3D/scripts/eval/sqa3d.sh \
    --gpu 0 \
    --model-path ${MODEL} \
    --questions ${ANNO_DIR}/scrap.json \
    --pred-answers ${EXP_DIR}/scrap.json \
    --gt-answers ${ANNO_DIR}/scrap_answers.json \
    --video-folder ${SCENES} \
    --outfile ${EXP_DIR}/SQA3D/em1_below_35/scrap.txt \
    --chunk-idx 0 \
    --num-chunks 1 \
    --generate echo 'scene0000_00' \


