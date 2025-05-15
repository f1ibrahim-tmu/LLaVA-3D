# run from ~/SceneUnderstanding/LLaVA-3D/ as ./scripts/eval/sqa3d.sh

export CUDA_VISIBLE_DEVICES=0

# --- set the pred_answers file ---
PRED_ANSWERS="/root/SceneUnderstanding/LLaVA-3D/experiments/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_pred-answers.json"

# --- set the question file ---
unlink playground/data/annotations/llava-3d-sqa3d_test_question.json
ln -s /data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d.json playground/data/annotations/llava-3d-sqa3d_test_question.json

# --- set the embodiedscan file ---
unlink playground/data/annotations/embodiedscan_infos.json
ln -s /root/SceneUnderstanding/LLaVA-3D/playground/data/annotations/embodiedscan_infos_full_formatted_cluster.json playground/data/annotations/embodiedscan_infos.json

# TODO: can I make this file only work on a partition of the file, for splitting? I think using the set_chunks setting, we can do this.
python llava/eval/model_sqa3d.py \
        --model-path /data/SceneUnderstanding/checkpoints/LLaVA-3D-7B/ \
        --question-file playground/data/annotations/llava-3d-sqa3d_test_question.json \
        --answers-file ${PRED_ANSWERS} \
        --video-folder /data/SceneUnderstanding/ScanNet/scans/ \
        --chunk-idx 0 \
        --num-chunks 1

# TODO: make this output to a text file
python llava/eval/sqa3d_evaluator.py \
        --pred-json ${PRED_ANSWERS} \
        --gt-json /data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d_answers.json \
        --chunk-idx 0 \
        --num-chunks 1 \
        > /root/SceneUnderstanding/LLaVA-3D/experiments/SQA3D/em1_below_35/SQA_em1-below-35_formatted_LLaVa3d_output-rerun.txt
