# run from ~/SceneUnderstanding/LLaVA-3D/ as ./scripts/eval/sqa3d.sh

export CUDA_VISIBLE_DEVICES=0

# --- set the question file ---
unlink playground/data/annotations/llava-3d-sqa3d_test_question.json
#ln -s /data/SceneUnderstanding/7792397/ScanQA_format/SQA_650_formatted_LLaVa3d.json playground/data/annotations/llava-3d-sqa3d_test_question.json
ln -s /data/SceneUnderstanding/7792397/ScanQA_format/SQA_scene0000_00_formatted_LLaVa3d.json playground/data/annotations/llava-3d-sqa3d_test_question.json

# --- set the embodiedscan file ---
unlink playground/data/annotations/embodiedscan_infos.json
ln -s /root/SceneUnderstanding/LLaVA-3D/playground/data/annotations/embodiedscan_infos_full_formatted_cluster.json playground/data/annotations/embodiedscan_infos.json

# python llava/eval/model_sqa3d.py \
#         --model-path ChaimZhu/LLaVA-3D-7B \
#         --question-file playground/data/annotations/llava-3d-sqa3d_test_question.json \
#         --answers-file ./llava-3d-7b-sqa3d_test_answer.json \
#         --video-folder /data/SceneUnderstanding/ScanNet/scans/

python llava/eval/sqa3d_evaluator.py \
        --gt-json /data/SceneUnderstanding/7792397/ScanQA_format/SQA_scene0000_00_formatted_LLaVa3d_answers.json
