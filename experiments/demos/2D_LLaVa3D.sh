# The vanilla expermient from the LLaVA-3D repo

export CUDA_VISIBLE_DEVICES=0

python /root/SceneUnderstanding/LLaVA-3D/llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg \
    --query "What are the things I should be cautious about when I visit here?"
