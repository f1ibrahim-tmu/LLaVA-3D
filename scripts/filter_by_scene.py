"""
    This script will:
    - For each scene in a given txt file:
        - filter the ground truth file to only have answers of those scenes (output in the form of a list)
        - filter the predictions (answer file predicted by LLaVa) to only have answers of those scenes (output in the form of a list)
        - Pass those lists to sqa3d_evaluator's functions to compute the output directly
        - get SQA3D's evaluation and store it.
"""

import json
import argparse
import pdb

import sys
import os
import pandas as pd
from llava.eval.sqa3d_evaluator import *

question_to_scene = {}
scene_to_lists = {}
scores = []

def main(args):

    with open(args.scene_file, 'r', encoding='utf-8') as f:
        scene_file = f.read().splitlines()
    f.close()
    
    with open(args.pred_file, 'r') as f:
        pred_file = json.load(f)
    f.close()

    with open(args.gt_file, 'r') as f:
        gt_file = json.load(f)
    f.close()

    with open(args.sqa_formatted_file, 'r') as f:
        sqa_formatted_file = json.load(f)
    f.close()

    for q in sqa_formatted_file:
        question_to_scene[q['question_id']] = q['scene_id']
        scene_to_lists[q['scene_id']] = {'preds': [], 'gts': []}

    for p in pred_file:
        scene = question_to_scene[p['question_id']]
        scene_to_lists[scene]['preds'].append(p)

    for g in gt_file:
        scene = question_to_scene[g['question_id']]
        scene_to_lists[scene]['gts'].append(g)

    for scene in scene_file:
        length = len(scene_to_lists[scene]['preds'])
        preds = sorted(scene_to_lists[scene]['preds'], key=lambda d: d['question_id'])
        gts = sorted(scene_to_lists[scene]['gts'], key=lambda d: d['question_id'])
        scores.append([scene] + [length] + list(calc_sqa3d_score(preds, gts).values()))

    header = ["scene", "question_count"] + [x[x.index(']')+2:] for x in calc_sqa3d_score(preds, gts).keys()]

    df = pd.DataFrame(scores, columns=header)
    df = df.sort_values(by='EM1')
    df.index = range(1, len(df.index)+1)

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get scene-by-scene performance for SQA3D.")
    parser.add_argument("--scene_file", help="txt file of the scenes to be used; each line is one scene", default="/data/SceneUnderstanding/7792397/ScanQA_format/the_650_scenes.txt")
    parser.add_argument("--pred_file", help="prediction file (the answers/output from LLaVa3D)", default="/root/SceneUnderstanding/LLaVA-3D/experiments/SQA3D/650/llava-3d-7b-sqa3d_test_answer-650_formatted.json")
    parser.add_argument("--gt_file", help="the ground truth against which pred_file is compared", default="/data/SceneUnderstanding/7792397/ScanQA_format/SQA_650_formatted_LLaVa3d_answers.json")
    parser.add_argument("--sqa_formatted_file", help="file with both the scene and the question ids", default="/data/SceneUnderstanding/7792397/ScanQA_format/SQA_formatted.json")
    parser.add_argument("--output_file", help="Path to output txt file where the by-scene results will be.", default="/root/SceneUnderstanding/LLaVA-3D/experiments/SQA3D/650/llava-3d-7b-sqa3d_test_answer-650-byscene.csv")
    args = parser.parse_args()

    main(args)
