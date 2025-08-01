import re
import os
from collections import defaultdict
import json
import csv
import numpy as np
from tqdm import tqdm
import mmengine
import argparse
import string # Added for punctuation
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    from model_sqa3d import get_chunk, split_list
except ModuleNotFoundError:
    from llava.eval.model_sqa3d import get_chunk, split_list

import pdb

# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub(r'[ ]+$' ,'', data)
    data = re.sub(r'^[ ]+' ,'', data)
    data = re.sub(r' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    #data = re.sub(r'[^a-zA-Z0-9,\'\\s\\-:]+', '', data)
    data = re.sub(r'[^a-zA-Z0-9,\'\\s:-]+', '', data)
    data = re.sub('ç' ,'c', data)
    #data = re.sub('' ,'\'', data) # Corrected: replaces smart quote with standard apostrophe
    data = re.sub('’', '\'', data)
    data = re.sub(r'\\bletf\\b' ,'left', data)
    data = re.sub(r'\\blet\\b' ,'left', data)
    data = re.sub(r'\\btehre\\b' ,'there', data)
    data = re.sub(r'\\brigth\\b' ,'right', data)
    data = re.sub(r'\\brght\\b' ,'right', data)
    data = re.sub(r'\\bbehine\\b', 'behind', data)
    data = re.sub(r'\\btv\\b' ,'TV', data)
    data = re.sub(r'\\bchai\\b' ,'chair', data)
    data = re.sub(r'\\bwasing\\b' ,'washing', data)
    data = re.sub(r'\\bwaslked\\b' ,'walked', data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)

    # digit to word, only for answer
    data = re.sub(r'\\b0\\b', 'zero', data)
    data = re.sub(r'\\bnone\\b', 'zero', data)
    data = re.sub(r'\\b1\\b', 'one', data)
    data = re.sub(r'\\b2\\b', 'two', data)
    data = re.sub(r'\\b3\\b', 'three', data)
    data = re.sub(r'\\b4\\b', 'four', data)
    data = re.sub(r'\\b5\\b', 'five', data)
    data = re.sub(r'\\b6\\b', 'six', data)
    data = re.sub(r'\\b7\\b', 'seven', data)
    data = re.sub(r'\\b8\\b', 'eight', data)
    data = re.sub(r'\\b9\\b', 'nine', data)
    data = re.sub(r'\\b10\\b', 'ten', data)
    data = re.sub(r'\\b11\\b', 'eleven', data)
    data = re.sub(r'\\b12\\b', 'twelve', data)
    data = re.sub(r'\\b13\\b', 'thirteen', data)
    data = re.sub(r'\\b14\\b', 'fourteen', data)
    data = re.sub(r'\\b15\\b', 'fifteen', data)
    data = re.sub(r'\\b16\\b', 'sixteen', data)
    data = re.sub(r'\\b17\\b', 'seventeen', data)
    data = re.sub(r'\\b18\\b', 'eighteen', data)
    data = re.sub(r'\\b19\\b', 'nineteen', data)
    data = re.sub(r'\\b20\\b', 'twenty', data)
    data = re.sub(r'\\b23\\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\\b([a-zA-Z]+)([0-9])\\b' ,'\\g<1>', data)
    # These article removals are specific; the new function has a more general approach.
    data = re.sub(r'\\ba\\b ([a-zA-Z]+)' ,'\\g<1>', data)
    data = re.sub(r'\\ban\\b ([a-zA-Z]+)' ,'\\g<1>', data)
    data = re.sub(r'\\bthe\\b ([a-zA-Z]+)' ,'\\g<1>', data)

    data = re.sub(r'\\bbackwards\\b', 'backward', data)

    return data

_ARTICLES = ['a', 'an', 'the']
_NUMERALS_WORD_TO_DIGIT = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16',
    'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    # Add more if common, e.g., thirty, forty, etc., or "oh" for "0"
}

def normalize_text_for_em(text: str) -> str:
    """Lower-cases, strips articles, removes ALL punctuation, and canonicalizes numerals (word to digit)."""
    text = text.lower()

    # Remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize and remove articles
    words = text.split()
    words_no_articles = [word for word in words if word not in _ARTICLES]
    
    # Canonicalize numerals (word to digit) in the new list of words
    processed_words = []
    for word in words_no_articles:
        processed_words.append(_NUMERALS_WORD_TO_DIGIT.get(word, word))
    
    text = ' '.join(processed_words)
    
    # Remove extra whitespace that might have been introduced
    text = re.sub(r'\\s+', ' ', text).strip()
    return text


def evaluate_answer(question: str, pred_answer: str, gt_answer: str) -> int:
    prompt = (
        "You are an expert evaluator. Given the question, the answer predicted by an LLM, "
        "and the ground-truth answer, decide whether the predicted answer is sufficiently "
        "close or equivalent to the ground truth; this may include ignoring lots of repeated substrings. " 
        "If they match in meaning or value, respond with the single character \"1\". "
        "Otherwise, respond with \"0\". Do NOT provide any additional text.\n\n"
        f"Question: \"{question}\"\n"
        f"Predicted Answer: \"{pred_answer}\"\n"
        f"Ground Truth: \"{gt_answer}\"\n"
    )

    # 2. New call using client.chat.completions.create(...)
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )

    # 3. Extract and convert to boolean
    reply = response.choices[0].message.content.strip()
    #breakpoint()
    return gt_answer if (bool(int(reply)) if reply.isdigit() else False) else pred_answer


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/evaluator/scanqa_eval.py#L41-L50
def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        # Refined EM: check if prediction is a substring of ground truth or vice versa, after removing spaces
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0

def calc_sqa3d_score(preds, gts, use_enhanced_normalization=False, use_openai_evaluation=False):
    val_scores = {}
    metrics = {
        'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
        'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
    }
    em_overall = 0
    em_refined_overall = 0
    em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    print("Total samples:", len(preds))

    assert len(preds) == len(gts)
    for pred, gt in tqdm(zip(preds, gts)):
        question_id = pred['question_id']
        gt_question_id = gt['question_id']
        assert question_id == gt_question_id
        prompt = pred['prompt']

        
        pred_answer_orig = pred['text']
        gt_answer_orig = gt['text']

        #breakpoint()
        
        if use_enhanced_normalization:
            pred_answer = normalize_text_for_em(pred_answer_orig)
            gt_answers = [normalize_text_for_em(gt_answer_orig)]
        else:
            pred_answer = clean_answer(pred_answer_orig)
            gt_answers = [clean_answer(gt_answer_orig)]

        if use_openai_evaluation:
            # Use OpenAI API to evaluate the answer
            pred_answer = evaluate_answer(prompt, pred_answer, gt_answers[0])
            
        print('pred_answer:', pred_answer, 'gt_answers:', gt_answers[0])
        em_flag, em_refined_flag = answer_match(pred_answer, gt_answers)
        em_overall += em_flag
        em_refined_overall += em_refined_flag
        sqa_type = int(gt['type'])
        em_type[sqa_type] += em_flag
        em_refined_type[sqa_type] += em_refined_flag
        metrics[f'type{sqa_type}_count'] += 1
        
    em_overall = em_overall / len(preds) if len(preds) > 0 else 0
    em_refined_overall = em_refined_overall / len(preds) if len(preds) > 0 else 0
    
    val_scores["[sqa3d] EM1"] = em_overall
    val_scores["[sqa3d] EM1_refined"] = em_refined_overall
    for key in em_type.keys():
        count = metrics[f'type{key}_count']
        # Ensure count is treated as float for division, and handle if it's the initial 1e-10
        actual_count = count if count > 1 else (1 if count == 1 else 0) # Avoid division by 1e-10 if no items of type
        if actual_count > 0:
             val_scores[f'[sqa3d] EM_type{key}'] = em_type[key] / actual_count
             val_scores[f'[sqa3d] EM_refined_type{key}'] = em_refined_type[key] / actual_count
        else:
             val_scores[f'[sqa3d] EM_type{key}'] = 0
             val_scores[f'[sqa3d] EM_refined_type{key}'] = 0
             
    return val_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", type=str, nargs='+', default='llava-3d-7b-sqa3d_test_answer.json', help="path(s) to 1+ jsons of LLaVa-3D predictions")
    parser.add_argument("--gt-json", type=str, default='playground/data/annotations/llava3d_sqa3d_test_answer.json', help="path to the ground truth json")
    parser.add_argument("--num-chunks", type=int, default=1, help="number of chunks that the predictions were partitioned into (for distributed inference)")
    parser.add_argument("--chunk-idx", type=int, nargs='+', default=0, help="list of 1+ chunk indices, representing the index of each args.pred_json entry; each should be in range [0, args.pred_json], in the same order of the args.pred_json files")
    parser.add_argument("--use_enhanced_normalization", action='store_true', help="Use enhanced text normalization (strips all punctuation, articles, canonicalizes numerals to digits) for EM calculation.")
    parser.add_argument("--use_openai_evaluation", action='store_true', help="Use OpenAI API to evaluate the predicted answers against ground truth answers.")
    args = parser.parse_args()

    assert len(args.pred_json) == len(args.chunk_idx)

    print(args)

    preds_list = []
    for pred_json_path in args.pred_json:
        if not os.path.exists(pred_json_path):
            print(f"Warning: Prediction file not found: {pred_json_path}")
            continue
        with open(pred_json_path, 'r') as f:
            try:
                preds_list.extend(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {pred_json_path}")
                continue
    
    if not os.path.exists(args.gt_json):
        print(f"Error: Ground truth file not found: {args.gt_json}")
        exit(1)
        
    gt_all = mmengine.load(args.gt_json)
    gts_list = []
    
    # If num_chunks is 1, we expect all GTs. Otherwise, select based on chunk_idx.
    if args.num_chunks == 1 and args.chunk_idx == [0]: # Or simply args.chunk_idx == [0] if default is [0]
        gts_list = gt_all
    elif args.num_chunks > 1 :
        for chunk in args.chunk_idx:
            if not (0 <= chunk < args.num_chunks):
                 print(f"Error: chunk_idx {chunk} is out of range for num_chunks {args.num_chunks}")
                 exit(1)
            gts_list.extend(get_chunk(gt_all, args.num_chunks, chunk))
    else: # num_chunks is 1, but chunk_idx might be something else or not properly handled.
          # This case might indicate an issue or a specific setup, assuming all GTs if not chunking.
        gts_list = gt_all


    if not preds_list:
        print("No predictions loaded. Exiting.")
        exit(1)
    if not gts_list:
        print("No ground truth data loaded. Exiting.")
        exit(1)

    # To ensure `calc_sqa3d_score` receives lists of same-type items (dicts)
    # and to handle potential variations in how GT items are structured if get_chunk returns a list of lists sometimes.
    # For SQA3D, gt_all is typically a list of dicts. get_chunk should return a slice (list of dicts).
    # This ensures consistency.
    
    final_preds = [p for p in preds_list if isinstance(p, dict) and 'question_id' in p and 'text' in p]
    final_gts = [g for g in gts_list if isinstance(g, dict) and 'question_id' in g and 'text' in g and 'type' in g]

    # Align preds and gts by question_id to be absolutely sure, though typically they are already aligned
    gt_map = {gt['question_id']: gt for gt in final_gts}
    aligned_preds = []
    aligned_gts = []

    for pred_item in final_preds:
        qid = pred_item['question_id']
        if qid in gt_map:
            aligned_preds.append(pred_item)
            aligned_gts.append(gt_map[qid])
        else:
            print(f"Warning: Question ID {qid} from predictions not found in ground truth. Skipping.")
            
    if not aligned_preds:
        print("No matching predictions and ground truths after alignment. Exiting.")
        exit(1)

    #breakpoint()

    val_scores = calc_sqa3d_score(aligned_preds, aligned_gts, args.use_enhanced_normalization, args.use_openai_evaluation)
    print(json.dumps(val_scores, indent=2))
