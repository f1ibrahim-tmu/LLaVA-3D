import os
import glob
import json
import base64
import requests
import subprocess

# —— Configuration ——  
API_KEY   = subprocess.run("bw get password OAI_API_KEY_neovim --session $BW_SESSION", shell=True, capture_output=True, text=True).stdout
API_URL   = "https://api.openai.com/v1/chat/completions"
MODEL     = "o4-mini-2025-04-16"
ANNOTATION_DIR = "/data/SceneUnderstanding/7792397/ScanQA_format/"
SCENES    = {
    "A": {"id": "scene0307_00", "questions": os.path.join(ANNOTATION_DIR, "SQA_scene0307_00_formatted_LLaVa3d.json")},
    "B": {"id": "scene0364_00", "questions": os.path.join(ANNOTATION_DIR, "SQA_scene0364_00_formatted_LLaVa3d.json")}
}
DATA_DIR  = "/data/SceneUnderstanding/ScanNet/scans/"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# —— Helper Functions ——  
def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # Load JSON object  [oai_citation:8‡Stack Overflow Blog](https://stackoverflow.blog/2022/06/02/a-beginners-guide-to-json-the-data-format-for-the-internet/?utm_source=chatgpt.com)

def encode_image(path):
    data = open(path, "rb").read()
    return base64.b64encode(data).decode()

def sample_frames(scene_id, step=30):
    pattern = os.path.join(DATA_DIR, scene_id, scene_id + "_sens", "color", "*.jpg")
    files = sorted(glob.glob(pattern))
    return files[::step]

def build_message_block(scene_label, frames, questions_path):
    # 1) Load and serialize questions  [oai_citation:9‡Stack Overflow](https://stackoverflow.com/questions/77397666/how-to-include-json-data-into-prompt-api?utm_source=chatgpt.com)
    questions = load_questions(questions_path)
    q_str = json.dumps(questions, separators=(",", ":"))
    
    # 2) Start block with JSON and instructions  [oai_citation:10‡Stack Overflow](https://stackoverflow.com/questions/76185628/how-to-prompt-chatgpt-api-to-give-completely-machine-readable-responses-without/76751441?utm_source=chatgpt.com)
    content = [
        {"type": "text", "text": f"Scene {scene_label} ID: {SCENES[scene_label]['id']}."},
        {"type": "text", "text": "Here are the full question JSON for this scene:"},
        {"type": "text", "text": q_str}
    ]
    
    # 3) Append sampled frames  [oai_citation:11‡OpenAI Platform](https://platform.openai.com/docs/guides/text?utm_source=chatgpt.com)
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}
        })
    return content

def analyze_and_compare():
    # Build messages for each scene  [oai_citation:12‡Gist](https://gist.github.com/pszemraj/c643cfe422d3769fd13b97729cf517c5?utm_source=chatgpt.com)
    msgs = []
    for label in ["A", "B"]:
        frames = sample_frames(SCENES[label]["id"])
        block = build_message_block(label, frames, SCENES[label]["questions"])
        msgs.append({"role": "user", "content": block})

    # Final comparative prompt  [oai_citation:13‡OpenAI Community](https://community.openai.com/t/how-do-i-call-chatgpt-api-with-python-code/554554?utm_source=chatgpt.com)
    comparison_text = [
        {"type": "text", "text": "Based on the visual frames and the question sets, explain why scene 0364_00 achieved EM@1=78% whereas scene 0307_00 only achieved EM@1=17%."}
    ]
    msgs.append({"role": "user", "content": comparison_text})

    payload = {
        "model": MODEL,
        "messages": msgs,
        "max_completion_tokens": 10000
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload).json()
    return response #response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    result = analyze_and_compare()
    print("=== Comparative Analysis ===\n", result)



"""
=== Comparative Analysis ===
 {'id': 'chatcmpl-BV0gHjBVCLzCr3aWKkOn3265Vh1Ui', 'object': 'chat.completion', 'created': 1746730485, 'model': 'o4-mini-2025-04-16', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Scene 0364_00 is essentially a “toy” scene with only a handful of large, static, easily‐recognized objects (a desk, a chair, a rad
iator, a window, a bed and a door) seen from very similar viewpoints, and the questions all collapse to a very small answer vocabulary and very simple spatial or parity judgments (“Which way to the door?”, “Is 2 even or odd?”, “What’s on my right – the desk or the bed?”, etc.).  In other words, there are only 3–4 objects to keep tr
ack of, the camera moves very little, and almost every question reduces to the same half-dozen answers.\n\nBy contrast, scene 0307_00 is a cluttered laundry/basement utility room filled with washing machines, dryers, drying racks, tables, multiple cabinets and shelves, a shower, a sink, water heaters, detergent boxes, chairs, windo
ws, radiators, etc., all viewed from constantly shifting angles.  The questions demand (1) identifying dozens of distinct object types, (2) constantly updating a 360° spatial map, (3) comparing colors and shapes, (4) giving directional navigation (“turn right,” “back up”), and even (5) applying real-world procedural knowledge (“wha
t to add so clothes come out smelling nice”).  All of those factors explode the answer space, introduce ambiguities and occlusions, and push well beyond the kind of template‐like questions that modern VQA models have learned to handle reliably.\n\nPut simply:\n • Scene 0364_00: very few, large, unambiguous objects + repetitive, tri
vial spatial/counting questions ⇒ EM@1≈78%.  \n • Scene 0307_00: many small/cluttered objects + complex spatial, color, and knowledge‐based queries ⇒ EM@1≈17%.', 'refusal': None, 'annotations': []}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 257478, 'completion_tokens': 1101, 'total_tokens': 258579, 'prompt_tokens_detail
s': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 704, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'service_tier': 'default', 'system_fingerprint': None}
"""
