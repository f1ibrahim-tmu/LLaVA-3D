from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
import json
import gc

def load_images(folder_path: str):
    """
    Load images from a folder.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: List of loaded images.
    """
    regular_images = []
    depth_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            regular_images.append(img_path)
        elif filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            depth_images.append(img_path)
    return regular_images, depth_images


def load_questions_json(file_path: str):
    """
    Load questions from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: List of questions.
    """
    with open(file_path, "r") as f:
        questions = json.load(f)
    return questions


def add_answers_to_questions(questions: list, answer_str: str):
    """
    Add answers to questions.
    Args:
        questions (list): List of questions.
        answer_str (str): Answer string to add.
    Returns:
        list: List of questions with answers.
    """

    # load answer from json file
    with open(answer_str, "r") as f:
        answers = json.load(f)
    for i, question in enumerate(questions):
        question["answer"] = answers[i]["text"]
        question["type"] = answers[i]["type"]
    return questions


def find_image_paths(questions: list, folder_path: str, sample_rate: int = 1):
    """
    Find image paths in questions.
    Args:
        questions (list): List of questions.
        folder_path (str): Path to the folder containing images.
        sample_rate (int): Sample rate for images.
    Returns:
        list: questions List with image paths.
    """
    image_paths = []
    for question in questions:
        scene_name = question["video"]
        scene_folder_path = os.path.join(folder_path, scene_name, scene_name+"_sens", "color")
        # add all jpg files in the folder to the image_paths list
        count = 0
        for filename in os.listdir(scene_folder_path):
            if filename.endswith(".jpg"):
                count += 1
                if count % sample_rate != 0:
                    continue
                img_path = os.path.join(scene_folder_path, filename)
                image_paths.append(img_path)
        question["scene_images_path"] = image_paths
    return questions


# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept.
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9
def qwen_video_test(image_paths: list, text_prompt: str, model_path: str, device: str = "cuda:2"):
    # You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
    messages = [
        # Image
        ## Local file path
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "file:///path/to/your/image.jpg"},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        ## Image URL
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## Base64 encoded image
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": "data:image;base64,/9j/..."},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## PIL.Image.Image
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": pil_image},
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # ## Model dynamically adjusts image size, specify dimensions if required.
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": "file:///path/to/your/image.jpg",
        #                 "resized_height": 280,
        #                 "resized_width": 420,
        #             },
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ],
        # Video
        # Local video frames
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": image_paths,
                        "resized_height": 280,
                        "resized_width": 280,
                        # "fps": 30.0,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a help assistant to answer the question with your separate reasoning trace.",
                    }
                ],
            },
        ],
        ## Model dynamically adjusts video nframes, video height and width. specify args if required.
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": "file:///path/to/video1.mp4",
        #                 "fps": 2.0,
        #                 "resized_height": 280,
        #                 "resized_width": 280,
        #             },
        #             {"type": "text", "text": "Describe this video."},
        #         ],
        #     }
        # ],
    ]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    with torch.no_grad():
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        # Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    gc.collect()
    torch.cuda.empty_cache()
    return output_text


def save_output(output_text: str, question: dict, output_file_path:str):
    # Parse the output text to extract the reasoning and answer
    # This is a placeholder implementation. You may need to adjust it based on the actual output format.
    try:
        output_json = json.loads(output_text)
        reasoning = output_json.get("reason", "")
        answer = output_json.get("answer", "")
        # append this new json enry to the output file
        with open(output_file_path, "a") as f:
            json.dump(
                {
                    "reason": reasoning,
                    "answer": answer,
                    "question_id": question["question_id"],
                    "scene_name": question["video"],
                    "question": question["text"],
                },
                f,
            )
            f.write("\n")
    except json.JSONDecodeError:
        print(f"Failed to parse output: {output_text}")


def main(question_file_path:str, answer_file_path:str, image_folder_path:str, export_json_path:str, model_path:str):
    # Load questions from a JSON file
    questions = load_questions_json(question_file_path)
    # Load answers from a JSON file
    questions = add_answers_to_questions(questions, answer_file_path)
    # Find image paths in questions
    questions = find_image_paths(questions, image_folder_path, 500)

    for question in questions:
        # print(question)
        # Get the image paths for the first question
        image_paths = question["scene_images_path"]
        # Get the text prompt for the first question
        question_text = question["text"]
        # Get text prompt from the question
        text_prompt = question_text + " Please reason step by step, and give your reason and answer in the json format:{\"reason\": \"\", \"answer\": \"\"}."
        # Run the Qwen video test
        output_text = qwen_video_test(image_paths, text_prompt, model_path)
        save_output(output_text, question, export_json_path)


if __name__ == "__main__":
    # Load images from a folder
    question_file_path = "/data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d.json"
    answer_file_path = "/data/SceneUnderstanding/7792397/ScanQA_format/SQA_em1-below-35_formatted_LLaVa3d_answers.json"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    image_folder_path = "/data/SceneUnderstanding/ScanNet/scans"
    export_path = "/root/research_projects/LLaVA-3D/reason_evaluation/qwen2.5vl_3d_test_results.json"
    main(question_file_path, answer_file_path, image_folder_path, export_path, model_path)
