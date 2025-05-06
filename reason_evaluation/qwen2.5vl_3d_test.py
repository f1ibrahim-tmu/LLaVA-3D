from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch


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


# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept.
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9
def main(image_paths: str, text_prompt: str, model_path: str):
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
                        # "resized_height": 280,
                        # "resized_width": 280,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
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
        device_map="cuda:2",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
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
    print(output_text)


if __name__ == "__main__":
    # Load images from a folder
    folder_path = "/root/research_projects/LLaVA-3D/demo/scannet/posed_images/scene0356_00"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    regular_images, depth_images = load_images(folder_path)
    text_prompt = "Tell me the only object that I could see from the other room and describe the object."
    main(regular_images, text_prompt, model_path)
