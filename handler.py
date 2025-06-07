import os
import torch
from hyvideo.hunyuan import HunyuanVideoSampler
from datetime import datetime
import base64 # For potential future use
import io # For potential future use
import imageio # For saving video
import numpy as np
import random

# Global variable for the sampler
SAMPLER = None

# Paths for model components (defined globally for clarity, used in load_model)
MODEL_FILEPATH = "/app/ckpts/hunyuan_video_720_bf16.safetensors"
# Determine text_encoder_filepath based on quantization (empty means fp16)
TEXT_ENCODER_QUANTIZATION = "" # Default to fp16 as per Dockerfile download
if TEXT_ENCODER_QUANTIZATION == "int8":
    TEXT_ENCODER_FILEPATH = "/app/ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_int8.safetensors"
else:
    TEXT_ENCODER_FILEPATH = "/app/ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"

VAE_FILEPATH = "/app/ckpts/hunyuan_video_VAE_fp32.safetensors"
VAE_CONFIG_FILEPATH = "/app/ckpts/hunyuan_video_VAE_config.json"
TOKENIZER_PATH = "/app/ckpts/llava-llama-3-8b/"
CLIP_TEXT_ENCODER_FILEPATH = "/app/ckpts/clip_vit_large_patch14/" # This should be a directory
CLIP_TOKENIZER_PATH = "/app/ckpts/clip_vit_large_patch14/"


def load_model():
    """
    Loads the HunyuanVideoSampler model and moves it to the GPU.
    """
    global SAMPLER
    if SAMPLER is not None:
        return SAMPLER

    print("Loading HunyuanVideoSampler...")

    # Ensure clip_text_encoder_filepath points to the model file if necessary,
    # or the directory if from_pretrained handles it.
    # Based on diffusers, tokenizer_path and clip_tokenizer_path are directories.
    # clip_text_encoder_filepath should be the directory containing the model files (e.g., model.safetensors)

    sampler = HunyuanVideoSampler.from_pretrained(
        model_filepath=MODEL_FILEPATH,
        text_encoder_filepath=TEXT_ENCODER_FILEPATH,
        vae_filepath=VAE_FILEPATH,
        vae_config_filepath=VAE_CONFIG_FILEPATH,
        tokenizer_path=TOKENIZER_PATH,
        clip_text_encoder_filepath=CLIP_TEXT_ENCODER_FILEPATH, # Pass the directory
        clip_tokenizer_path=CLIP_TOKENIZER_PATH,
        dtype=torch.bfloat16, # As per model filename "bf16"
        VAE_dtype=torch.float32, # As per VAE filename "fp32"
        text_encoder_quantization=TEXT_ENCODER_QUANTIZATION,
        # Add any other necessary static parameters here
    )

    print("Moving sampler to CUDA device...")
    sampler.to(torch.device('cuda'))
    print("Model loaded and moved to CUDA.")
    return sampler

def save_video_frames(frames_tensor, output_path, fps):
    """
    Saves a tensor of video frames as an MP4 video file.
    Args:
        frames_tensor: A torch tensor of shape (T, H, W, C) or (T, C, H, W)
                       with pixel values in range [0, 1] or [-1, 1].
        output_path: Path to save the MP4 file.
        fps: Frames per second for the output video.
    """
    print(f"Saving video to {output_path} with FPS {fps}...")
    if frames_tensor.ndim == 4 and frames_tensor.shape[1] == 3: # T, C, H, W
        frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # To T, H, W, C

    # Normalize frames to [0, 1] if they are in [-1, 1]
    if frames_tensor.min() < 0:
        frames_tensor = (frames_tensor + 1) / 2

    # Convert to uint8 [0, 255]
    frames_uint8 = (frames_tensor.clamp(0, 1) * 255).byte().cpu().numpy()

    imageio.mimsave(output_path, frames_uint8, fps=fps, quality=8) # quality can be adjusted
    print(f"Video saved successfully to {output_path}")


def handler(job):
    """
    RunPod serverless handler function.
    """
    global SAMPLER

    try:
        if SAMPLER is None:
            SAMPLER = load_model()

        job_input = job.get('input', {})

        # Extract parameters
        prompt = job_input.get('prompt')
        if not prompt:
            return {"error": "Prompt is a required parameter."}

        height = job_input.get('height', 720)
        width = job_input.get('width', 1280)
        num_inference_steps = job_input.get('num_inference_steps', 30)
        guidance_scale = job_input.get('guidance_scale', 7.0)
        seed = job_input.get('seed', random.randint(0, 2**32 - 1))
        video_length = job_input.get('video_length', 97) # Number of frames
        fps = job_input.get('fps', 24)
        negative_prompt = job_input.get('negative_prompt', "")
        # 'shift' and 'embedded_guidance_scale' might be specific params for certain samplers/models
        # For HunyuanVideoSampler, check its generate() method signature
        # For now, let's assume they are part of a general 'extra_kwargs' or similar if supported
        # flow_shift = job_input.get('shift', 5.0) # Example, might not be directly used
        # embedded_guidance_scale = job_input.get('embedded_guidance_scale', 6.0) # Example

        print(f"Received job with prompt: '{prompt}'")
        print(f"Parameters: H={height}, W={width}, Steps={num_inference_steps}, Scale={guidance_scale}, Seed={seed}, Frames={video_length}, FPS={fps}")

        # Generate video frames
        # The generate method signature for HunyuanVideoSampler needs to be confirmed.
        # Assuming it takes these common parameters.
        # It might also need `device`, `dtype`, etc., but those are usually handled internally by the sampler instance.

        # Default parameters for HunyuanVideoSampler.generate, check hyvideo.hunyuan.py for exact signature
        # common ones: prompt, negative_prompt, height, width, frame_num, num_inference_steps, guidance_scale, seed
        # specific ones: VAE_tile_size, flow_shift, etc.

        # For `HunyuanVideoSampler`, `frame_num` is the parameter for video length.
        video_frames = SAMPLER.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            frame_num=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            # flow_shift=flow_shift, # If applicable
            # Add other specific parameters if the sampler expects them
            # e.g., VAE_tile_size=(256, 256) # A common default if tiling is used
        )

        print(f"Generated video frames tensor with shape: {video_frames.shape}")

        # Save the video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Ensure output directory exists if not /app root
        output_dir = "/app/output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"video_{timestamp}.mp4")

        save_video_frames(video_frames, output_filename, fps)

        # Optionally, if base64 is needed in the future:
        # with open(output_filename, "rb") as video_file:
        #     video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        # return {"video_base64": video_base64, "message": "Video generated successfully."}

        return {"video_path": output_filename, "message": "Video generated successfully."}

    except Exception as e:
        print(f"Error during handler execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == '__main__':
    # This block is for local testing if needed, RunPod calls handler() directly.
    print("Starting local test of handler...")

    # Mock SAMPLER for local testing if GPU is not available or models are not downloaded
    class MockSampler:
        def generate(self, **kwargs):
            print(f"MockSampler.generate called with: {kwargs}")
            # Return a dummy tensor (T, H, W, C) - e.g., 10 frames, 64x64, 3 channels
            # For save_video_frames to work, it expects values in [0,1] or [-1,1]
            # T, C, H, W format is also common from models
            # return torch.rand(kwargs.get('frame_num', 10), 3, kwargs.get('height',64)//8, kwargs.get('width',64)//8)
            # Let's try T, H, W, C for imageio direct compatibility
            return torch.rand(kwargs.get('frame_num', 10), kwargs.get('height',64)//8, kwargs.get('width',64)//8, 3)


        def to(self, device):
            print(f"MockSampler.to({device}) called.")
            return self

    if os.environ.get("RUNPOD_TEST_ENV"): # Simple flag to use mock
        SAMPLER = MockSampler()
        print("Using MockSampler for testing.")
    else:
        # Attempt to load real model if not in a specific test env
        # This will likely fail if models aren't present or no CUDA
        try:
            SAMPLER = load_model()
        except Exception as e:
            print(f"Failed to load real model for local test: {e}. Using MockSampler.")
            SAMPLER = MockSampler()


    test_job = {
        "input": {
            "prompt": "A beautiful sunset over the mountains",
            "height": 256, # Smaller for faster local test
            "width": 256, # Smaller for faster local test
            "video_length": 10, # Fewer frames
            "fps": 5,
            "num_inference_steps": 5, # Fewer steps
            "seed": 42
        }
    }
    result = handler(test_job)
    print(f"Handler result: {result}")

    # Example with error
    # test_job_error = {
    #     "input": {} # Missing prompt
    # }
    # result_error = handler(test_job_error)
    # print(f"Handler error result: {result_error}")
