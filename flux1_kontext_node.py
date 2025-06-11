import requests
import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import time


class FluxKontextProNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "input_image": ("IMAGE", {"forceInput": False, "default": None}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "16:21"], {"default": "3:4"}),
                "guidance": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max":2147483647}),
                "control_after_generate": (["randomize", "fixed"], {"default": "randomize"}),
                "prompt_upsampling": ("BOOLEAN", {"default": True}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "api_url": ("STRING", {"multiline": False, "default": "https://api.us1.bfl.ai/v1/flux-kontext-pro"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("IMAGE", "polling_url",)
    FUNCTION = "run"
    CATEGORY = "flux1-Kontext"

    def run(self, prompt, input_image, aspect_ratio, guidance, steps, seed, control_after_generate, prompt_upsampling, api_key, api_url):
        # --- API Key and Endpoint ---
        if not api_key or api_key == 'your_flux_api_key_here':
             raise ValueError(f"Please provide your actual API key via the node input.")

        # --- Prepare Request Payload ---
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "guidance": guidance,
            "steps": steps,
            "seed": seed,
            "control_after_generate": control_after_generate,
            "prompt_upsampling": prompt_upsampling,
            "output_format": "png", # 保持输出格式为png，方便api处理
        }

        # Convert input_image (if provided) to base64
        if input_image is not None:
            # ComfyUI's IMAGE tensor is (batch, height, width, channels)
            # Take the first image from the batch (assuming batch size is 1)
            i = input_image[0] # Shape is (height, width, channels) or (height, width)

            # If it's grayscale (height, width), add a channel dimension and repeat to simulate RGB
            if i.ndim == 2:
                i = i.unsqueeze(-1).repeat(1, 1, 3) # Shape becomes (height, width, 3)
            elif i.ndim != 3:
                 raise ValueError(f"Unexpected input image shape after removing batch dimension: {i.shape}")

            # Perform operations using torch methods
            # Scale to 0-255 and clip
            i = torch.clamp(i * 255., 0, 255)
            # Convert to uint8 tensor
            i = i.to(torch.uint8)

            # Convert to NumPy array for PIL
            # Use .cpu() to ensure the tensor is on the CPU before converting to NumPy
            i_np = i.cpu().numpy()

            # Create PIL Image from NumPy array
            img = Image.fromarray(i_np)

            # Save PIL Image to BytesIO and encode to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            payload["input_image"] = img_str
        else:
            payload["input_image"] = None


        headers = {
            "Content-Type": "application/json",
            "x-key": api_key
        }
        # print(f"Sending request to {api_url} with payload: {json.dumps(payload, indent=2)}")
        # --- Make Initial API Call ---
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            task_id = result.get("id")
            polling_url = result.get("polling_url")

            if not task_id or not polling_url:
                 raise ValueError(f"API response missing id or polling_url: {result}")

            # print(f"API call successful. Task ID: {task_id}, Polling URL: {polling_url}")

        except requests.exceptions.RequestException as e:
            # print(f"Initial API request failed: {e}")
            raise RuntimeError(f"Flux API initial request failed: {e}")
        except Exception as e:
            # print(f"An error occurred during initial API call: {e}")
            raise RuntimeError(f"An error occurred during initial API call: {e}")

        # --- Poll for Result ---
        timeout = 300
        poll_interval = 5
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                poll_response = requests.get(polling_url, headers=headers)
                poll_response.raise_for_status()
                poll_result = poll_response.json()
                # print(f"Polling {polling_url}: Status - {poll_result.get('status')}")

                status = poll_result.get("status")
                # Check for "Ready" status instead of "completed"
                if status == "Ready":
                    # Use "sample" key for result URL
                    result_data = poll_result.get("result")
                    if result_data and "sample" in result_data:
                        result_url = result_data["sample"]
                    else:
                         raise ValueError(f"Polling response missing 'sample' URL in 'result' on 'Ready' status: {poll_result}")

                    # print(f"Task Ready. Result URL (sample): {result_url}")

                    # --- Download and Load Image into ComfyUI format ---
                    # print(f"Downloading image from {result_url}")
                    image_response = requests.get(result_url)
                    image_response.raise_for_status()

                    # Load image using PIL
                    img = Image.open(BytesIO(image_response.content))

                    # Debugging: Print image info
                    # print(f"Downloaded image format: {img.format}, mode: {img.mode}, size: {img.size}")

                    # Convert to RGB to ensure consistent 3 channels before converting to numpy
                    # This handles potential grayscale or paletted images
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Convert PIL Image to NumPy array (H, W, C)
                    img_np = np.array(img).astype(np.float32) / 255.0

                    # Convert NumPy array to PyTorch tensor (H, W, C)
                    img_tensor = torch.from_numpy(img_np)

                    # Add batch dimension (1, H, W, C)
                    img_tensor = img_tensor.unsqueeze(0)

                    # Ensure image has 4 channels (RGBA) for ComfyUI consistency if it was RGB
                    # Although Save Image node usually handles 3 channels, adding alpha channel is good practice
                    if img_tensor.shape[-1] == 3:
                         alpha_channel = torch.ones(*img_tensor.shape[:-1], 1)
                         img_tensor = torch.cat([img_tensor, alpha_channel], dim=-1)

                    # 返回图像张量和polling_url
                    return (img_tensor, polling_url,)

                elif status in ["failed", "cancelled"]:
                     error_message = poll_result.get("error", "Unknown error")
                     raise RuntimeError(f"Task failed or cancelled. Status: {status}, Error: {error_message}")

                # If status is not "Ready", "failed", or "cancelled", continue polling
                time.sleep(poll_interval)

            except requests.exceptions.RequestException as e:
                # print(f"Polling request failed: {e}")
                # Continue polling even on request failure, maybe it's a temporary network issue
                time.sleep(poll_interval)
            except Exception as e:
                print(f"An error occurred during polling: {e}")
                # For other errors during polling, re-raise
                raise RuntimeError(f"An error occurred during polling: {e}")

        raise TimeoutError(f"Task did not complete within {timeout} seconds.")

# A dictionary that contains all nodes you want to export with their names
# NOTE: While this works, the preferred method is to use NODE_CLASS_MAPPINGS
# and NODE_DISPLAY_NAME_MAPPINGS in __init__.py
# NODE_CLASS_MAPPINGS = {
#     "FluxKontextPro": FluxKontextProNode
# }

# A dictionary that contains the friendly/humanly readable titles for the nodes
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FluxKontextPro": "Flux Kontext Pro"
# }
