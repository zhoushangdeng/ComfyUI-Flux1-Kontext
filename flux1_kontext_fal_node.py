import os
import json
import time
import tempfile
import uuid
import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch

import folder_paths
from comfy.model_management import get_torch_device

# FAL.AI API INTEGRATION
try:
    import fal_client
except ImportError:
    print("Warning: fal_client not installed!")
    print("Please install dependencies with: pip install -r requirements.txt")

# Model ID for FLUX.1 Kontext on fal.ai
FLUX1_KONTEXT_MODEL_ID = "fal-ai/flux-pro/kontext"

class Flux1KontextFalNode:
    """
    ComfyUI node for image editing using fal.ai's FLUX.1 Kontext model.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_files = []
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node"""
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}), # Kontex doesn't seem to have negative prompt, but including for consistency
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}), # Kontex doesn't seem to have num_inference_steps, but including for consistency
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], {"default": "1:1"}),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "image effects/falai" # New category for fal.ai nodes

    def _pil_to_base64(self, pil_image, format="JPEG", quality=95):
        """Convert PIL image to base64 data URI"""
        buffered = BytesIO()
        pil_image.save(buffered, format=format, quality=quality)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    
    def _create_temp_image(self, image_array, format="JPEG"):
        """Create a temporary image file from a numpy array or torch tensor"""
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        
        img = Image.fromarray((image_array * 255).astype(np.uint8))
        
        if format.upper() == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        
        fd, file_path = tempfile.mkstemp(suffix=f".{format.lower()}")
        os.close(fd)
        self.temp_files.append(file_path)
        
        img.save(file_path, format=format, quality=95)
        return file_path
    
    def generate_image(self, image, prompt, api_key, negative_prompt, num_inference_steps, guidance_scale, seed, 
                       num_images, safety_tolerance, output_format, aspect_ratio):
        """Process input image and return generated images."""
        
        if not api_key:
            raise ValueError("FAL API key is required as a node input.")
        
        # Set FAL_KEY environment variable for fal_client
        os.environ["FAL_KEY"] = api_key
        
        input_img_format = "PNG" if output_format.lower() == "png" else "JPEG"
        input_image_path = self._create_temp_image(image[0], format=input_img_format)
        
        request_params = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
        }
        
        if seed != -1:
            request_params["seed"] = seed
        
        try:
            input_image_url = fal_client.upload_file(input_image_path)
            request_params["image_url"] = input_image_url
            
            print(f"Sending request to fal.ai FLUX.1 Kontext API with params: {request_params}")
            
            def on_queue_update(update):
                if hasattr(update, 'logs'):
                    for log in update.logs:
                        print(f"FLUX.1 Kontext API: {log.get('message', '')}")
            
            result = fal_client.subscribe(
                FLUX1_KONTEXT_MODEL_ID,
                arguments=request_params,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            print(f"Received response from fal.ai FLUX.1 Kontext")
            
            images = []
            
            for img_data in result.get("images", []):
                img_url = img_data.get("url")
                if not img_url:
                    continue
                
                response = requests.get(img_url)
                if response.status_code != 200:
                    print(f"Failed to download image: {response.status_code}")
                    continue
                
                img = Image.open(BytesIO(response.content))
                
                if img.mode == "RGBA" and output_format.lower() == "jpeg":
                    img = img.convert("RGB")
                elif img.mode not in ["RGB", "RGBA"]:
                    img = img.convert("RGB")
                
                filename = f"flux1_kontext_{uuid.uuid4()}.{output_format}"
                save_path = os.path.join(self.output_dir, filename)
                img.save(save_path)
                
                img_tensor = np.array(img).astype(np.float32) / 255.0
                images.append(img_tensor)
            
            if images:
                output_images = np.stack(images)
                output_images = torch.from_numpy(output_images)
                return (output_images,)
            else:
                raise ValueError("No images were generated or could be processed")
                
        except Exception as e:
            print(f"Error generating images with FLUX.1 Kontext: {str(e)}")
            raise
        finally:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"Error cleaning up temp file {temp_file}: {str(e)}")
            self.temp_files = []
    
    def __del__(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

NODE_CLASS_MAPPINGS = {
    "Flux1KontextFal": Flux1KontextFalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux1KontextFal": "FLUX.1 Kontext (fal.ai)"
} 