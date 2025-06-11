from .flux1_kontext_node import FluxKontextProNode
from .flux1_kontext_fal_node import Flux1KontextFalNode

NODE_CLASS_MAPPINGS = {
    "FluxKontextPro": FluxKontextProNode,
    "Flux1KontextFal": Flux1KontextFalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextPro": "Flux Kontext Pro",
    "Flux1KontextFal": "FLUX.1 Kontext (fal.ai)"
}

# Optional: A list of node names that are `latent_image` nodes
# LATENT_IMAGE_MODES = []
