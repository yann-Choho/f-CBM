"""
CLIP/BLIP Model Registry
=========================

Centralized mapping of model names to HuggingFace checkpoints and embedding dimensions.
ALL files in the project should import from here instead of hardcoding checkpoint strings.

Usage:
    from clip_config import get_clip_checkpoint, get_clip_dim, is_clip_model

    checkpoint = get_clip_checkpoint(config.model_name)  # "openai/clip-vit-large-patch14"
    dim = get_clip_dim(config.model_name)                # 768
    is_clip = is_clip_model(config.model_name)           # True
"""

# ============================================================================
# MODEL REGISTRY
# ============================================================================

CLIP_REGISTRY = {
    # CLIP variants
    'clip': {
        'checkpoint': 'openai/clip-vit-base-patch32',
        'text_dim': 512,
        'image_dim': 512,
        'multimodal_dim': 1024,   # text_dim + image_dim
        'max_tokens': 77,
        'family': 'clip',
    },
    'clip-large': {
        'checkpoint': 'openai/clip-vit-large-patch14',
        'text_dim': 768,
        'image_dim': 768,
        'multimodal_dim': 1536,   # text_dim + image_dim
        'max_tokens': 77,
        'family': 'clip',
    },
    # BLIP variants
    'blip': {
        'checkpoint': 'Salesforce/blip-image-captioning-base',
        'text_dim': 768,
        'image_dim': 768,
        'multimodal_dim': 1536,
        'max_tokens': 512,
        'family': 'blip',
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_clip_checkpoint(model_name: str) -> str:
    """Get HuggingFace checkpoint string for a CLIP/BLIP model."""
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        raise ValueError(
            f"Unknown CLIP/BLIP model: '{model_name}'. "
            f"Available: {list(CLIP_REGISTRY.keys())}"
        )
    return CLIP_REGISTRY[model_name]['checkpoint']


def get_clip_dim(model_name: str, mode: str = 'text') -> int:
    """
    Get embedding dimension for a CLIP/BLIP model.
    
    Args:
        model_name: 'clip', 'clip-large', 'blip'
        mode: 'text', 'image', or 'multimodal'
    
    Returns:
        int: Embedding dimension
    """
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    
    reg = CLIP_REGISTRY[model_name]
    if mode == 'text':
        return reg['text_dim']
    elif mode == 'image':
        return reg['image_dim']
    elif mode == 'multimodal':
        return reg['multimodal_dim']
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'text', 'image', or 'multimodal'")


def get_clip_max_tokens(model_name: str) -> int:
    """Get max token limit for a CLIP/BLIP model."""
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        return 512  # default fallback
    return CLIP_REGISTRY[model_name]['max_tokens']


def is_clip_model(model_name: str) -> bool:
    """Check if model is a CLIP variant (not BLIP)."""
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        return False
    return CLIP_REGISTRY[model_name]['family'] == 'clip'


def is_blip_model(model_name: str) -> bool:
    """Check if model is a BLIP variant."""
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        return False
    return CLIP_REGISTRY[model_name]['family'] == 'blip'


def is_clip_family(model_name: str) -> bool:
    """Check if model is any CLIP or BLIP variant."""
    return model_name.lower() in CLIP_REGISTRY


def get_clip_family(model_name: str) -> str:
    """Get model family ('clip' or 'blip')."""
    model_name = model_name.lower()
    if model_name not in CLIP_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    return CLIP_REGISTRY[model_name]['family']
