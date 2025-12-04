"""Utility helpers for optional bitsandbytes quantization.

Provides a small helper to construct quantization kwargs for
`from_pretrained(..., **quant_kwargs)` that gracefully handles
missing or incompatible `bitsandbytes` installs.
"""
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def get_quant_kwargs(load_in_4bit: bool = True) -> Dict:
    """Return kwargs containing a `quantization_config` when bitsandbytes is available.

    If bitsandbytes is not installed or creating the config fails, an empty dict is returned.
    """
    quant_kwargs = {}
    if not load_in_4bit:
        return quant_kwargs
    try:
        # Import lazily to avoid hard dependency at module import time
        from transformers import BitsAndBytesConfig
        import torch

        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        quant_kwargs['quantization_config'] = qcfg
        logger.info("quant_utils: BitsAndBytesConfig created — 4-bit quantization enabled.")
    except Exception as e:
        logger.warning(f"quant_utils: bitsandbytes/quantization unavailable — continuing without quantization: {e}")
    return quant_kwargs
