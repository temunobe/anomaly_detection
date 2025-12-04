# Small verification script: checks HF token in env and attempts to load the Mistral tokenizer
import os
import traceback
from transformers import AutoTokenizer

MODEL = 'mistralai/Mistral-Large-3-675B-Instruct-2512'

print('HF_TOKEN present in env:', bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')))
print('Transformers version:', end=' ')
try:
    import transformers
    print(transformers.__version__)
except Exception:
    print('unknown')

try:
    print(f"Attempting to load tokenizer for {MODEL} with use_fast=False, fix_mistral_regex=True")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False, fix_mistral_regex=True)
    print('Loaded tokenizer ok')
    print('Tokenizer class:', type(tok))
    print('Vocab size (if available):', getattr(tok, 'vocab_size', None))
except Exception as e:
    print('Failed to load tokenizer:')
    traceback.print_exc()
