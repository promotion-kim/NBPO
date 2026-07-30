"""Compatibility shim for serving vLLM 0.5.1 with newer Transformers.

This is loaded only when `vllm_compat` is prepended to PYTHONPATH. It avoids
editing site-packages and restores the `LogitsWarper` symbol expected by
lm-format-enforcer in the vLLM OpenAI server import path.
"""

try:
    import transformers.generation.logits_process as _logits_process

    if not hasattr(_logits_process, "LogitsWarper") and hasattr(
        _logits_process, "LogitsProcessor"
    ):
        _logits_process.LogitsWarper = _logits_process.LogitsProcessor
except Exception:
    pass
