"""Model and tokenizer loading utilities."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_tokenizer(cfg):
    name = cfg.model.tokenizer_name or cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    # Most causal LMs ship without a pad token; reuse EOS for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg):
    kwargs = {"trust_remote_code": bool(cfg.model.trust_remote_code)}
    if cfg.model.torch_dtype:
        kwargs["dtype"] = _DTYPE[cfg.model.torch_dtype]
    if cfg.model.load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"

    # transformers 5.x reads ``config.pad_token_id`` during Qwen2Model init, so
    # the config needs the attribute set even when the tokenizer's pad token is
    # the same as bos/eos (the default for Qwen2.5-Instruct). Qwen2Config leaves
    # it as ``None``, which trips an AttributeError, so seed it from the tokenizer.
    config = AutoConfig.from_pretrained(
        cfg.model.name, trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if not getattr(config, "pad_token_id", None):
        tokenizer = load_tokenizer(cfg)
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, config=config, **kwargs)

    # When not using 8-bit/device_map, place the model explicitly.
    if not cfg.model.load_in_8bit:
        model.to(cfg.device)
    return model


def maybe_wrap_lora(cfg, model):
    """Attach LoRA adapters when ``cfg.lora.enable`` is set.

    Returns the (possibly wrapped) model. No-op if PEFT is unavailable or
    LoRA is disabled, so the rest of the pipeline stays decoupled from PEFT.
    """
    if not (cfg.lora and cfg.lora.enable):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("LoRA requested but `peft` is not installed.")

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def load_model_and_tokenizer(cfg):
    return load_model(cfg), load_tokenizer(cfg)


def load_vllm(cfg, model_path, adapter_dir=None):
    """Build a vLLM engine for fast batched evaluation.

    ``model_path`` is what vLLM loads: a full checkpoint dir, or the base model
    name when ``adapter_dir`` points at a LoRA adapter saved by the train stage.
    Returns ``(llm, lora_request)`` -- ``lora_request`` is ``None`` unless a LoRA
    adapter is attached, in which case pass it to ``llm.generate(...)``.
    """
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError(
            "vLLM eval backend requested but `vllm` is not installed. Install it "
            "with `pip install vllm` (CUDA GPU required), or set eval.backend=hf."
        ) from exc

    kwargs = {
        "model": model_path,
        "trust_remote_code": bool(cfg.model.trust_remote_code),
        "dtype": cfg.model.torch_dtype or "auto",
        "gpu_memory_utilization": cfg.get_path("eval.gpu_memory_utilization") or 0.5,
        # Skip CUDA-graph capture; avoids the vLLM 0.10 V1 JIT-compile path
        # (FlashInfer/Triton kernels) that needs CUDA dev headers not present
        # on many boxes. For full startup, also set VLLM_USE_V1=0.
        "enforce_eager": True,
    }
    if cfg.model.max_length:
        kwargs["max_model_len"] = cfg.model.max_length

    lora_request = None
    if adapter_dir:
        kwargs["enable_lora"] = True
        lora_request = LoRARequest("trained", 1, adapter_dir)

    return LLM(**kwargs), lora_request
