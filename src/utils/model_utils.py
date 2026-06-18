"""Model and tokenizer loading utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)

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
