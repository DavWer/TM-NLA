from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
import torch
from typing import Optional
import re

MODEL_ID = "Qwen/Qwen3.5-0.8B"

HIDDEN_DIM = 1024
NUM_LAYERS = 24
VISION_START_ID = 248053
VISION_END_ID = 248054
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
VOCAB_SIZE = 248320


def _device_map_for(device: str):
    if device.startswith("cuda"):
        _validate_cuda_arch_for_current_gpu()
        cuda_index = int(device.split(":", 1)[1]) if ":" in device else 0
        return {"": cuda_index}
    return device


def load_qwen_4bit(device: str = "cuda") -> tuple:
    device_map = _device_map_for(device)
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Qwen 4-bit loading requires bitsandbytes. Install it separately "
            "on a supported CUDA setup, or use --qwen_precision fp16/fp32."
        ) from exc

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    freeze_model(model)
    return model, processor


def load_qwen_fp16(device: str = "cuda") -> tuple:
    """Load the frozen Qwen model without weight quantization.

    On CUDA this uses fp16 weights, which is the intended non-quantized baseline
    for an 8GB card. On CPU it falls back to fp32 because many CPU kernels do not
    support half precision well.
    """
    device_map = _device_map_for(device)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    freeze_model(model)
    return model, processor


def load_qwen_fp32(device: str = "cuda") -> tuple:
    device_map = _device_map_for(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    freeze_model(model)
    return model, processor


def load_qwen(device: str = "cuda", precision: str = "4bit") -> tuple:
    if precision == "4bit":
        return load_qwen_4bit(device)
    if precision in {"fp16", "float16", "non_quantized"}:
        return load_qwen_fp16(device)
    if precision in {"fp32", "float32"}:
        return load_qwen_fp32(device)
    raise ValueError(f"Unknown Qwen precision: {precision}")


def cast_qwen_language_model(model, dtype: torch.dtype):
    model.model.language_model.to(dtype=dtype)
    model.eval()
    freeze_model(model)
    return model


def qwen_cache_model_id(precision: str) -> str:
    if precision == "4bit":
        return f"{MODEL_ID}:bnb4bit-nf4-compute-fp16"
    if precision in {"fp16", "float16", "non_quantized"}:
        return f"{MODEL_ID}:fp16-non-quantized"
    if precision in {"fp32", "float32"}:
        return f"{MODEL_ID}:fp32-non-quantized"
    raise ValueError(f"Unknown Qwen precision: {precision}")


def _validate_cuda_arch_for_current_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    capability = torch.cuda.get_device_capability(0)
    arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list()
    if arch not in arch_list:
        raise RuntimeError(
            "The installed PyTorch CUDA wheel does not support this GPU. "
            f"GPU compute capability is {arch}, but torch was built for {arch_list}. "
            "Rebuild the venv with a PyTorch wheel that includes this architecture."
        )


def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def resolve_special_token_id(tokenizer, token: str, fallback_id: int) -> int:
    if tokenizer is None:
        return fallback_id
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        return fallback_id
    return int(token_id)


def get_vision_token_ids(tokenizer=None) -> tuple[int, int]:
    return (
        resolve_special_token_id(tokenizer, VISION_START_TOKEN, VISION_START_ID),
        resolve_special_token_id(tokenizer, VISION_END_TOKEN, VISION_END_ID),
    )


def get_visual_token_span(
    input_ids: torch.Tensor,
    vision_start_id: int = VISION_START_ID,
    vision_end_id: int = VISION_END_ID,
) -> tuple[int, int] | None:
    """Return the image-token span inside Qwen chat input IDs."""
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    vs_mask = ids == vision_start_id
    ve_mask = ids == vision_end_id
    if not vs_mask.any() or not ve_mask.any():
        return None
    start = vs_mask.nonzero(as_tuple=True)[0][0].item()
    end_candidates = ve_mask.nonzero(as_tuple=True)[0]
    end_candidates = end_candidates[end_candidates > start]
    if end_candidates.numel() == 0:
        return None
    end = end_candidates[0].item()
    return start + 1, end


def get_visual_token_spans(
    input_ids: torch.Tensor,
    vision_start_id: int = VISION_START_ID,
    vision_end_id: int = VISION_END_ID,
) -> list[tuple[int, int] | None]:
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    return [
        get_visual_token_span(row, vision_start_id, vision_end_id)
        for row in ids
    ]


def debug_visual_span(input_ids: torch.Tensor, tokenizer) -> dict:
    vision_start_id, vision_end_id = get_vision_token_ids(tokenizer)
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    seq_len = input_ids.size(-1)
    vs_mask = ids == vision_start_id
    ve_mask = ids == vision_end_id
    info = {
        "seq_len": seq_len,
        "input_ids": ids.detach().cpu().tolist(),
        "vision_start_id": vision_start_id,
        "vision_end_id": vision_end_id,
        "vision_start_token": tokenizer.decode([vision_start_id]) if tokenizer is not None else VISION_START_TOKEN,
        "vision_end_token": tokenizer.decode([vision_end_id]) if tokenizer is not None else VISION_END_TOKEN,
        "has_vision_start": vs_mask.any().item(),
        "has_vision_end": ve_mask.any().item(),
        "vision_start_count": vs_mask.sum().item(),
        "vision_end_count": ve_mask.sum().item(),
    }
    if info["has_vision_start"] and info["has_vision_end"]:
        start = vs_mask.nonzero(as_tuple=True)[0][0].item()
        end = ve_mask.nonzero(as_tuple=True)[0][0].item()
        info["vision_start_pos"] = start
        info["vision_end_pos"] = end
        info["candidate_vision_start_positions"] = vs_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        info["candidate_vision_end_positions"] = ve_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        info["num_visual_tokens"] = end - start - 1
        info["tokens_before_vision"] = start
        info["tokens_after_vision"] = seq_len - end - 1
        if tokenizer is not None:
            info["decoded_before"] = tokenizer.decode(ids[:start].squeeze())
            info["decoded_span"] = tokenizer.decode(
                ids[start : end + 1].squeeze()
            )
            info["decoded_after"] = tokenizer.decode(ids[end + 1 :].squeeze())
    return info


def prepare_generation_inputs(
    model,
    processor,
    prefix_embeds: torch.Tensor,
    anchor_text: Optional[str] = None,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.0,
):
    batch_size = prefix_embeds.size(0)
    bos_id = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
    if anchor_text is not None:
        anchor_ids = processor.tokenizer.encode(anchor_text, add_special_tokens=False)
        anchor_tensor = torch.tensor(anchor_ids, device=prefix_embeds.device).unsqueeze(0)
        anchor_embeds = model.model.language_model.embed_tokens(anchor_tensor)
        anchor_embeds = anchor_embeds.expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)
    else:
        bos_tensor = torch.tensor([[bos_id]], device=prefix_embeds.device)
        bos_embeds = model.model.language_model.embed_tokens(bos_tensor)
        bos_embeds = bos_embeds.expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prefix_embeds, bos_embeds], dim=1)

    gen_kwargs = dict(
        inputs_embeds=combined_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    return gen_kwargs


def clean_nla_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "")
    text = text.replace("assistant", "")
    text = text.replace("Latent scene state:", "")
    text = text.replace("English scene keywords:", "")
    text = re.sub(r"\bThinking(?: Process)?:.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_generation_bad_words_ids(tokenizer) -> list[list[int]]:
    blocked = [
        "<think>",
        "</think>",
        "Thinking",
        "Thinking Process",
        "Hypothesis",
        "translation",
        "user",
        "assistant",
    ]
    bad_words = []
    for text in blocked:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            bad_words.append(ids)
    return bad_words


def build_english_token_allowlist(tokenizer) -> list[int]:
    allowed = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        if not text:
            continue
        if len(text) > 24:
            continue
        if any(ord(ch) < 9 or ord(ch) > 126 for ch in text):
            continue
        if any(ch.isdigit() for ch in text):
            continue
        if not any(ch.isalpha() for ch in text):
            continue
        if not all(ch.isalpha() or ch in " -_.,;:/" for ch in text):
            continue
        lowered = text.lower()
        blocked_fragments = (
            "http",
            "www",
            "javascript",
            "function",
            "class",
            "import",
            "return",
            "user",
            "assistant",
            "think",
        )
        if any(fragment in lowered for fragment in blocked_fragments):
            continue
        allowed.append(token_id)

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        allowed.append(eos_id)
    return sorted(set(allowed))
