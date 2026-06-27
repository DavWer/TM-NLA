import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activation_verbalizer import ProbePrefixAdapter
from utils.qwen_utils import (
    build_english_token_allowlist,
    build_generation_bad_words_ids,
)


class TMNLA(nn.Module):
    def __init__(
        self,
        av: ProbePrefixAdapter,
        qwen_model,
        processor=None,
        nla_len: int = 8,
        placeholder_text: str = "English scene keywords:",
        normalize_prefix: bool = True,
        prefix_scale: float = 1.0,
    ):
        super().__init__()
        self.av = av
        object.__setattr__(self, "qwen", qwen_model)
        self.processor = processor
        self.nla_len = nla_len
        self.placeholder_text = placeholder_text
        self.normalize_prefix = normalize_prefix
        self.prefix_scale = prefix_scale
        self._freeze_qwen()

        embed = self.qwen.model.language_model.embed_tokens
        embed_weight = embed.weight.detach().float()
        embed_rms = embed_weight.pow(2).mean(dim=-1).sqrt().mean()
        self.register_buffer("_embed_mean", embed_weight.mean(dim=0))
        self.register_buffer("_embed_rms", embed_rms)

        placeholder_ids = self._build_placeholder_ids()
        self.register_buffer(
            "_placeholder_ids",
            placeholder_ids,
        )

    def _freeze_qwen(self) -> None:
        self.qwen.eval()
        for param in self.qwen.parameters():
            param.requires_grad = False

    def _build_placeholder_ids(self) -> torch.Tensor:
        cfg = self.qwen.config.text_config
        fallback_id = getattr(cfg, "eos_token_id", None)
        if fallback_id is None:
            fallback_id = getattr(cfg, "pad_token_id", None)
        if fallback_id is None:
            raise ValueError("Qwen text config must expose eos_token_id or pad_token_id")

        if self.processor is not None:
            ids = self.processor.tokenizer.encode(
                self.placeholder_text,
                add_special_tokens=False,
            )
        else:
            ids = []

        if not ids:
            ids = [fallback_id]
        ids = ids[: self.nla_len]
        ids.extend([fallback_id] * (self.nla_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def _build_inputs_embeds(self, prefix: torch.Tensor):
        batch = prefix.size(0)
        lang = self.qwen.model.language_model
        embed = lang.embed_tokens
        prefix = self._normalize_prefix(prefix)
        placeholder_ids = self._placeholder_ids.to(device=prefix.device)
        placeholder_ids = placeholder_ids.view(1, -1).expand(batch, -1)
        with torch.no_grad():
            placeholder_embeds = embed(placeholder_ids).detach()
        placeholder_embeds = placeholder_embeds.to(dtype=prefix.dtype)
        inputs_embeds = torch.cat([prefix, placeholder_embeds], dim=1)
        return inputs_embeds

    def _normalize_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        if not self.normalize_prefix:
            return prefix
        dtype = prefix.dtype
        normalized = F.layer_norm(prefix.float(), (prefix.size(-1),))
        embed_mean = self._embed_mean.to(device=prefix.device)
        embed_rms = self._embed_rms.to(device=prefix.device)
        normalized = normalized * embed_rms * self.prefix_scale + embed_mean
        return normalized.to(dtype=dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self._freeze_qwen()
        prefix = self.av(h)
        inputs_embeds = self._build_inputs_embeds(prefix)
        lang = self.qwen.model.language_model
        model_dtype = next(lang.parameters()).dtype
        attn_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=prefix.device
        )
        outputs = lang(
            inputs_embeds=inputs_embeds.to(dtype=model_dtype),
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        hidden = outputs.last_hidden_state
        nla_emb = hidden[:, self.av.prefix_len :, :].mean(dim=1)
        return nla_emb

    def generate_nla(
        self,
        h: torch.Tensor,
        mode: str = "lean",
        max_new_tokens: int = 16,
        do_sample: bool = False,
        eos_token_id: int = None,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        english_only: bool = False,
    ) -> torch.Tensor:
        self._freeze_qwen()
        prefix = self._normalize_prefix(self.av(h))
        batch = prefix.size(0)
        lang = self.qwen.model.language_model
        embed = lang.embed_tokens
        model_dtype = next(lang.parameters()).dtype

        if mode == "lean":
            cfg = self.qwen.config.text_config
            bos_id = getattr(cfg, 'bos_token_id', None) or cfg.eos_token_id
            bos = torch.tensor([[bos_id]], device=prefix.device)
            bos_embeds = embed(bos).expand(batch, -1, -1)
            inputs_embeds = torch.cat([prefix, bos_embeds], dim=1)
        elif mode == "anchored":
            if self.processor is None:
                raise ValueError("processor required for anchored mode")
            anchor_tokens = self.processor.tokenizer.encode(
                self.placeholder_text, add_special_tokens=False
            )
            anchor = torch.tensor([anchor_tokens], device=prefix.device)
            anchor_embeds = embed(anchor).expand(batch, -1, -1)
            inputs_embeds = torch.cat([prefix, anchor_embeds], dim=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        eos = eos_token_id if eos_token_id is not None else self.qwen.config.text_config.eos_token_id
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds.to(dtype=model_dtype),
            attention_mask=torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=prefix.device
            ),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos,
            pad_token_id=eos,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if self.processor is not None:
            gen_kwargs["bad_words_ids"] = build_generation_bad_words_ids(
                self.processor.tokenizer
            )
            if english_only:
                allowed_ids = build_english_token_allowlist(self.processor.tokenizer)
                gen_kwargs["prefix_allowed_tokens_fn"] = (
                    lambda batch_id, input_ids: allowed_ids
                )
        if do_sample:
            gen_kwargs["temperature"] = 0.7

        with torch.no_grad():
            output_ids = self.qwen.generate(**gen_kwargs)

        return output_ids
