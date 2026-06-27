import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.qwen_utils import MODEL_ID, freeze_model


@dataclass
class QwenNLASettings:
    model_id: str = MODEL_ID
    hidden_dim: int = 1024
    injection_token: str = "<|vision_start|>"
    injection_scale: float | None = None
    mse_scale: float | None = None
    av_prompt_template: str = (
        "Write only one short concrete English visual phrase, 2 to 6 words, "
        "with no punctuation.\n"
        "{injection_token}\n"
        "Phrase:"
    )
    ar_prompt_template: str = (
        "Reconstruct the hidden visual-temporal activation described by this "
        "short explanation.\n"
        "Explanation: {explanation}\n"
        "Activation summary:"
    )

    def resolved_injection_scale(self) -> float:
        return self.injection_scale or math.sqrt(self.hidden_dim)

    def resolved_mse_scale(self) -> float:
        return self.mse_scale or math.sqrt(self.hidden_dim)


def qwen_text_config(qwen_model):
    return getattr(qwen_model.config, "text_config", qwen_model.config)


def qwen_language_model(qwen_model):
    if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "language_model"):
        return qwen_model.model.language_model
    if hasattr(qwen_model, "language_model"):
        return qwen_model.language_model
    raise AttributeError("Could not find Qwen language_model on the loaded model")


def qwen_lm_head(qwen_model, language_model):
    get_output_embeddings = getattr(qwen_model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_embeddings = get_output_embeddings()
        if output_embeddings is not None:
            return output_embeddings
    for candidate in (qwen_model, language_model):
        if hasattr(candidate, "lm_head"):
            return candidate.lm_head
    raise AttributeError("Could not find Qwen LM head for AV SFT")


def qwen_input_embeddings(language_model):
    embed = language_model.get_input_embeddings()
    if embed is None and hasattr(language_model, "model"):
        embed = language_model.model.embed_tokens
    if embed is None:
        raise AttributeError("Could not find Qwen input embeddings")
    return embed


@torch.no_grad()
def sampled_embedding_l2(language_model, max_tokens: int = 8192) -> float:
    embed = qwen_input_embeddings(language_model)
    weight = embed.weight.detach()
    if weight.size(0) <= max_tokens:
        sample = weight
    else:
        indices = torch.linspace(
            0,
            weight.size(0) - 1,
            steps=max_tokens,
            device=weight.device,
        ).long()
        sample = weight.index_select(0, indices)
    norms = sample.float().norm(dim=-1)
    return float(norms.median().item())


def assert_qwen_nla_compatible(qwen_model, settings: QwenNLASettings) -> None:
    config = qwen_text_config(qwen_model)
    hidden_size = int(getattr(config, "hidden_size", 0))
    if hidden_size != settings.hidden_dim:
        raise ValueError(
            f"Qwen hidden size mismatch: model has {hidden_size}, "
            f"settings expect {settings.hidden_dim}"
        )
    model_type = str(getattr(config, "model_type", "")).lower()
    if "qwen" not in model_type:
        raise ValueError(
            f"Expected a Qwen text config for TM-NLA, got model_type={model_type!r}"
        )


def injection_token_id(tokenizer, settings: QwenNLASettings) -> int:
    ids = tokenizer.encode(settings.injection_token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Injection token {settings.injection_token!r} must be one token; got {ids}"
        )
    return int(ids[0])


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs eos_token or pad_token for batching")
        tokenizer.pad_token = tokenizer.eos_token


def normalize_activation(value: torch.Tensor, scale: float) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1) * scale


def direction_mse(predicted: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return F.mse_loss(
        normalize_activation(predicted, scale),
        normalize_activation(target, scale),
    )


def direction_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        F.normalize(predicted.float(), dim=-1),
        F.normalize(target.float(), dim=-1),
        dim=-1,
    )


class ActivationInjectionAdapter(nn.Module):
    """Trainable adapter from target activation space to Qwen token-embedding space."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.eye_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, activation: torch.Tensor, target_l2: float) -> torch.Tensor:
        projected = self.net(activation.float())
        return normalize_activation(projected, target_l2)


class ActivationVerbalizer(nn.Module):
    """NLA-style AV: replace one prompt token embedding with an activation."""

    def __init__(
        self,
        qwen_model,
        tokenizer,
        settings: QwenNLASettings | None = None,
        train_backbone: bool = False,
    ):
        super().__init__()
        self.settings = settings or QwenNLASettings()
        assert_qwen_nla_compatible(qwen_model, self.settings)
        self.tokenizer = tokenizer
        ensure_pad_token(self.tokenizer)
        self.injection_id = injection_token_id(tokenizer, self.settings)
        object.__setattr__(self, "qwen", qwen_model)
        self.lang = qwen_language_model(qwen_model)
        self.lm_head = qwen_lm_head(qwen_model, self.lang)
        self.default_injection_scale = sampled_embedding_l2(self.lang)
        self.adapter = ActivationInjectionAdapter(self.settings.hidden_dim)
        if not train_backbone:
            freeze_model(self.qwen)
        self._validate_prompt()

    def _prompt_text(self) -> str:
        return self.settings.av_prompt_template.format(
            injection_token=self.settings.injection_token
        )

    def _prompt_ids(self, device: torch.device) -> torch.Tensor:
        ids = self.tokenizer.encode(self._prompt_text(), add_special_tokens=True)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _validate_prompt(self) -> None:
        ids = self.tokenizer.encode(self._prompt_text(), add_special_tokens=True)
        positions = [index for index, token_id in enumerate(ids) if token_id == self.injection_id]
        if len(positions) != 1:
            raise ValueError(
                "AV prompt must contain exactly one injection token; "
                f"found {len(positions)} positions"
            )

    def _target_injection_l2(self) -> float:
        if self.settings.injection_scale is not None:
            return self.settings.resolved_injection_scale()
        return self.default_injection_scale

    def _build_training_batch(
        self,
        activations: torch.Tensor,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = activations.device
        prompt_ids = self._prompt_ids(device)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.pad_token_id
        if eos is None:
            raise ValueError("Tokenizer needs eos_token_id or pad_token_id")

        rows = []
        labels = []
        for text in texts:
            target_ids = self.tokenizer.encode(text, add_special_tokens=False)
            target_ids = target_ids + [eos]
            row = torch.cat(
                [
                    prompt_ids,
                    torch.tensor(target_ids, dtype=torch.long, device=device),
                ]
            )
            label = torch.full_like(row, -100)
            label[prompt_ids.numel() :] = torch.tensor(
                target_ids,
                dtype=torch.long,
                device=device,
            )
            rows.append(row)
            labels.append(label)

        max_len = max(row.numel() for row in rows)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        input_ids = torch.full(
            (len(rows), max_len),
            pad,
            dtype=torch.long,
            device=device,
        )
        label_ids = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)
        for index, (row, label) in enumerate(zip(rows, labels)):
            input_ids[index, : row.numel()] = row
            label_ids[index, : label.numel()] = label
            attention_mask[index, : row.numel()] = 1
        return input_ids, attention_mask, label_ids

    def _replace_injection_embedding(
        self,
        input_ids: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        embed = qwen_input_embeddings(self.lang)

        model_dtype = next(self.lang.parameters()).dtype
        inputs_embeds = embed(input_ids).to(dtype=model_dtype)
        injected = self.adapter(
            activations,
            target_l2=self._target_injection_l2(),
        ).to(device=input_ids.device, dtype=model_dtype)

        for batch_index in range(input_ids.size(0)):
            positions = (input_ids[batch_index] == self.injection_id).nonzero(as_tuple=True)[0]
            if positions.numel() != 1:
                raise ValueError("Each AV prompt row must contain exactly one injection token")
            inputs_embeds[batch_index, positions[0]] = injected[batch_index]
        return inputs_embeds

    def sft_loss(self, activations: torch.Tensor, texts: list[str]) -> torch.Tensor:
        input_ids, attention_mask, labels = self._build_training_batch(
            activations,
            texts,
        )
        inputs_embeds = self._replace_injection_embedding(input_ids, activations)
        outputs = self.lang(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        logits = self.lm_head(hidden)
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        activation: torch.Tensor,
        max_new_tokens: int = 24,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
        device = activation.device
        prompt_ids = self._prompt_ids(device).unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)
        inputs_embeds = self._replace_injection_embedding(prompt_ids, activation)
        eos = self.tokenizer.eos_token_id
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos,
            "pad_token_id": eos,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output_ids = self.qwen.generate(**gen_kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.strip()


class TextReconstructor(nn.Module):
    """NLA-style AR: text -> Qwen residual stream -> linear value head -> vector."""

    def __init__(
        self,
        qwen_model,
        tokenizer,
        settings: QwenNLASettings | None = None,
        train_backbone: bool = False,
        strip_final_norm: bool = True,
    ):
        super().__init__()
        self.settings = settings or QwenNLASettings()
        assert_qwen_nla_compatible(qwen_model, self.settings)
        self.tokenizer = tokenizer
        ensure_pad_token(self.tokenizer)
        object.__setattr__(self, "qwen", qwen_model)
        self.lang = qwen_language_model(qwen_model)
        self.value_head = nn.Linear(
            self.settings.hidden_dim,
            self.settings.hidden_dim,
            bias=False,
        )
        nn.init.eye_(self.value_head.weight)
        if strip_final_norm:
            self._strip_final_norm()
        if not train_backbone:
            freeze_model(self.qwen)

    def _strip_final_norm(self) -> None:
        inner = getattr(self.lang, "model", self.lang)
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                return

    def _prompts(self, texts: Iterable[str]) -> list[str]:
        return [
            self.settings.ar_prompt_template.format(explanation=text)
            for text in texts
        ]

    def forward_text(self, texts: list[str]) -> torch.Tensor:
        device = next(self.value_head.parameters()).device
        tokenized = self.tokenizer(
            self._prompts(texts),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        if not any(param.requires_grad for param in self.lang.parameters()):
            with torch.no_grad():
                outputs = self.lang(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            outputs = self.lang(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = outputs.last_hidden_state.float()
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        final_hidden = hidden[batch_indices, last_indices]
        return self.value_head(final_hidden.to(dtype=self.value_head.weight.dtype))

    def reconstruction_loss(
        self,
        texts: list[str],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        predicted = self.forward_text(texts)
        scale = self.settings.resolved_mse_scale()
        loss = direction_mse(predicted, targets.to(predicted.device), scale=scale)
        cosine = direction_cosine(predicted, targets.to(predicted.device)).mean()
        return loss, {
            "loss": loss.item(),
            "cosine": cosine.item(),
        }
