import torch
from typing import Optional
from utils.qwen_utils import (
    VISION_END_ID,
    VISION_START_ID,
    get_visual_token_spans,
    get_vision_token_ids,
)


class HiddenStateExtractor:
    def __init__(
        self,
        model,
        target_layer: int = 20,
        tokenizer=None,
        strict_visual_span: bool = True,
    ):
        self.hidden = None
        self.target_layer = target_layer
        self.model = model
        self.strict_visual_span = strict_visual_span
        self.vision_start_id, self.vision_end_id = (
            get_vision_token_ids(tokenizer)
            if tokenizer is not None
            else (VISION_START_ID, VISION_END_ID)
        )
        lang_model = model.model.language_model
        self.handle = lang_model.layers[target_layer].register_forward_hook(
            self._capture
        )

    def _capture(self, module, input, output):
        self.hidden = output[0] if isinstance(output, (tuple, list)) else output

    def extract(
        self,
        inputs: dict,
        pool_visual: bool = True,
    ) -> torch.Tensor:
        self.hidden = None
        with torch.no_grad():
            self.model(**inputs, output_hidden_states=True)
        h = self.hidden
        if h is None:
            raise RuntimeError(
                f"Forward hook on layer {self.target_layer} did not fire"
            )
        if pool_visual:
            return self._pool_visual_tokens(h, inputs["input_ids"])
        return h.mean(dim=1)

    def extract_at_positions(
        self, inputs: dict, positions: slice = None
    ) -> torch.Tensor:
        self.hidden = None
        with torch.no_grad():
            self.model(**inputs, output_hidden_states=True)
        h = self.hidden
        if h is None:
            raise RuntimeError(f"Forward hook on layer {self.target_layer} did not fire")
        if positions is not None:
            h = h[:, positions, :]
        return h

    def _pool_visual_tokens(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        spans = get_visual_token_spans(
            input_ids,
            vision_start_id=self.vision_start_id,
            vision_end_id=self.vision_end_id,
        )
        pooled = []
        for batch_idx, span in enumerate(spans):
            if span is None:
                if self.strict_visual_span:
                    raise RuntimeError(
                        "Could not find vision_start/vision_end token span for "
                        f"batch item {batch_idx}. Run debug_visual_span before training."
                    )
                pooled.append(h[batch_idx].mean(dim=0))
                continue
            start, end = span
            if end <= start:
                raise RuntimeError(
                    f"Empty visual token span for batch item {batch_idx}: {span}"
                )
            pooled.append(h[batch_idx, start:end, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _get_visual_span(self, input_ids: torch.Tensor) -> Optional[tuple]:
        return get_visual_token_spans(
            input_ids,
            vision_start_id=self.vision_start_id,
            vision_end_id=self.vision_end_id,
        )[0]

    def close(self):
        self.handle.remove()
