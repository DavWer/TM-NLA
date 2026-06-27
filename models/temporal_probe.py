import torch
import torch.nn as nn

from models.temporal_contextualizer import TemporalContextualizer
from models.tm_nla import TMNLA


class TemporalProbe(nn.Module):
    """Applies temporal context before the TM-NLA projection."""

    def __init__(
        self,
        contextualizer: TemporalContextualizer,
        tm_nla: TMNLA,
    ):
        super().__init__()
        self.contextualizer = contextualizer
        self.tm_nla = tm_nla

    def contextualize(self, context: torch.Tensor) -> torch.Tensor:
        return self.contextualizer(context)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        h_temporal = self.contextualize(context)
        return self.tm_nla(h_temporal)

    @property
    def av(self):
        return self.tm_nla.av
