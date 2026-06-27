import torch
import torch.nn as nn


class TemporalContextualizer(nn.Module):
    def __init__(self, hidden_dim: int = 1024, mode: str = "temporal"):
        super().__init__()
        if mode not in {"temporal", "center_only"}:
            raise ValueError(f"Unknown temporal contextualizer mode: {mode}")
        self.hidden_dim = hidden_dim
        self.mode = mode
        input_dim = hidden_dim * 3 if mode == "temporal" else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() != 3 or context.size(1) != 3:
            raise ValueError(
                "TemporalContextualizer expects [batch, 3, hidden_dim] context"
            )
        if self.mode == "temporal":
            features = context.reshape(context.size(0), -1)
        else:
            features = context[:, 1, :]
        return self.net(features.to(dtype=self.net[0].weight.dtype))
