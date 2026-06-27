import torch
import torch.nn as nn


class ProbePrefixAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 1024, prefix_len: int = 8):
        super().__init__()
        self.prefix_len = prefix_len
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prefix_len * hidden_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(dtype=self.net[0].weight.dtype)
        out = self.net(h)
        batch = h.size(0)
        return out.view(batch, self.prefix_len, -1)


class ResidualProbePrefixAdapter(nn.Module):
    """Residual prefix adapter used by the released temporal probe."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        prefix_len: int = 8,
        residual_layers: int = 2,
        zero_residual: bool = True,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_dim = hidden_dim
        self.residual_layers_count = residual_layers
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)
        self.residual_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(residual_layers)
        )
        self.output_layer = nn.Linear(hidden_dim, prefix_len * hidden_dim)
        self.activation = nn.GELU()
        self._reset_parameters(zero_residual=zero_residual)

    def _reset_parameters(self, zero_residual: bool = True):
        nn.init.kaiming_normal_(
            self.input_layer.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(self.input_layer.bias)
        nn.init.kaiming_normal_(
            self.output_layer.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(self.output_layer.bias)
        for layer in self.residual_layers:
            if zero_residual:
                nn.init.zeros_(layer.weight)
            else:
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )
            nn.init.zeros_(layer.bias)

    def initialize_from_2layer_state_dict(self, state_dict: dict):
        """Copy compatible weights from the original prefix adapter."""
        self.input_layer.weight.data.copy_(state_dict["net.0.weight"])
        self.input_layer.bias.data.copy_(state_dict["net.0.bias"])
        self.output_layer.weight.data.copy_(state_dict["net.2.weight"])
        self.output_layer.bias.data.copy_(state_dict["net.2.bias"])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(dtype=self.input_layer.weight.dtype)
        x = self.activation(self.input_layer(h))
        for layer in self.residual_layers:
            x = x + self.activation(layer(x))
        out = self.output_layer(x)
        batch = h.size(0)
        return out.view(batch, self.prefix_len, -1)


def build_activation_verbalizer(
    architecture: str = "mlp2",
    hidden_dim: int = 1024,
    prefix_len: int = 8,
):
    if architecture in {"mlp2", "activation_verbalizer"}:
        return ProbePrefixAdapter(hidden_dim=hidden_dim, prefix_len=prefix_len)
    if architecture in {"residual4", "residual_av4"}:
        return ResidualProbePrefixAdapter(
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            residual_layers=2,
            zero_residual=True,
        )
    raise ValueError(f"Unknown AV architecture: {architecture}")
