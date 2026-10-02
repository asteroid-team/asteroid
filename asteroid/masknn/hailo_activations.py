from torch import nn


def get_hailo_activation(name: str) -> nn.Module:
    key = name.lower()
    if key in {"relu", "r"}:
        return nn.ReLU(inplace=False)
    if key in {"sigmoid", "s"}:
        return nn.Sigmoid()
    if key in {"tanh", "t"}:
        return nn.Tanh()
    if key in {"linear", "identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported Hailo activation: {name}")
