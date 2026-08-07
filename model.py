import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable depth, width, and activation."""

    def __init__(
        self,
        input_size=2304,
        hidden_sizes=(512,),
        num_classes=7,
        dropout_rate=0.5,
        activation="relu",
    ):
        super().__init__()

        act_layer = self._get_activation(activation)

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(act_layer())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name):
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        elif name == "sigmoid":
            return nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.network(x)
