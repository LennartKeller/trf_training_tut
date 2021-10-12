from logging import lastResort
import torch
from torch import nn


class FeedForwardClassifier(nn.Module):
    def __init__(self, input_features, n_classes, hidden_layer_sizes=None):
        super().__init__()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [256, 128, 64, 32]

        input_layer = nn.Linear(in_features=input_features, out_features=256)
        output_layer = nn.Linear(
            in_features=hidden_layer_sizes[-1], out_features=n_classes
        )
        layers = [input_layer]
        for idx, hidden_size in enumerate(hidden_layer_sizes):
            in_features = (
                hidden_layer_sizes[idx - 1] if idx > 0 else input_layer.out_features
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
        layers.extend([nn.GELU(), output_layer])

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        logits = self.model(inputs)
        return logits


if __name__ == "__main__":
    batch = torch.randn(16, 512)
    labels = torch.randint(0, 10, (16,))

    model = FeedForwardClassifier(512, 10)
    print(model)
    logits = model(batch)
    print(torch.softmax(logits, dim=1))
