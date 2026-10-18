import random

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Baseline(nn.Module):

    def __init__(
        self,
        num_char: int,
        hidden_dimension: int = 768,
        char_dimension: int = 768,
        text_max_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):

        super(Baseline, self).__init__()

        self.text_max_len = text_max_len
        self.hidden_dimension = hidden_dimension
        self.char_dimension = char_dimension
        self.num_char = num_char
        self.num_layers = num_layers
        self.dropout_p = dropout

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        encoder_feat_dim = resnet.fc.in_features

        modules = list(resnet.children())[:-1]

        self.encoder = nn.Sequential(*modules)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.feature_proj = nn.Linear(encoder_feat_dim, hidden_dimension)

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            char_dimension,
            hidden_dimension,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.embed_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dimension, num_char)

        self.lup = nn.Embedding(num_char, char_dimension)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, token_labels: torch.Tensor, teacher_forcing: float = 0.5):
        outputs = []
        features = self.encoder(x).flatten(1, -1)
        features = self.feature_proj(features)
        hidden_state = features.repeat(self.num_layers, 1, 1)

        max_iterations = min(self.text_max_len, token_labels.shape[1])

        for t in range(max_iterations):
            if (random.random() > teacher_forcing) and (t != 0):
                tokens = torch.argmax(char_logits, dim=1)
            else:
                tokens = token_labels[:, t]

            decoder_input = self.embed_dropout(self.lup(tokens)).unsqueeze(1)

            rnn_output, hidden_state = self.rnn(decoder_input, hidden_state)

            char_logits = self.proj(self.out_dropout(rnn_output[:, -1, :].relu()))

            outputs.append(char_logits)

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    model = Baseline(num_char=10)
    print(model)

    input_tensor = torch.randn(32, 3, 224, 224)
    tokens = torch.randint(size=(32, 5), high=10)
    output = model(input_tensor, tokens)
    print(output.shape)
