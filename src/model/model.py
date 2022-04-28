import torch
import torch.nn as nn
from transformers import AutoModel

pretrained_transformer = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')


class CommaModel(nn.Module):
    def __init__(self, num_class: int):
        super(CommaModel, self).__init__()
        bert_dim = 768
        hidden_size = bert_dim

        self.hidden_size = hidden_size
        self.pretrained_transformer = pretrained_transformer
        self.lstm = nn.LSTM(input_size=bert_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)

        self.linear = nn.Linear(in_features=hidden_size * 2,
                                out_features=num_class)

    def forward(self, x: torch.tensor, attn_masks: torch.tensor) -> torch.tensor:
        # add dummy batch for single sample
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        # (B, N, E) -> (B, N, E)
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x
