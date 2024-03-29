import torch
from torch.utils.data import Dataset


class CommaDataset(Dataset):
    def __init__(self, input_ids: list, input_targets: list, word_mask: list, attention_mask: list):
        self.input_ids = list(map(torch.tensor, input_ids))
        self.input_targets = list(map(torch.tensor, input_targets))
        self.word_mask = list(map(torch.tensor, word_mask))
        self.attention_mask = list(map(torch.tensor, attention_mask))

    def __getitem__(self, item):
        return {
            'feature': self.input_ids[item],
            'target': self.input_targets[item],
            'word_mask': self.word_mask[item],
            'attention_mask': self.attention_mask[item]
        }

    def __len__(self) -> int:
        return len(self.input_ids)
