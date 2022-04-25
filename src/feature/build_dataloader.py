import torch
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch):
    max_len = max(len(row["feature"]) for row in batch)

    input_ids = torch.empty((len(batch), max_len), dtype=torch.long)
    input_target = torch.empty((len(batch), max_len), dtype=torch.long)
    target_mask = torch.empty((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.empty((len(batch), max_len), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row["feature"])
        input_ids[idx] = torch.cat((row["feature"], torch.zeros(to_pad)))
        input_target[idx] = torch.cat((row["target"], torch.zeros(to_pad)))
        target_mask[idx] = torch.cat((row["target_mask"], torch.zeros(to_pad)))
        attention_mask[idx] = torch.cat((row["attention_mask"], torch.zeros(to_pad)))

    return {
        'feature': input_ids,
        'target': input_target,
        'target_mask': target_mask,
        'attention_mask': attention_mask
    }


def build_dataloader(dataset: Dataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader
