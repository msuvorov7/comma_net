import re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


TARGETS = (',', '.', '-', '_')
IND_TO_TARGET_TOKEN = {1: 20, 2: 24, 3: 45, 4: 107}
TARGET_TOKEN_TO_IDS = {20: 1, 24: 2, 45: 3, 107: 4}
TOKENIZER = DistilBertTokenizerFast.from_pretrained('DeepPavlov/distilrubert-tiny-cased-conversational-v1')


class Augmentation:
    def __init__(
            self,
            max_words: int,
    ):
        self.max_words = max_words
        self.letters = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

    def sample_string(self, text: str) -> str:
        words = text.split()
        if len(words) > self.max_words:
            sample_words = np.random.randint(self.max_words // 2, self.max_words)
            start_pos = np.random.randint(0, len(words) - sample_words)
            return ' '.join(words[start_pos:start_pos + sample_words])
        return text
    
    def make_error(self, text: str, error_prob: float = 0.2) -> str:
        if (not text) or (len(text) < 10) or (np.random.random() > error_prob):
            return text

        error_type = np.random.choice(['swap', 'insert', 'delete', 'replace'])
        error_idx = np.random.randint(0, len(text) - 1)

        if error_type == 'swap':
            text = text[:error_idx] + text[error_idx + 1] + text[error_idx] + text[error_idx + 2:]
        elif error_type == 'insert':
            char_to_insert = np.random.choice(self.letters)
            text = text[:error_idx] + char_to_insert + text[error_idx:]
        elif error_type == 'delete':
            text = text[:error_idx] + text[error_idx+1:]
        elif error_type == 'replace':
            char_to_replace = np.random.choice(self.letters)
            text = text[:error_idx] + char_to_replace + text[error_idx+1:]

        return text


class CommaV2(Dataset):
    def __init__(
            self,
            data: list,
            tokenizer,
            max_length: int,
            mode: str,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        self.augmentation = Augmentation(max_words=int(max_length * 0.9))

        self.ind_to_target_token = dict(
            enumerate(
                self.tokenizer.convert_tokens_to_ids(TARGETS),
                start=1,
            )
        )
        self.target_token_to_ids = {v: k for k, v in self.ind_to_target_token.items()}
        # print(self.ind_to_target_token)
        # print(self.target_token_to_ids)

    def __getitem__(self, index) -> dict:
        text = self.data[index]
        if self.mode == 'train':
            text = self.augmentation.sample_string(text)
            text = self.augmentation.make_error(text)

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
        )

        x, y, attn_mask = [], [], []
        for i in range(len(inputs['input_ids'])):
            token = inputs['input_ids'][i]
            if token in self.target_token_to_ids:
                if token == 45:
                    # отличаем дефис от тире
                    if (inputs['offset_mapping'][i][0] == inputs['offset_mapping'][i - 1][1]):
                        token = 107
                y.pop()
                y.append(self.target_token_to_ids[token])
            else:
                x.append(token)
                y.append(0)
                attn_mask.append(1)

        return {
            'x': torch.as_tensor(x),
            'y': torch.as_tensor(y),
            'attn_mask': torch.as_tensor(attn_mask),
        }

    def __len__(self) -> int:
        return len(self.data)
    

def collate_fn(batch) -> dict:
    """
    Обработчик батча перед входом в модель.
    Забивает предложения pad-токенами до длинны самого длинного
    предложения в батче
    :param batch: батч данных
    :return:
    """
    max_len = max(len(row["x"]) for row in batch)

    x = torch.empty((len(batch), max_len), dtype=torch.long)
    y = torch.empty((len(batch), max_len), dtype=torch.long)
    attn_mask = torch.empty((len(batch), max_len), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row["x"])
        x[idx] = torch.cat((row["x"], torch.zeros(to_pad)))
        y[idx] = torch.cat((row["y"], torch.zeros(to_pad)))
        attn_mask[idx] = torch.cat((row["attn_mask"], torch.zeros(to_pad)))

    return {
        'x': x,
        'y': y,
        'attn_mask': attn_mask
    }


if __name__ == '__main__':
    # only for debugging

    df = pd.read_parquet('data/raw/test.parquet', columns=['text'])
    normalize_pattern = re.compile(r'[^а-яa-z0-9\s\,\.\-]')
    text = (
        df['text']
        .dropna()
        .apply(lambda item: item.lower())
        .apply(lambda item: re.sub(normalize_pattern, '', item))
    ).values.tolist()
    
    dataset = CommaV2(
        data=text,
        tokenizer=TOKENIZER,
        max_length=128,
        mode='train',
    )

    stat = list(map(lambda item: len(item.split()), text))
    print(max(stat), min(stat), np.mean(stat))

    print(text[:3])
    
    print(dataset[0])

    print(collate_fn((dataset[100], dataset[2])))