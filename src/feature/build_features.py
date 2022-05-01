import os
import pickle
import sys

import pandas as pd

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
targets = {',': 1, '.': 2}


def extract_sample(dataset_path: str = 'data/raw/lenta/lenta.csv', sample_size: int = 100_000):
    output_path = os.path.join('data/raw/lenta', 'lenta_text.csv')
    pd.read_csv(dataset_path, low_memory=False)['text'][:sample_size].to_csv(output_path)


def cut_text(max_length: int,
             sample_path: str = 'data/raw/lenta/lenta_text.csv',
             text: list = None,
             return_text: bool = False,
             is_train: bool = True,
             ):
    if text is None:
        text = pd.read_csv(sample_path)['text'].values

    def reshape_sentence(sentence_splitted: list, n: int) -> str:
        for i in range(0, len(sentence_splitted), n):
            yield ' '.join(sentence_splitted[i:i + n])

    res = []
    for sample in text:
        for sentence in reshape_sentence(sample.split(), max_length):
            res.append(sentence)

    if is_train:
        pd.DataFrame(res, columns=['text']).to_csv('data/interim/lenta_cutted.csv', index=False)

    if return_text:
        return res


def build_features(sample_path: str = 'data/interim/lenta_cutted.csv',
                   text: list = None,
                   return_features: bool = False,
                   is_train: bool = True
                   ):
    if text is None:
        text = pd.read_csv(sample_path)['text'].values
    # text = ['казнить, нельзя помиловать#.', 'привет со дна #38.', 'что-то пошло не так (.']
    # text = text[0:1]
    tokenized_text = [tokenizer.tokenize(sent) for sent in text]
    tokenized_text = [['[SOS]'] + sentence + ['[EOS]'] for sentence in tokenized_text]
    # print(tokenized_text)

    input_tokens = list(
        map(lambda sentence: list(
            filter(
                lambda x: x not in targets.keys(),
                sentence)
        ),
            tokenized_text)
    )
    # print(input_tokens)

    input_ids = list(map(tokenizer.convert_tokens_to_ids, input_tokens))

    # print(input_ids)

    def shift_target(arr: list) -> list:
        res = []
        for i in arr:
            if i != 0:
                res.pop()
            res.append(i)
        return res

    input_targets = list(map(lambda sentence: shift_target([targets.get(x, 0) for x in sentence]), tokenized_text))

    # print(input_targets)

    def mask_tokens(tokens: list) -> list:
        res = []
        for i in range(len(tokens) - 1):
            if tokens[i + 1][0] != '#':
                res.append(1)
            elif tokens[i + 1] == '#':
                res.append(1)
            elif tokens[i + 1][0] == '#':
                res.append(0)
            else:
                raise NotImplementedError
        res.append(1)
        assert len(res) == len(tokens)
        return res

    target_mask = list(map(lambda x: mask_tokens(x), input_tokens))
    # print(target_mask)

    attention_mask = list(map(lambda x: [1 for _ in range(len(x))], input_ids))
    # print(attention_mask)

    if is_train:
        with open('data/processed/input_ids.pkl', 'wb') as f:
            pickle.dump(input_ids, f)
        with open('data/processed/input_targets.pkl', 'wb') as f:
            pickle.dump(input_targets, f)
        with open('data/processed/target_mask.pkl', 'wb') as f:
            pickle.dump(target_mask, f)
        with open('data/processed/attention_mask.pkl', 'wb') as f:
            pickle.dump(attention_mask, f)

    if return_features:
        return input_ids, input_targets, target_mask, attention_mask
