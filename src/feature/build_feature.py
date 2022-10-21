import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.dataset import CommaDataset

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
targets = {',': 1, '.': 2}


class ContentWrapper:
    def __init__(self, max_size: int):
        self.batches = []
        self.max_size = max_size

    def fit(self, text):
        sentence_splitted = text.split()
        for i in tqdm(range(0, len(sentence_splitted), self.max_size)):
            self.batches.append(' '.join(sentence_splitted[i:i + self.max_size]))
        return self

    def get_split(self):
        return self.batches


def extract_sample(directiry_path: str, mode: str, sample_size: int):
    dataset_path = directiry_path + f'{mode}.csv'
    pd.read_csv(dataset_path, low_memory=False)['text'][:sample_size].to_csv(directiry_path + f'{mode}_sample.csv')


def cut_text(max_length: int,
             sample_path: str = 'data_load/raw/lenta/lenta_text.csv',
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
        pd.DataFrame(res, columns=['text']).to_csv('data_load/interim/lenta_cutted.csv', index=False)

    if return_text:
        return res


def build_features(text: str):
    content = ContentWrapper(max_size=150).fit(text)
    reshaped_text = content.get_split()
    # text = ['казнить, нельзя помиловать#.', 'привет со дна #38.', 'что-то пошло не так (.']
    # text = text[0:1]
    tokenized_text = [tokenizer.tokenize(sent) for sent in reshaped_text]
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
    logging.info('input_ids created')
    # print(input_ids)

    def shift_target(arr: list) -> list:
        res = []
        for i in arr:
            if i != 0:
                res.pop()
            res.append(i)
        return res

    input_targets = list(map(lambda sentence: shift_target([targets.get(x, 0) for x in sentence]), tokenized_text))
    logging.info('input_targets created')
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
    logging.info('target_mask created')
    # print(target_mask)

    attention_mask = list(map(lambda x: [1 for _ in range(len(x))], input_ids))
    logging.info('attention_mask created')
    # print(attention_mask)

    return input_ids, input_targets, target_mask, attention_mask


def download_dataframe(filename: str) -> str:
    text = pd.read_csv(filename)['text'].fillna(' ').values
    text = ' '.join(text)
    return text


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--mode', default='train', dest='mode')
    args_parser.add_argument('--sample_size', default=-1, dest='sample_size', type=int)
    args = args_parser.parse_args()

    assert args.mode in ('train', 'test')
    assert args.sample_size != 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    data_processed_dir = fileDir + config['data']['processed']

    if args.sample_size > 0:
        extract_sample(data_raw_dir, args.mode, args.sample_size)
        dataset = download_dataframe(f'{data_raw_dir}{args.mode}_sample.csv')
    else:
        dataset = download_dataframe(f'{data_raw_dir}{args.mode}.csv')

    input_ids, input_targets, target_mask, attention_mask = build_features(dataset)

    comma_dataset = CommaDataset(input_ids, input_targets, target_mask, attention_mask)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as f:
        pickle.dump(comma_dataset, f)

    logging.info(f'artefacts saved in {data_processed_dir}')
