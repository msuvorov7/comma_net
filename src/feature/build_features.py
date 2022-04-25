import os
import pickle
import sys

import pandas as pd
import re

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
targets = {',': 1, '.': 2}


def extract_sample(dataset_path: str = 'data/raw/lenta/lenta.csv', sample_size: int = 100_000):
    output_path = os.path.join('data/raw/lenta', 'lenta_text.csv')
    pd.read_csv(dataset_path, low_memory=False)['text'][:sample_size].to_csv(output_path)


def punctuation():
    pass


def foo(sample_path: str = 'data/raw/lenta/lenta_text.csv'):
    text = ['казнить, нельзя помиловать.', 'привет со дна.']
    tokenized_texts = [tokenizer.tokenize(sent) for sent in text]
    tokenized_texts = [['[SOS]'] + sentense + ['[EOS]'] for sentense in tokenized_texts]

    TOKENS = []
    Y = []
    for i in tokenized_texts:
        token = []
        y = []
        y_mask = []
        # для простоты оставили только два класса
        for word in i:
            if word == ',':
                y = y[:-1]
                y.append(1)
            elif word == '.':
                y = y[:-1]
                y.append(2)
            else:
                token.append(word)
                y.append(0)
        TOKENS.append(token)
        Y.append(y)
    print(Y[0])
    print(TOKENS[0])

    Y_MASK = []
    for i in text:
        y_mask = [1]
        for word in i.replace('—', '').replace(',', '').replace('.', '').split():
            # print(tokenizer.tokenize(word))
            word_pieces = tokenizer.tokenize(word)
            if len(word_pieces) == 1:
                y_mask.append(1)
            else:
                y_mask += [0 for _ in range(len(word_pieces) - 1)]
                y_mask.append(1)
        y_mask.append(1)
        Y_MASK.append(y_mask)
    print(Y_MASK[0])

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in TOKENS]
    return text


def build_features(sample_path: str = 'data/raw/lenta/lenta_text.csv'):
    text = pd.read_csv(sample_path)['text'].values
    # text = ['казнить, нельзя помиловать.', 'привет со дна.', 'что-то пошло не так (.']
    # text = text[1:2]
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

    def mask_tokens(arr: list) -> list:
        res = [1]  # for [SOS]
        lens = list(
            map(
                lambda x: len(  # длина массива токенов слова, чтобы найти последний токен (если в слове > 1 токена)
                    list(
                        filter(  # выбор токенов не из таргета
                            lambda token: token not in targets.keys(),
                            tokenizer.tokenize(x))
                    )
                ),
                arr)
        )
        for i in lens:
            if i > 1:
                res += [0 for _ in range(i - 1)]  # все промежуточные токены заполнены нулями
            res.append(1)
        res.append(1)  # for [EOS]
        return res

    target_mask = list(
        map(lambda sentence: mask_tokens(sentence.split()), text))
    # print(target_mask)

    attention_mask = list(map(lambda x: [1 for _ in range(len(x))], input_ids))
    # print(attention_mask)

    # return input_ids, input_targets, target_mask, attention_mask

    with open('data/interim/input_ids.pkl', 'wb') as f:
        pickle.dump(input_ids, f)
    with open('data/interim/input_targets.pkl', 'wb') as f:
        pickle.dump(input_targets, f)
    with open('data/interim/target_mask.pkl', 'wb') as f:
        pickle.dump(target_mask, f)
    with open('data/interim/attention_mask.pkl', 'wb') as f:
        pickle.dump(attention_mask, f)

    with open('data/interim/input_ids.pkl', 'rb') as f:
        input_ids = pickle.load(f)
    # print(input_ids)


if __name__ == '__main__':
    extract_sample()
    foo()
