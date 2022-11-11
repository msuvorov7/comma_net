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
    """
    Класс для обработки сырого текста
    """
    def __init__(self, max_size: int):
        """
         Конструктор для класса. Создает пустой массив подстрок
        :param max_size: максимальный размер токенов подстроки (.split())
        """
        self.batches = []
        self.max_size = max_size

    def fit(self, text):
        """
        Разделить длинную строку на подстроки каждая их которых
        не более чем из max_size токенов. Потребность в методе связана
        с тем, чтобы уметь обрабатывать очень длинные последовательности.
        Для bert модели установлен лимит на длину в 500 токенов.
        Дополнительно приводит текст к нижнему регистру.
        :param text: Строка из текста, которую нужно разделить на подстроки
        :return:
        """
        sentence_splitted = text.lower().split()
        for i in tqdm(range(0, len(sentence_splitted), self.max_size)):
            self.batches.append(' '.join(sentence_splitted[i:i + self.max_size]))
        return self

    def get_split(self) -> list:
        """
        Вернуть массив подстрок из исходной строки
        :return:
        """
        return self.batches


def extract_sample(directiry_path: str, mode: str, sample_size: int):
    """
    Вспомогательная функция для взятия подвыборки из датасета
    :param directiry_path: путь до raw директории
    :param mode: train или test файл бьем
    :param sample_size: количество строк в выборке
    :return: сохраняет файл в суффиксом _sample.csv
    """
    dataset_path = directiry_path + f'{mode}.csv'
    pd.read_csv(dataset_path, low_memory=False)['text'][:sample_size].to_csv(directiry_path + f'{mode}_sample.csv')


def build_features(text: str) -> tuple:
    """
    Функция для создания признаков в модель
    Разберем принцип работы на примере:
    text: 'казнить нельзя, помиловать.'
    reshaped_text: ['казнить нельзя, помиловать.']
    tokenized_text:[['[SOS]', 'казнить', 'нельзя', ',', 'помил', '#овать', '.', '[EOS]']]
    input_tokens:  [['[SOS]', 'казнить', 'нельзя', 'помил', '#овать', '[EOS]']]  # не содержит target
    input_targets: [[  0,        0,        1,        0,        2,        0   ]]  # помечаем токен ПЕРЕД таргетом
    attention_mask:[[  1,        1,        1,        1,        1,        1   ]]  # нули будут говорить о <pad> - токенах
    words_mask:    [[  1,        1,        1,        0,        1,        1   ]]  # для склейки токенов в одно слово
    :param text: строка с текстом
    :return:
    """
    content = ContentWrapper(max_size=130).fit(text)
    reshaped_text = content.get_split()

    tokenized_text = [tokenizer.tokenize(sent) for sent in reshaped_text]
    # для моделей BERT проставляем токены на начало и конец последовательностей
    tokenized_text = [['[SOS]'] + sentence + ['[EOS]'] for sentence in tokenized_text]

    # удаляем из текста токены с классифицуремыми знаками препинания
    input_tokens = list(
        map(lambda sentence: list(
            filter(
                lambda x: x not in targets.keys(),
                sentence)
        ),
            tokenized_text)
    )

    # переводим токены в индексы
    input_ids = list(map(tokenizer.convert_tokens_to_ids, input_tokens))
    logging.info('input_ids created')

    # находим и кодируем таргет, делаем по ним маску и сдвигаем влево на 1,
    # удаляем из массива с таргетами элементы по индексам из маски. формально задача удалить элемент перед таргетом
    target_tokens = list(map(lambda sentence: np.array([targets.get(x, 0) for x in sentence]), tokenized_text))
    target_tokens_mask = [(target_sentence != 0).astype(int) for target_sentence in target_tokens]
    remove_ids = [np.nonzero(np.roll(mask_sentence, -1)) for mask_sentence in target_tokens_mask]
    input_targets = [np.delete(tokens, ids) for tokens, ids in zip(target_tokens, remove_ids)]
    logging.info('input_targets created')

    # находим цельное токен-слово, делаем сдвиг на 1 влево
    is_full_token = lambda item: 0 if (item[0] == '#') & (item != '#') else 1
    end_token_mask = list(map(lambda sentence: np.array([is_full_token(token) for token in sentence]), input_tokens))
    words_mask = [np.roll(sentence, -1) for sentence in end_token_mask]
    logging.info('words_mask created')

    attention_mask = [np.ones(len(x)) for x in input_ids]
    logging.info('attention_mask created')

    for inp, tar, word, attn in zip(input_ids, input_targets, words_mask, attention_mask):
        assert (np.asarray(inp).shape == tar.shape == word.shape == attn.shape)

    logging.info('checks passed')

    return input_ids, input_targets, words_mask, attention_mask


def download_dataframe(filename: str) -> str:
    """
    Загрузка датасета в память и объединение всех записей в одну строку
    :param filename: имя файла
    :return: строка с текстом
    """
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

    logging.info('dataset loaded')

    inp_ids, inp_tar, w_mask, attn_mask = build_features(dataset)

    comma_dataset = CommaDataset(inp_ids, inp_tar, w_mask, attn_mask)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as f:
        pickle.dump(comma_dataset, f)

    logging.info(f'artefacts saved in {data_processed_dir}')
