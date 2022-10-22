import argparse
import logging
import os
import re
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.build_feature import build_features, ContentWrapper
from src.feature.dataset import CommaDataset
from src.model.train_model import collate_fn


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

decode_map = {0: '', 1: ',', 2: '.'}


def decode(words_original_case: list, y_predict: torch.Tensor, w_mask: torch.Tensor) -> str:
    result = ""
    decode_idx = 0

    for index in range(w_mask.shape[0]):
        if w_mask[index] == 1:
            if words_original_case[decode_idx] not in ('SOS', 'EOS'):
                result += words_original_case[decode_idx]
                result += decode_map[y_predict[index].item()]
                result += ' '
            decode_idx += 1

    result = result.strip()
    return result


def predict(text: str, model: nn.Module) -> str:

    content = ContentWrapper(max_size=150).fit(text)
    reshaped_text = content.get_split()

    input_ids, input_targets, word_mask, attention_mask = build_features(text)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    batch_size = 32
    dataset = CommaDataset(input_ids, input_targets, word_mask, attention_mask)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    result = ""
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x, y, w_mask, att_mask = batch['feature'], batch['target'], batch['word_mask'], batch['attention_mask']
            w_mask = w_mask.view(-1)
            x = x.to(device)
            att_mask = att_mask.to(device)
            y_predict = model(x, att_mask)

            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            print(y_predict)

            string = ' '.join(list(
                map(
                    lambda sent: 'SOS ' + sent + ' EOS',
                    reshaped_text[i * batch_size: i * batch_size + batch_size]
                )
            ))
            words_original_case = list(filter(lambda word: word not in (' ', ''), re.split('(\W)', string)))
            print(words_original_case)
            result += decode(words_original_case, y_predict, w_mask) + ' '
    return result


def load_model(directory_path: str) -> nn.Module:
    """
    функция для загрузки модели
    :param directory_path: имя директории
    :return:
    """
    model_path = directory_path + 'model.torch'
    model = torch.load(model_path)
    return model


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model = load_model(fileDir + config['models'])
    logging.info(f'model loaded')

    text = """
    Это аналитический жанр  Он не столько информационный сколько 
    аналитический Тут осмысливаем события приводим какие-то 
    примеры и исследуем явления"""

    predicted = predict(text, model)
    logging.info('answer is ready:')

    print(predicted)
